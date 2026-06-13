"""V26: backfill historical Binance aggregate-trade pressure features.

This script uses Binance aggTrades, not order-book depth. It enriches existing
market_features rows with historical taker-pressure features while live
order-book data continues to accumulate forward.

No trading, no strategy, no profitability claim.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.market.features import MarketFeatures, bucket_start_timestamp
from app.market.trade_pressure import aggregate_trade_pressure_by_bucket, parse_agg_trades
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.time import utc_now
from app.utils.timeframe import timeframe_to_seconds
from scripts.backtest_strategy import _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill historical aggregate-trade pressure features.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--start", default=None, help="UTC ISO start, e.g. 2026-06-13T00:00:00+00:00")
    parser.add_argument("--end", default=None, help="UTC ISO end. Default: now.")
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--candle-limit", type=int, default=50000)
    parser.add_argument("--max-requests", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true", help="Fetch/compute only; do not save.")
    parser.add_argument("--no-save", action="store_true", help="Alias for --dry-run.")
    return parser


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_timeframes(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in result:
            result.append(item)
    if not result:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return result


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.candle_limit <= 0:
        raise SystemExit("--candle-limit must be positive")
    if args.lookback_hours <= 0 and args.start is None:
        raise SystemExit("--lookback-hours must be positive when --start is omitted")

    market_data_source, use_testnet_data, exchange_id = _resolve_market_data_source(args.market_data_source)
    symbol = (args.symbol or settings.normalized_symbol).upper().strip()
    end_at = _parse_dt(args.end) or utc_now()
    start_at = _parse_dt(args.start) or (end_at - timedelta(hours=args.lookback_hours))
    if start_at >= end_at:
        raise SystemExit("start must be before end")

    logger.info(
        "trade_pressure_backfill_started",
        symbol=symbol,
        exchange=exchange_id,
        market_data_source=market_data_source,
        start=start_at.isoformat(),
        end=end_at.isoformat(),
        timeframes=args.timeframes,
        dry_run=args.dry_run or args.no_save,
    )

    client = BinanceRestClient(testnet=use_testnet_data)
    try:
        raw_trades = await client.get_historical_agg_trades(
            symbol=symbol,
            start_time_ms=_ms(start_at),
            end_time_ms=_ms(end_at),
            max_requests=args.max_requests,
        )
    finally:
        await client.close()

    trades = parse_agg_trades(raw_trades)
    logger.info("trade_pressure_trades_fetched", raw=len(raw_trades), parsed=len(trades))

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    try:
        source_candles = await repository.load_recent_candles(
            exchange=exchange_id,
            symbol=symbol,
            timeframe=args.source_timeframe,
            limit=args.candle_limit,
        )
        source_candles = [c for c in source_candles if start_at <= c.open_time <= end_at]
        if not source_candles:
            raise SystemExit("No source candles in requested window. Run backfill_candles first.")

        for timeframe in _parse_timeframes(args.timeframes):
            if timeframe == args.source_timeframe:
                candles = source_candles
            else:
                candles = resample_candles(
                    source_candles,
                    target_timeframe=timeframe,
                    source_timeframe=args.source_timeframe,
                )
            if not candles:
                logger.warning("trade_pressure_timeframe_skipped", timeframe=timeframe, reason="no_candles")
                continue

            target_seconds = timeframe_to_seconds(timeframe)
            by_bucket = aggregate_trade_pressure_by_bucket(trades, target_timeframe=timeframe)
            rows: list[MarketFeatures] = []
            matched = 0
            for candle in candles:
                bucket_key = bucket_start_timestamp(candle.open_time, target_seconds)
                pressure = by_bucket.get(bucket_key)
                if pressure is None:
                    continue
                matched += 1
                rows.append(
                    MarketFeatures(
                        exchange=exchange_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        open_time=candle.open_time,
                        close_time=candle.close_time,
                        close_price=candle.close,
                        volume=candle.volume,
                        trade_count=pressure.trade_count,
                        taker_buy_trade_count=pressure.taker_buy_trade_count,
                        taker_sell_trade_count=pressure.taker_sell_trade_count,
                        taker_buy_base_volume_trades=pressure.taker_buy_base_volume,
                        taker_sell_base_volume_trades=pressure.taker_sell_base_volume,
                        taker_buy_quote_volume_trades=pressure.taker_buy_quote_volume,
                        taker_sell_quote_volume_trades=pressure.taker_sell_quote_volume,
                        taker_net_base_volume=pressure.taker_net_base_volume,
                        taker_net_quote_volume=pressure.taker_net_quote_volume,
                        taker_buy_trade_ratio=pressure.taker_buy_trade_ratio,
                        taker_buy_base_ratio_trades=pressure.taker_buy_base_ratio,
                        taker_buy_quote_ratio_trades=pressure.taker_buy_quote_ratio,
                        avg_trade_quote_size=pressure.avg_trade_quote_size,
                        trade_count_intensity=pressure.trade_count_intensity,
                        quote_volume_intensity=pressure.quote_volume_intensity,
                    )
                )

            logger.info(
                "trade_pressure_timeframe_ready",
                timeframe=timeframe,
                candles=len(candles),
                buckets=len(by_bucket),
                matched_candles=matched,
                save=not (args.dry_run or args.no_save),
            )
            if rows and not (args.dry_run or args.no_save):
                saved = await repository.upsert_market_features_trade_pressure(rows)
                logger.info("trade_pressure_saved", timeframe=timeframe, rows=saved)
    finally:
        await db.close()

    logger.info("trade_pressure_backfill_finished")


if __name__ == "__main__":
    asyncio.run(main())
