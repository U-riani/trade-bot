"""V22 Phase 2: build non-price market features into the market_features table.

Pipeline:
  1. Load 1m candles from DB (or REST) for OHLCV/price.
  2. Resample to each requested timeframe (5m, 15m).
  3. Fetch raw 1m klines from Binance to read taker-buy / quote volumes
     (available historically) and aggregate them into the same buckets.
  4. Build feature rows and insert them (duplicates skipped).

HONESTY: order-book features (imbalance / spread) are NOT available historically
from Binance REST, so those columns stay NULL for every historical row. This
script never invents them from price. Pass --with-current-order-book to fetch a
single CURRENT depth snapshot and print its forward-only features, which are
explicitly not written to historical rows.

Example:
    python -m scripts.build_market_features --market-data-source production \
        --source db --limit 50000 --timeframes 5m,15m
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.market.features import (
    aggregate_taker_by_bucket,
    bucket_start_timestamp,
    build_market_features,
    order_book_features_from_depth,
    parse_taker_rows,
)
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.timeframe import timeframe_to_seconds
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build non-price market features from candles + Binance klines.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--timeframes", default="5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--min-candles-per-timeframe", type=int, default=50)
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument(
        "--with-taker",
        dest="with_taker",
        action="store_true",
        default=True,
        help="Fetch taker/quote volumes from Binance klines (default).",
    )
    parser.add_argument(
        "--no-taker",
        dest="with_taker",
        action="store_false",
        help="Skip taker fetch; only volume/price features are stored.",
    )
    parser.add_argument(
        "--with-current-order-book",
        action="store_true",
        help="Also fetch ONE current depth snapshot and print forward-only order-book features.",
    )
    parser.add_argument("--no-save", action="store_true", help="Compute and log only; do not write to DB.")
    return parser


async def _fetch_taker_rows(*, symbol: str, source_timeframe: str, limit: int, use_testnet: bool):
    client = BinanceRestClient(testnet=use_testnet)
    try:
        raw = await client.get_historical_klines(symbol=symbol, interval=source_timeframe, limit=limit)
        rows = parse_taker_rows(raw)
        logger.info("market_features_taker_fetched", fetched=len(raw), parsed=len(rows))
        return rows
    finally:
        await client.close()


async def _print_current_order_book(*, symbol: str, use_testnet: bool) -> None:
    client = BinanceRestClient(testnet=use_testnet)
    try:
        depth = await client.get_order_book(symbol=symbol, limit=100)
        bids = [(float(p), float(q)) for p, q in depth.get("bids", [])]
        asks = [(float(p), float(q)) for p, q in depth.get("asks", [])]
        features = order_book_features_from_depth(bids, asks)
        logger.info(
            "current_order_book_snapshot_forward_only",
            note="NOT written to historical rows; live snapshot only",
            bid_volume=features.bid_volume,
            ask_volume=features.ask_volume,
            imbalance=round(features.imbalance, 6),
            spread_pct=None if features.spread_pct is None else round(features.spread_pct, 6),
        )
    except Exception as exc:  # noqa: BLE001 - research script, log and continue
        logger.warning("current_order_book_snapshot_failed", error=str(exc))
    finally:
        await client.close()


def _parse_timeframes(value: str) -> list[str]:
    timeframes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in timeframes:
            timeframes.append(item)
    if not timeframes:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return timeframes


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    source_candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available to build features")

    market_data_source, use_testnet_data, exchange_id = _resolve_market_data_source(args.market_data_source)
    logger.info(
        "market_features_started",
        source=args.source,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        timeframes=args.timeframes,
        with_taker=args.with_taker,
        save=not args.no_save,
    )

    taker_rows = []
    if args.with_taker:
        try:
            taker_rows = await _fetch_taker_rows(
                symbol=settings.normalized_symbol,
                source_timeframe=args.source_timeframe,
                limit=args.limit,
                use_testnet=use_testnet_data,
            )
        except Exception as exc:  # noqa: BLE001 - degrade gracefully without taker
            logger.warning("market_features_taker_fetch_failed", error=str(exc))
            taker_rows = []

    db: Database | None = None
    repository: TradingRepository | None = None
    if not args.no_save:
        db = Database(settings.database_url)
        await db.connect()
        repository = TradingRepository(db)

    try:
        for timeframe in _parse_timeframes(args.timeframes):
            target_candles = resample_candles(
                source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe
            )
            if len(target_candles) < args.min_candles_per_timeframe:
                logger.warning("market_features_timeframe_skipped", timeframe=timeframe, candles=len(target_candles))
                continue

            taker_by_close_time = {}
            if taker_rows:
                target_seconds = timeframe_to_seconds(timeframe)
                bucket_map = aggregate_taker_by_bucket(taker_rows, target_timeframe=timeframe)
                for candle in target_candles:
                    key = bucket_start_timestamp(candle.open_time, target_seconds)
                    agg = bucket_map.get(key)
                    if agg is not None:
                        taker_by_close_time[candle.close_time] = agg

            rows = build_market_features(
                target_candles,
                exchange=exchange_id,
                taker_by_close_time=taker_by_close_time,
            )

            taker_available = sum(1 for row in rows if row.taker_buy_ratio is not None)
            logger.info(
                "market_features_generated",
                timeframe=timeframe,
                features=len(rows),
                taker_available=taker_available,
                taker_missing=len(rows) - taker_available,
                order_book_available=0,
                order_book_note="order-book features unavailable historically (NULL)",
            )

            if repository is not None:
                inserted = await repository.insert_market_features(rows)
                total = await repository.count_market_features(
                    exchange=exchange_id, symbol=settings.normalized_symbol, timeframe=timeframe
                )
                logger.info(
                    "market_features_inserted",
                    timeframe=timeframe,
                    inserted=inserted,
                    skipped_existing=len(rows) - inserted,
                    table_total=total,
                )
    finally:
        if db is not None:
            await db.close()

    if args.with_current_order_book:
        await _print_current_order_book(symbol=settings.normalized_symbol, use_testnet=use_testnet_data)

    logger.info("market_features_finished")


if __name__ == "__main__":
    asyncio.run(main())
