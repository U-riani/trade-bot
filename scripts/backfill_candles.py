from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.exchange.binance_rest import BinanceRestClient
from app.market.bootstrap import validate_startup_candles
from app.storage.db import Database
from app.storage.repositories import TradingRepository

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill historical closed candles from Binance REST into PostgreSQL."
    )
    parser.add_argument("--limit", type=int, default=5000, help="Number of recent closed candles to fetch.")
    parser.add_argument("--symbol", type=str, default=None, help="Override SYMBOL from .env.")
    parser.add_argument("--timeframe", type=str, default=None, help="Override TIMEFRAME from .env.")
    parser.add_argument(
        "--market-data-source",
        choices=("production", "testnet"),
        default=None,
        help=(
            "Historical candle source. Defaults to HISTORICAL_MARKET_DATA_SOURCE. "
            "Use production for realistic backtests; testnet data is synthetic."
        ),
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Fetch and validate only. Do not save candles into PostgreSQL.",
    )
    parser.add_argument(
        "--validate-continuity",
        action="store_true",
        help="Validate that fetched candles are fresh enough and continuous before saving.",
    )
    return parser


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    parser = _parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    settings = get_settings()
    symbol = (args.symbol or settings.normalized_symbol).upper()
    timeframe = args.timeframe or settings.timeframe
    market_data_source = args.market_data_source or settings.historical_market_data_source.value
    use_testnet_data = market_data_source == "testnet"
    exchange_id = "binance_testnet" if use_testnet_data else "binance_spot"

    client = BinanceRestClient(testnet=use_testnet_data)
    try:
        candles = await client.get_historical_closed_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=args.limit,
            exchange=exchange_id,
        )
    finally:
        await client.close()

    if not candles:
        raise SystemExit("No candles returned from Binance REST")

    logger.info(
        "historical_candle_backfill_fetched",
        symbol=symbol,
        timeframe=timeframe,
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        requested_limit=args.limit,
        fetched_count=len(candles),
        first_close_time=candles[0].close_time.isoformat(),
        last_close_time=candles[-1].close_time.isoformat(),
        first_price=candles[0].close,
        last_price=candles[-1].close,
    )

    if args.validate_continuity:
        validation = validate_startup_candles(
            candles,
            timeframe=timeframe,
            max_age_seconds=max(settings.startup_candle_max_age_seconds, 24 * 60 * 60),
            gap_tolerance_seconds=settings.startup_candle_gap_tolerance_seconds,
        )
        if validation.can_use:
            logger.info(
                "historical_candle_backfill_validation_ok",
                reason=validation.reason,
                loaded_count=validation.loaded_count,
            )
        else:
            logger.warning(
                "historical_candle_backfill_validation_failed",
                reason=validation.reason,
                loaded_count=validation.loaded_count,
            )

    if args.no_save:
        logger.info("historical_candle_backfill_save_skipped", reason="--no-save")
        return

    if not settings.database_enabled:
        raise SystemExit("DATABASE_ENABLED=false, cannot save backfilled candles")

    db = Database(settings.database_url)
    await db.connect()
    try:
        repository = TradingRepository(db)
        before = await repository.count_candles(
            exchange=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        await repository.save_candles(candles)
        after = await repository.count_candles(
            exchange=exchange_id,
            symbol=symbol,
            timeframe=timeframe,
        )
        logger.info(
            "historical_candle_backfill_saved",
            symbol=symbol,
            timeframe=timeframe,
            market_data_source=market_data_source,
            exchange_id=exchange_id,
            fetched_count=len(candles),
            before_count=before,
            after_count=after,
            inserted_count=max(after - before, 0),
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
