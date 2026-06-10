"""V23 Phase 3: aggregate raw order-book snapshots into candle-aligned features.

Buckets the live order_book_snapshots into 1m / 5m / 15m candle windows and
upserts the per-bucket averages (spread %, imbalance at top 5/10/20, snapshot
count, average top-20 volumes) into the V22 market_features table, keyed by
(exchange, symbol, timeframe, close_time).

A bucket is only written when it actually contains snapshots, so this never
fabricates order-book data for periods that were not observed. Because snapshots
accumulate forward in time, only candle buckets that overlap the collection
window get updated; everything older stays NULL, honestly.

Example:
    python -m scripts.aggregate_order_book_features --market-data-source production \
        --source db --candle-limit 5000 --timeframes 1m,5m,15m
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from collections.abc import Sequence

from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures, bucket_start_timestamp
from app.market.models import Candle
from app.market.order_book import OrderBookSnapshot, aggregate_snapshots
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.timeframe import timeframe_to_seconds
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


def build_order_book_feature_rows(
    candles: list[Candle],
    snapshots: list[OrderBookSnapshot],
    *,
    exchange: str,
    symbol: str,
    timeframe: str,
    min_snapshots: int = 1,
) -> list[MarketFeatures]:
    """Join snapshots to candle buckets and produce upsertable feature rows.

    Pure function (no I/O) so the bucketing is unit-testable. Snapshots are
    grouped by the same UTC-aligned bucket key the resampler uses, then matched
    to each candle by its open_time bucket. Candle buckets with fewer than
    ``min_snapshots`` observations are skipped rather than written thin.
    """
    target_seconds = timeframe_to_seconds(timeframe)
    by_bucket: dict[int, list[OrderBookSnapshot]] = defaultdict(list)
    for snapshot in snapshots:
        by_bucket[bucket_start_timestamp(snapshot.collected_at, target_seconds)].append(snapshot)

    rows: list[MarketFeatures] = []
    for candle in candles:
        key = bucket_start_timestamp(candle.open_time, target_seconds)
        bucket_snapshots = by_bucket.get(key)
        if not bucket_snapshots or len(bucket_snapshots) < min_snapshots:
            continue

        agg = aggregate_snapshots(bucket_snapshots)
        if agg is None:
            continue

        rows.append(
            MarketFeatures(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                open_time=candle.open_time,
                close_time=candle.close_time,
                close_price=candle.close,
                volume=candle.volume,
                order_book_bid_volume=agg.avg_bid_volume_top_20,
                order_book_ask_volume=agg.avg_ask_volume_top_20,
                order_book_imbalance=agg.avg_imbalance_top_10,
                spread_pct=agg.avg_spread_pct,
                imbalance_top_5=agg.avg_imbalance_top_5,
                imbalance_top_10=agg.avg_imbalance_top_10,
                imbalance_top_20=agg.avg_imbalance_top_20,
                order_book_snapshot_count=agg.snapshot_count,
            )
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate raw order-book snapshots into 1m/5m/15m market_features buckets."
    )
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--exchange", default=None, help="Exchange id override. Default from --market-data-source.")
    parser.add_argument("--candle-limit", type=int, default=5000, help="How many recent candles to load.")
    parser.add_argument("--snapshot-limit", type=int, default=200000, help="How many recent snapshots to load.")
    parser.add_argument("--min-snapshots-per-bucket", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Compute and log only; do not upsert.")
    return parser


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

    if args.candle_limit <= 0:
        raise SystemExit("--candle-limit must be positive")
    if args.snapshot_limit <= 0:
        raise SystemExit("--snapshot-limit must be positive")

    source_candles = await _load_candles(args.source, args.candle_limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available for order-book aggregation")

    _market_data_source, _use_testnet, default_exchange = _resolve_market_data_source(args.market_data_source)
    exchange = args.exchange or default_exchange
    symbol = settings.normalized_symbol

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)

    try:
        snapshots = await repository.load_order_book_snapshots(
            exchange=exchange, symbol=symbol, limit=args.snapshot_limit
        )
        logger.info(
            "order_book_aggregation_started",
            exchange=exchange,
            symbol=symbol,
            source_candles=len(source_candles),
            snapshots_loaded=len(snapshots),
            timeframes=args.timeframes,
            dry_run=args.dry_run,
        )
        if not snapshots:
            logger.warning(
                "order_book_aggregation_no_snapshots",
                note="collect snapshots first with scripts.collect_order_book_features",
            )

        for timeframe in _parse_timeframes(args.timeframes):
            candles = resample_candles(
                source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe
            )
            rows = build_order_book_feature_rows(
                candles,
                snapshots,
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                min_snapshots=args.min_snapshots_per_bucket,
            )
            total_snapshots_used = sum(r.order_book_snapshot_count or 0 for r in rows)
            logger.info(
                "order_book_aggregation_buckets",
                timeframe=timeframe,
                candles=len(candles),
                buckets_with_snapshots=len(rows),
                snapshots_used=total_snapshots_used,
            )

            if rows and not args.dry_run:
                upserted = await repository.upsert_market_features_order_book(rows)
                logger.info("order_book_aggregation_upserted", timeframe=timeframe, rows=upserted)
    finally:
        await db.close()

    logger.info("order_book_aggregation_finished")


if __name__ == "__main__":
    asyncio.run(main())
