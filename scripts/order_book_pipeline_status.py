"""V24 Phase 1: order-book data pipeline status.

Answers the operational questions that V23 left implicit:
  * Are snapshots being collected, and how often?
  * Do candles in the DB cover the snapshot times yet?
  * How many snapshots actually match candle buckets (vs are too new)?
  * Is there enough data to analyze order-book features yet, and if not, why?

It then prints the exact next command to run. Read-only: it never writes.

Example:
    python -m scripts.order_book_pipeline_status --market-data-source production \
        --symbol BTCUSDT --timeframes 1m,5m,15m
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.backtesting.order_book_pipeline import (
    REASON_NO_FEATURE_ROWS,
    REASON_NO_FORWARD_RETURNS,
    REASON_NO_SNAPSHOTS,
    REASON_NOT_ENOUGH_SAMPLES,
    compute_match_status,
    compute_readiness,
    compute_snapshot_status,
    snapshots_after_latest_candle,
)
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.time import utc_now
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report order-book data pipeline status and next steps.")
    parser.add_argument("--symbol", default=None, help="Symbol. Default from settings (BTCUSDT).")
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--exchange", default=None, help="Exchange id override. Default from --market-data-source.")
    parser.add_argument("--candle-limit", type=int, default=50000, help="Recent candles to load for bucket checks.")
    parser.add_argument("--snapshot-limit", type=int, default=500000)
    parser.add_argument("--min-feature-samples", type=int, default=100, help="Samples needed before analysis is useful.")
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


def _next_step(reason: str) -> str:
    return {
        REASON_NO_SNAPSHOTS: "Run scripts.collect_order_book_features (leave it running).",
        REASON_NO_FEATURE_ROWS: (
            "Run scripts.backfill_candles, wait for buckets to close, then scripts.aggregate_order_book_features."
        ),
        REASON_NO_FORWARD_RETURNS: "Keep collecting; wait for candles AFTER the snapshots to close, then re-aggregate.",
        REASON_NOT_ENOUGH_SAMPLES: "Keep collecting for longer (days), re-aggregate periodically.",
    }.get(reason, "Ready: run scripts.analyze_market_features.")


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()
    reference_time = utc_now()

    _market_data_source, _use_testnet, default_exchange = _resolve_market_data_source(args.market_data_source)
    exchange = args.exchange or default_exchange
    symbol = (args.symbol or settings.normalized_symbol).upper().strip()
    timeframes = _parse_timeframes(args.timeframes)

    source_candles = await _load_candles(args.source, args.candle_limit, args.market_data_source)

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    try:
        snapshots = await repository.load_order_book_snapshots(
            exchange=exchange, symbol=symbol, limit=args.snapshot_limit
        )
        snapshot_status = compute_snapshot_status(snapshots, reference_time=reference_time)
        logger.info(
            "pipeline_snapshot_status",
            exchange=exchange,
            symbol=symbol,
            total=snapshot_status.total,
            first_at=None if snapshot_status.first_at is None else snapshot_status.first_at.isoformat(),
            latest_at=None if snapshot_status.latest_at is None else snapshot_status.latest_at.isoformat(),
            last_1h=snapshot_status.last_1h,
            last_24h=snapshot_status.last_24h,
            avg_interval_seconds=None
            if snapshot_status.avg_interval_seconds is None
            else round(snapshot_status.avg_interval_seconds, 2),
        )

        for timeframe in timeframes:
            candles = resample_candles(source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe)
            latest_candle_close = candles[-1].close_time if candles else None
            feature_rows = await repository.load_market_features(
                exchange=exchange, symbol=symbol, timeframe=timeframe, limit=len(candles) + 10
            )

            match = compute_match_status(
                timeframe=timeframe,
                snapshots=snapshots,
                candles=candles,
                feature_rows=feature_rows,
                latest_candle_close=latest_candle_close,
            )
            logger.info(
                "pipeline_candle_and_match_status",
                timeframe=timeframe,
                latest_candle_close=None if latest_candle_close is None else latest_candle_close.isoformat(),
                total_candle_buckets=match.total_candle_buckets,
                buckets_with_order_book=match.buckets_with_order_book,
                latest_order_book_bucket_close=(
                    None
                    if match.latest_order_book_bucket_close is None
                    else match.latest_order_book_bucket_close.isoformat()
                ),
                matched_snapshots=match.matched_snapshots,
                unmatched_snapshots=match.unmatched_snapshots,
                too_new_snapshots=match.too_new_snapshots,
            )

            readiness = compute_readiness(
                timeframe=timeframe,
                candles=candles,
                feature_rows=feature_rows,
                snapshot_total=snapshot_status.total,
                min_samples=args.min_feature_samples,
            )
            logger.info(
                "pipeline_analysis_readiness",
                timeframe=timeframe,
                sample_size_h1=readiness.sample_size_h1,
                sample_size_h3=readiness.sample_size_h3,
                sample_size_h6=readiness.sample_size_h6,
                ready=readiness.ready,
                reason=readiness.reason,
                next_step=_next_step(readiness.reason),
            )

        too_new_total = snapshots_after_latest_candle(
            snapshots,
            max((c.close_time for c in resample_candles(source_candles, target_timeframe="1m", source_timeframe=args.source_timeframe)), default=None),
        )
        if too_new_total > 0:
            logger.warning(
                "pipeline_snapshots_after_latest_candle",
                count=too_new_total,
                advice="Snapshots exist after latest candle close_time. Run backfill_candles, wait for candles to close, then aggregate again.",
            )
    finally:
        await db.close()

    logger.info("pipeline_status_finished")


if __name__ == "__main__":
    asyncio.run(main())
