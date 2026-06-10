"""V25 Phase 3: order-book dataset health report.

Read-only. Reports collection consistency, collector run outcomes, gaps,
recent throughput, and an estimate of how much usable coverage exists, so you
can answer: is the collector running consistently, did it stop/fail, are there
large gaps, and is there plausibly enough data to analyze yet.

This reports dataset HEALTH, not predictive value or profitability.

Example:
    python -m scripts.order_book_dataset_health --market-data-source production \
        --symbol BTCUSDT --expected-interval-seconds 5 --gap-multiplier 3
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.backtesting.dataset_health import (
    count_runs_by_status,
    coverage_is_enough,
    detect_gaps,
    distinct_bucket_count,
    hourly_counts_last_24h,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.collector_runtime import STATUS_RUNNING
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from app.utils.time import utc_now
from scripts.backtest_strategy import _resolve_market_data_source

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report order-book dataset health (read-only).")
    parser.add_argument("--symbol", default=None, help="Symbol. Default from settings (BTCUSDT).")
    parser.add_argument("--exchange", default=None, help="Exchange id override. Default from --market-data-source.")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--expected-interval-seconds", type=float, default=5.0)
    parser.add_argument("--gap-multiplier", type=float, default=3.0)
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--snapshot-limit", type=int, default=2000000)
    parser.add_argument("--run-limit", type=int, default=1000)
    return parser


def _parse_timeframes(value: str) -> list[str]:
    timeframes: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in timeframes:
            timeframes.append(item)
    return timeframes or ["1m", "5m", "15m"]


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()
    reference_time = utc_now()

    if args.expected_interval_seconds <= 0:
        raise SystemExit("--expected-interval-seconds must be positive")
    if args.gap_multiplier <= 0:
        raise SystemExit("--gap-multiplier must be positive")

    _market_data_source, _use_testnet, default_exchange = _resolve_market_data_source(args.market_data_source)
    exchange = args.exchange or default_exchange
    symbol = (args.symbol or settings.normalized_symbol).upper().strip()

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    try:
        snapshots = await repository.load_order_book_snapshots(exchange=exchange, symbol=symbol, limit=args.snapshot_limit)
        runs = await repository.load_collector_runs(exchange=exchange, symbol=symbol, limit=args.run_limit)

        times = [s.collected_at for s in snapshots]
        first_at = min(times) if times else None
        latest_at = max(times) if times else None

        run_status_counts = count_runs_by_status(runs)
        logger.info(
            "dataset_health_runs",
            exchange=exchange,
            symbol=symbol,
            total_runs=len(runs),
            running=run_status_counts.get("running", 0),
            stopped=run_status_counts.get("stopped", 0),
            failed=run_status_counts.get("failed", 0),
            interrupted=run_status_counts.get("interrupted", 0),
        )

        logger.info(
            "dataset_health_snapshots",
            total_snapshots=len(snapshots),
            first_at=None if first_at is None else first_at.isoformat(),
            latest_at=None if latest_at is None else latest_at.isoformat(),
        )

        gaps = detect_gaps(
            times, expected_interval_seconds=args.expected_interval_seconds, gap_multiplier=args.gap_multiplier
        )
        largest_gap = max((g.gap_seconds for g in gaps), default=0.0)
        logger.info(
            "dataset_health_gaps",
            expected_interval_seconds=args.expected_interval_seconds,
            gap_multiplier=args.gap_multiplier,
            gap_threshold_seconds=args.expected_interval_seconds * args.gap_multiplier,
            gap_count=len(gaps),
            largest_gap_seconds=round(largest_gap, 1),
        )

        hourly = hourly_counts_last_24h(times, reference_time=reference_time)
        total_last_24h = sum(count for _hour, count in hourly)
        active_hours = sum(1 for _hour, count in hourly if count > 0)
        logger.info(
            "dataset_health_last_24h",
            total_last_24h=total_last_24h,
            active_hours=active_hours,
            empty_hours=24 - active_hours,
            per_hour=",".join(str(count) for _hour, count in hourly),
        )

        any_running = run_status_counts.get(STATUS_RUNNING, 0) > 0
        for timeframe in _parse_timeframes(args.timeframes):
            buckets = distinct_bucket_count(times, timeframe)
            enough = coverage_is_enough(buckets, args.min_feature_samples)
            logger.info(
                "dataset_health_coverage",
                timeframe=timeframe,
                distinct_buckets=buckets,
                min_feature_samples=args.min_feature_samples,
                enough_estimated=enough,
                note=(
                    "coverage estimate only; run order_book_pipeline_status for true per-horizon readiness"
                ),
            )

        logger.info(
            "dataset_health_summary",
            collector_currently_running=any_running,
            total_snapshots=len(snapshots),
            gap_count=len(gaps),
            ready_for_analysis_estimate=False
            if not times
            else coverage_is_enough(distinct_bucket_count(times, "5m"), args.min_feature_samples),
            reminder="V25 measures data health only; it does not evaluate profitability.",
        )
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
