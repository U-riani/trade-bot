"""V25 order-book dataset health (pure logic).

Answers operational questions about the collected dataset: is collection
consistent, were there gaps, how much coverage exists, and is there plausibly
enough data to bother analyzing. All functions are pure and take a fixed
reference time, so they are unit-testable without a clock or DB.

This is health/observability only. It does not evaluate predictive value or
profitability; "enough data" here is a coverage estimate, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.market.collector_runtime import CollectorRun
from app.market.features import bucket_start_timestamp
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class GapInfo:
    start_at: datetime
    end_at: datetime
    gap_seconds: float


def detect_gaps(
    times: list[datetime],
    *,
    expected_interval_seconds: float,
    gap_multiplier: float,
) -> list[GapInfo]:
    """Find consecutive snapshots spaced more than expected_interval * multiplier apart.

    A gap means the collector was down, throttled, or restarted. Returns gaps in
    chronological order.
    """
    if expected_interval_seconds <= 0 or gap_multiplier <= 0:
        raise ValueError("expected_interval_seconds and gap_multiplier must be positive")
    if len(times) < 2:
        return []

    threshold = expected_interval_seconds * gap_multiplier
    ordered = sorted(times)
    gaps: list[GapInfo] = []
    for previous, current in zip(ordered, ordered[1:]):
        delta = (current - previous).total_seconds()
        if delta > threshold:
            gaps.append(GapInfo(start_at=previous, end_at=current, gap_seconds=delta))
    return gaps


def hourly_counts_last_24h(
    times: list[datetime],
    *,
    reference_time: datetime,
) -> list[tuple[datetime, int]]:
    """Snapshot counts per hour for the 24 hours ending at reference_time's hour.

    Returns 24 (hour_start, count) pairs, oldest first, including empty hours so
    a stalled collector shows up as zeros.
    """
    current_hour = reference_time.replace(minute=0, second=0, microsecond=0)
    buckets = [current_hour - timedelta(hours=offset) for offset in range(23, -1, -1)]
    counts = {bucket: 0 for bucket in buckets}
    earliest = buckets[0]
    for t in times:
        hour = t.replace(minute=0, second=0, microsecond=0)
        if hour in counts and hour >= earliest:
            counts[hour] += 1
    return [(bucket, counts[bucket]) for bucket in buckets]


def distinct_bucket_count(times: list[datetime], timeframe: str) -> int:
    """How many distinct candle buckets of ``timeframe`` the snapshots touch."""
    target_seconds = timeframe_to_seconds(timeframe)
    return len({bucket_start_timestamp(t, target_seconds) for t in times})


def count_runs_by_status(runs: list[CollectorRun]) -> dict[str, int]:
    """Tally collector runs by status (running/stopped/failed/interrupted/...)."""
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.status] = counts.get(run.status, 0) + 1
    return counts


def coverage_is_enough(distinct_buckets: int, min_samples: int) -> bool:
    """Coverage estimate: are there at least ``min_samples`` distinct buckets?

    This is an upper-bound estimate of usable samples. The real per-horizon
    sample size (which also needs following candles) is computed by
    scripts.order_book_pipeline_status; this is the cheap dataset-side check.
    """
    return distinct_buckets >= min_samples
