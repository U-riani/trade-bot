from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.backtesting.dataset_health import (
    coverage_is_enough,
    count_runs_by_status,
    detect_gaps,
    distinct_bucket_count,
    hourly_counts_last_24h,
)
from app.market.collector_runtime import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    CollectorRun,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_run(run_id: str, status: str) -> CollectorRun:
    return CollectorRun(
        run_id=run_id, exchange="binance_spot", symbol="BTCUSDT", started_at=T0, stopped_at=None,
        status=status, interval_seconds=5.0, depth_limit=100, collected_count=0, failure_count=0,
        last_snapshot_at=None, stop_reason=None,
    )


# --- gap detection ---

def test_detect_gaps_flags_large_interval() -> None:
    times = [T0, T0 + timedelta(seconds=5), T0 + timedelta(seconds=40)]
    gaps = detect_gaps(times, expected_interval_seconds=5.0, gap_multiplier=3.0)  # threshold 15s
    assert len(gaps) == 1
    assert gaps[0].gap_seconds == 35.0
    assert gaps[0].start_at == T0 + timedelta(seconds=5)


def test_detect_gaps_none_when_consistent() -> None:
    times = [T0 + timedelta(seconds=5 * i) for i in range(10)]
    assert detect_gaps(times, expected_interval_seconds=5.0, gap_multiplier=3.0) == []


def test_detect_gaps_too_few() -> None:
    assert detect_gaps([T0], expected_interval_seconds=5.0, gap_multiplier=3.0) == []


def test_detect_gaps_validates_args() -> None:
    with pytest.raises(ValueError):
        detect_gaps([T0, T0], expected_interval_seconds=0, gap_multiplier=3.0)


# --- hourly counts ---

def test_hourly_counts_last_24h() -> None:
    reference = datetime(2026, 1, 1, 12, 30, tzinfo=UTC)
    times = [
        datetime(2026, 1, 1, 12, 5, tzinfo=UTC),   # in 12:00 bucket
        datetime(2026, 1, 1, 11, 50, tzinfo=UTC),  # in 11:00 bucket
        datetime(2025, 12, 31, 10, 0, tzinfo=UTC),  # >24h ago, excluded
    ]
    hourly = hourly_counts_last_24h(times, reference_time=reference)
    assert len(hourly) == 24
    counts = dict(hourly)
    assert counts[datetime(2026, 1, 1, 12, 0, tzinfo=UTC)] == 1
    assert counts[datetime(2026, 1, 1, 11, 0, tzinfo=UTC)] == 1
    assert sum(c for _h, c in hourly) == 2  # the old one is excluded


# --- bucket coverage ---

def test_distinct_bucket_count() -> None:
    times = [T0, T0 + timedelta(seconds=30), T0 + timedelta(minutes=1)]
    assert distinct_bucket_count(times, "1m") == 2  # minute 0 and minute 1
    assert distinct_bucket_count(times, "5m") == 1  # all in one 5m bucket


# --- run status tally ---

def test_count_runs_by_status() -> None:
    runs = [make_run("a", STATUS_RUNNING), make_run("b", STATUS_STOPPED), make_run("c", STATUS_STOPPED),
            make_run("d", STATUS_FAILED)]
    counts = count_runs_by_status(runs)
    assert counts == {STATUS_RUNNING: 1, STATUS_STOPPED: 2, STATUS_FAILED: 1}


def test_coverage_is_enough() -> None:
    assert coverage_is_enough(100, 100) is True
    assert coverage_is_enough(99, 100) is False
