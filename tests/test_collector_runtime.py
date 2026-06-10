from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.market.collector_runtime import (
    apply_jitter,
    average_interval_seconds,
    compute_backoff_seconds,
    default_run_id,
    should_emit,
    should_stop_on_failures,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


# --- run id ---

def test_default_run_id_uses_provided() -> None:
    assert default_run_id("my-run") == "my-run"
    assert default_run_id("  spaced  ") == "spaced"


def test_default_run_id_generates_when_missing() -> None:
    a = default_run_id(None)
    b = default_run_id("   ")
    assert a.startswith("obcollect_") and len(a) > len("obcollect_")
    assert b.startswith("obcollect_")
    assert a != b  # unique


# --- backoff ---

def test_compute_backoff_exponential_and_capped() -> None:
    assert compute_backoff_seconds(0, initial=1.0, maximum=60.0) == 0.0
    assert compute_backoff_seconds(1, initial=1.0, maximum=60.0) == 1.0
    assert compute_backoff_seconds(2, initial=1.0, maximum=60.0) == 2.0
    assert compute_backoff_seconds(3, initial=1.0, maximum=60.0) == 4.0
    assert compute_backoff_seconds(10, initial=1.0, maximum=60.0) == 60.0  # capped


def test_apply_jitter_equal_jitter() -> None:
    assert apply_jitter(10.0, 0.0) == 5.0  # half fixed
    assert apply_jitter(10.0, 1.0) == 10.0
    assert apply_jitter(10.0, 0.5) == 7.5


# --- stop condition ---

def test_should_stop_on_failures() -> None:
    assert should_stop_on_failures(2, 3) is False
    assert should_stop_on_failures(3, 3) is True
    assert should_stop_on_failures(4, 3) is True
    assert should_stop_on_failures(100, 0) is False  # 0 disables
    assert should_stop_on_failures(100, -1) is False


# --- heartbeat / stats cadence ---

def test_should_emit() -> None:
    assert should_emit(5, 5) is True
    assert should_emit(10, 5) is True
    assert should_emit(4, 5) is False
    assert should_emit(0, 5) is False
    assert should_emit(5, 0) is False  # disabled


def test_average_interval_seconds() -> None:
    assert average_interval_seconds(None, T0, 1) is None
    assert average_interval_seconds(T0, T0, 1) is None  # need >= 2
    assert average_interval_seconds(T0, T0 + timedelta(seconds=40), 3) == 20.0
    assert average_interval_seconds(T0, T0, 3) == 0.0  # zero span
