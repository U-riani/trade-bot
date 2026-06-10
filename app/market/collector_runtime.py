"""V25 collector runtime helpers and run model (pure, no I/O).

These small deterministic functions hold the reliability logic for the
long-running order-book collector — backoff, run-id defaulting, heartbeat/stats
cadence, and the failure stop condition — so the loop in the script stays thin
and the behavior is unit-testable without sleeping, networking, or a clock.

Nothing here trades or evaluates anything. It only governs how the collector
behaves while gathering data over days/weeks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

# Run status values (constants so script, repository, and tests agree exactly).
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"
STATUS_INTERRUPTED = "interrupted"


def default_run_id(provided: str | None) -> str:
    """Return the provided run id, or generate a fresh prefixed one.

    A whitespace-only id is treated as not provided.
    """
    if provided is not None and provided.strip():
        return provided.strip()
    return f"obcollect_{uuid4().hex[:16]}"


def compute_backoff_seconds(consecutive_failures: int, *, initial: float, maximum: float) -> float:
    """Exponential backoff: initial * 2^(n-1), capped at maximum.

    Returns 0.0 when there are no consecutive failures (collect at normal pace).
    """
    if consecutive_failures <= 0:
        return 0.0
    value = initial * (2 ** (consecutive_failures - 1))
    return min(value, maximum)


def apply_jitter(value: float, rand: float) -> float:
    """Equal jitter: keep half the delay fixed, randomize the other half.

    ``rand`` is a value in [0, 1) (e.g. random.random()). Pure so tests pass a
    fixed value. Spreads retries so many collectors don't hammer in lockstep.
    """
    half = value / 2.0
    return half + half * rand


def should_stop_on_failures(consecutive_failures: int, max_failures: int) -> bool:
    """Stop after ``max_failures`` consecutive failures. 0 (or less) disables stopping."""
    if max_failures <= 0:
        return False
    return consecutive_failures >= max_failures


def should_emit(count: int, every: int) -> bool:
    """True when ``count`` is a positive multiple of ``every`` (every<=0 disables)."""
    if every <= 0 or count <= 0:
        return False
    return count % every == 0


def average_interval_seconds(
    first_at: datetime | None,
    latest_at: datetime | None,
    count: int,
) -> float | None:
    """Mean seconds between snapshots, or None when fewer than two exist."""
    if first_at is None or latest_at is None or count < 2:
        return None
    span = (latest_at - first_at).total_seconds()
    return span / (count - 1) if span > 0 else 0.0


@dataclass(slots=True, frozen=True)
class CollectorRun:
    """One row of order_book_collector_runs (id/created_at/updated_at are DB-managed)."""

    run_id: str
    exchange: str
    symbol: str
    started_at: datetime
    stopped_at: datetime | None
    status: str
    interval_seconds: float
    depth_limit: int
    collected_count: int
    failure_count: int
    last_snapshot_at: datetime | None
    stop_reason: str | None
