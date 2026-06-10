"""V24 order-book pipeline observability (pure logic).

The V23 collector works, but it is easy to be confused about why analysis still
shows nothing: snapshots are collected in real time, while candles must be
backfilled and *closed* before a snapshot's bucket can be aggregated and have
forward returns. These pure functions make that state legible, so the operator
knows whether to keep collecting, run a backfill, or actually analyze.

No I/O here. The script loads snapshots / candles / feature rows from the DB and
passes them in, so every calculation is unit-testable with synthetic data and a
fixed reference time (no hidden clock).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.market.features import bucket_start_timestamp
from app.market.models import Candle
from app.market.order_book import OrderBookSnapshot
from app.utils.timeframe import timeframe_to_seconds

# Reasons (kept as constants so the script and tests agree on exact strings).
REASON_NO_SNAPSHOTS = "no_snapshots"
REASON_NO_FEATURE_ROWS = "no_order_book_feature_rows_run_backfill_then_aggregate"
REASON_NO_FORWARD_RETURNS = "feature_rows_exist_but_no_forward_returns_yet"
REASON_NOT_ENOUGH_SAMPLES = "not_enough_samples"
REASON_READY = "ready"


@dataclass(slots=True, frozen=True)
class SnapshotStatus:
    total: int
    first_at: datetime | None
    latest_at: datetime | None
    last_1h: int
    last_24h: int
    avg_interval_seconds: float | None


def compute_snapshot_status(
    snapshots: list[OrderBookSnapshot],
    *,
    reference_time: datetime,
) -> SnapshotStatus:
    """Summarize collection cadence. ``reference_time`` anchors the rolling windows."""
    total = len(snapshots)
    if total == 0:
        return SnapshotStatus(0, None, None, 0, 0, None)

    times = sorted(s.collected_at for s in snapshots)
    first_at = times[0]
    latest_at = times[-1]
    last_1h = sum(1 for t in times if t >= reference_time - timedelta(hours=1))
    last_24h = sum(1 for t in times if t >= reference_time - timedelta(hours=24))

    avg_interval: float | None = None
    if total >= 2:
        span = (latest_at - first_at).total_seconds()
        avg_interval = span / (total - 1) if span > 0 else 0.0

    return SnapshotStatus(total, first_at, latest_at, last_1h, last_24h, avg_interval)


@dataclass(slots=True, frozen=True)
class TimeframeMatchStatus:
    timeframe: str
    total_candle_buckets: int
    buckets_with_order_book: int
    latest_order_book_bucket_close: datetime | None
    matched_snapshots: int
    unmatched_snapshots: int
    too_new_snapshots: int


def _bucket_close_time(bucket_start_ts: int, target_seconds: int) -> datetime:
    return datetime.fromtimestamp(bucket_start_ts + target_seconds, tz=timezone.utc)


def compute_match_status(
    *,
    timeframe: str,
    snapshots: list[OrderBookSnapshot],
    candles: list[Candle],
    feature_rows,
    latest_candle_close: datetime | None,
) -> TimeframeMatchStatus:
    """Classify each snapshot as matched / unmatched / too-new for this timeframe.

    * matched: the snapshot's bucket has a candle in the DB (it can be aggregated)
    * too_new: no candle yet AND the bucket closes after the latest candle close
      (candle not backfilled/closed yet) -- the usual cause of "nothing matched"
    * unmatched: everything not matched (too_new is a subset, reported separately)
    """
    target_seconds = timeframe_to_seconds(timeframe)
    candle_bucket_keys = {bucket_start_timestamp(c.open_time, target_seconds) for c in candles}

    matched = 0
    too_new = 0
    for snapshot in snapshots:
        key = bucket_start_timestamp(snapshot.collected_at, target_seconds)
        if key in candle_bucket_keys:
            matched += 1
        elif latest_candle_close is not None and _bucket_close_time(key, target_seconds) > latest_candle_close:
            too_new += 1
    unmatched = len(snapshots) - matched

    ob_rows = [r for r in feature_rows if (r.order_book_snapshot_count or 0) > 0]
    latest_ob_close = max((r.close_time for r in ob_rows), default=None)

    return TimeframeMatchStatus(
        timeframe=timeframe,
        total_candle_buckets=len(candles),
        buckets_with_order_book=len(ob_rows),
        latest_order_book_bucket_close=latest_ob_close,
        matched_snapshots=matched,
        unmatched_snapshots=unmatched,
        too_new_snapshots=too_new,
    )


@dataclass(slots=True, frozen=True)
class TimeframeReadiness:
    timeframe: str
    sample_size_h1: int
    sample_size_h3: int
    sample_size_h6: int
    ready: bool
    reason: str


def order_book_sample_size(candles: list[Candle], feature_rows, horizon: int) -> int:
    """How many order-book buckets have at least ``horizon`` following candles.

    Only those can contribute a forward return at that horizon. Counts feature
    rows whose order-book imbalance is present and whose candle index leaves room
    for ``horizon`` more candles.
    """
    index_by_close = {c.close_time: i for i, c in enumerate(candles)}
    n = len(candles)
    count = 0
    for row in feature_rows:
        if row.order_book_imbalance is None and (row.order_book_snapshot_count or 0) == 0:
            continue
        index = index_by_close.get(row.close_time)
        if index is None:
            continue
        if index + horizon < n:
            count += 1
    return count


def compute_readiness(
    *,
    timeframe: str,
    candles: list[Candle],
    feature_rows,
    snapshot_total: int,
    min_samples: int,
    horizons: tuple[int, int, int] = (1, 3, 6),
) -> TimeframeReadiness:
    """Decide whether order-book analysis is meaningful yet, and why not."""
    h1, h3, h6 = horizons
    s1 = order_book_sample_size(candles, feature_rows, h1)
    s3 = order_book_sample_size(candles, feature_rows, h3)
    s6 = order_book_sample_size(candles, feature_rows, h6)

    ob_rows = [r for r in feature_rows if (r.order_book_snapshot_count or 0) > 0]

    if snapshot_total == 0:
        reason = REASON_NO_SNAPSHOTS
        ready = False
    elif not ob_rows:
        reason = REASON_NO_FEATURE_ROWS
        ready = False
    elif s6 == 0:
        reason = REASON_NO_FORWARD_RETURNS
        ready = False
    elif s6 < min_samples:
        reason = REASON_NOT_ENOUGH_SAMPLES
        ready = False
    else:
        reason = REASON_READY
        ready = True

    return TimeframeReadiness(timeframe, s1, s3, s6, ready, reason)


def snapshots_after_latest_candle(
    snapshots: list[OrderBookSnapshot],
    latest_candle_close: datetime | None,
) -> int:
    """Count snapshots collected after the latest candle close (need backfill)."""
    if latest_candle_close is None:
        return len(snapshots)
    return sum(1 for s in snapshots if s.collected_at > latest_candle_close)
