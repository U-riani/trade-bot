"""Order-book availability audit for multi-timeframe price setups.

This module does not backtest a strategy.  It answers a narrower data-quality
question before V31: when a valid price-only setup occurs, how recent is the
last *observed* order-book feature and is there a recent prior observation from
which a meaningful order-book change can be measured?

No feature values are filled, interpolated, or fabricated.  The caller can
measure as-of freshness while preserving the actual observation timestamps.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass

from app.backtesting.order_book_strategy import feature_value
from app.backtesting.multitimeframe_pullback_strategy import PullbackSetup
from app.market.features import MarketFeatures


@dataclass(slots=True, frozen=True)
class AsOfOrderBookPair:
    """Latest and prior observed values known at a price-setup timestamp."""

    signal_time: object
    current_time: object
    previous_time: object | None
    current_age_seconds: float
    pair_gap_seconds: float | None
    previous_value: float | None
    current_value: float
    delta: float | None
    is_exact: bool


@dataclass(slots=True, frozen=True)
class AlignmentRecord:
    segment: str
    feature_name: str
    signal_time: object
    asof_pair: AsOfOrderBookPair | None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "segment": self.segment,
            "feature_name": self.feature_name,
            "signal_time": self.signal_time.isoformat(),
            "has_asof_observation": self.asof_pair is not None,
        }
        if self.asof_pair is None:
            return payload
        pair = self.asof_pair
        payload.update(
            {
                "observed_time": pair.current_time.isoformat(),
                "previous_observed_time": pair.previous_time.isoformat() if pair.previous_time is not None else None,
                "current_age_seconds": pair.current_age_seconds,
                "pair_gap_seconds": pair.pair_gap_seconds,
                "previous_value": pair.previous_value,
                "current_value": pair.current_value,
                "delta": pair.delta,
                "is_exact": pair.is_exact,
            }
        )
        return payload


def observed_rows(rows: list[MarketFeatures], *, feature_name: str) -> list[MarketFeatures]:
    """Return rows with an actual observed feature value in chronological order."""

    return sorted(
        (row for row in rows if feature_value(row, feature_name) is not None),
        key=lambda row: row.close_time,
    )


def asof_observed_pair(
    observed: list[MarketFeatures],
    *,
    signal_time: object,
    feature_name: str,
) -> AsOfOrderBookPair | None:
    """Return the latest observed feature at/before ``signal_time`` and its predecessor.

    The returned pair is purely backward-looking.  ``current_age_seconds``
    quantifies how stale the latest observation was at signal time; callers must
    choose a maximum age explicitly rather than treating an old value as live.
    """

    if not observed:
        return None
    close_times = [row.close_time for row in observed]
    index = bisect_right(close_times, signal_time) - 1
    if index < 0:
        return None

    current_row = observed[index]
    current_value = feature_value(current_row, feature_name)
    if current_value is None:  # Defensive: observed_rows should already ensure this.
        return None

    previous_row = observed[index - 1] if index > 0 else None
    previous_value = feature_value(previous_row, feature_name) if previous_row is not None else None
    current_age_seconds = max(0.0, (signal_time - current_row.close_time).total_seconds())
    pair_gap_seconds = None
    delta = None
    if previous_row is not None and previous_value is not None:
        pair_gap_seconds = max(0.0, (current_row.close_time - previous_row.close_time).total_seconds())
        delta = current_value - previous_value

    return AsOfOrderBookPair(
        signal_time=signal_time,
        current_time=current_row.close_time,
        previous_time=previous_row.close_time if previous_row is not None else None,
        current_age_seconds=current_age_seconds,
        pair_gap_seconds=pair_gap_seconds,
        previous_value=previous_value,
        current_value=current_value,
        delta=delta,
        is_exact=current_row.close_time == signal_time,
    )


def alignment_records(
    *,
    segment: str,
    entry_rows: list[MarketFeatures],
    setups: list[PullbackSetup],
    feature_name: str,
    observed_feature_rows: list[MarketFeatures],
) -> list[AlignmentRecord]:
    """Build as-of order-book availability records for price-only setup candles."""

    ordered = sorted(entry_rows, key=lambda row: row.close_time)
    if len(ordered) != len(setups):
        raise ValueError("setups must align one-to-one with entry_rows")
    observed = observed_rows(observed_feature_rows, feature_name=feature_name)

    records: list[AlignmentRecord] = []
    for row, setup in zip(ordered, setups, strict=True):
        if not setup.price_setup:
            continue
        records.append(
            AlignmentRecord(
                segment=segment,
                feature_name=feature_name,
                signal_time=row.close_time,
                asof_pair=asof_observed_pair(
                    observed,
                    signal_time=row.close_time,
                    feature_name=feature_name,
                ),
            )
        )
    return records


def alignment_summary(
    records: list[AlignmentRecord], *, max_age_seconds: list[int], max_pair_gap_seconds: list[int]
) -> dict[str, object]:
    """Summarize exact/as-of availability without turning stale values into signals."""

    summaries: dict[str, object] = {
        "price_setup_candidates": len(records),
        "exact_observation_at_setup": sum(1 for record in records if record.asof_pair is not None and record.asof_pair.is_exact),
        "has_any_asof_observation": sum(1 for record in records if record.asof_pair is not None),
        "has_observed_pair": sum(1 for record in records if record.asof_pair is not None and record.asof_pair.delta is not None),
        "age_seconds": {},
        "pair_gap_seconds": {},
        "eligible_pair_by_age_and_gap": {},
    }
    age_summary = summaries["age_seconds"]
    pair_summary = summaries["pair_gap_seconds"]
    eligible_summary = summaries["eligible_pair_by_age_and_gap"]
    assert isinstance(age_summary, dict)
    assert isinstance(pair_summary, dict)
    assert isinstance(eligible_summary, dict)

    for age in max_age_seconds:
        age_summary[str(age)] = sum(
            1
            for record in records
            if record.asof_pair is not None and record.asof_pair.current_age_seconds <= age
        )
    for gap in max_pair_gap_seconds:
        pair_summary[str(gap)] = sum(
            1
            for record in records
            if record.asof_pair is not None
            and record.asof_pair.pair_gap_seconds is not None
            and record.asof_pair.pair_gap_seconds <= gap
        )
    for age in max_age_seconds:
        for gap in max_pair_gap_seconds:
            eligible_summary[f"age<={age}_pair_gap<={gap}"] = sum(
                1
                for record in records
                if record.asof_pair is not None
                and record.asof_pair.current_age_seconds <= age
                and record.asof_pair.pair_gap_seconds is not None
                and record.asof_pair.pair_gap_seconds <= gap
            )
    return summaries
