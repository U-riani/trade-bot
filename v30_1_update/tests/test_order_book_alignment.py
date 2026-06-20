from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.backtesting.multitimeframe_pullback_strategy import PullbackSetup
from app.backtesting.order_book_alignment import alignment_records, alignment_summary, asof_observed_pair, observed_rows
from app.market.features import MarketFeatures


def _row(index: int, *, imbalance: float | None) -> MarketFeatures:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        close_price=100 + index,
        volume=1.0,
        imbalance_top_20=imbalance,
    )


def test_asof_pair_is_backward_looking_and_reports_age() -> None:
    rows = [_row(0, imbalance=-0.2), _row(2, imbalance=0.3), _row(4, imbalance=0.6)]
    observed = observed_rows(rows, feature_name="imbalance_top_20")
    signal_time = _row(5, imbalance=None).close_time

    pair = asof_observed_pair(observed, signal_time=signal_time, feature_name="imbalance_top_20")

    assert pair is not None
    assert pair.current_time == rows[2].close_time
    assert pair.previous_time == rows[1].close_time
    assert pair.current_age_seconds == 60.0
    assert pair.pair_gap_seconds == 120.0
    assert pair.delta == 0.3
    assert not pair.is_exact


def test_alignment_records_only_include_price_setups() -> None:
    rows = [_row(0, imbalance=-0.2), _row(1, imbalance=0.1), _row(2, imbalance=0.4)]
    setups = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "price_setup"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    records = alignment_records(
        segment="validation",
        entry_rows=rows,
        setups=setups,
        feature_name="imbalance_top_20",
        observed_feature_rows=rows,
    )

    assert len(records) == 1
    assert records[0].asof_pair is not None
    assert records[0].asof_pair.is_exact


def test_alignment_summary_counts_fresh_pairs() -> None:
    rows = [_row(0, imbalance=-0.2), _row(1, imbalance=0.2), _row(3, imbalance=0.4)]
    setups = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "price_setup"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    records = alignment_records(
        segment="validation",
        entry_rows=rows,
        setups=setups,
        feature_name="imbalance_top_20",
        observed_feature_rows=rows,
    )
    summary = alignment_summary(records, max_age_seconds=[60, 120], max_pair_gap_seconds=[60, 120])

    assert summary["price_setup_candidates"] == 1
    assert summary["exact_observation_at_setup"] == 1
    assert summary["has_observed_pair"] == 1
    assert summary["age_seconds"]["60"] == 1
    assert summary["eligible_pair_by_age_and_gap"]["age<=60_pair_gap<=60"] == 1
