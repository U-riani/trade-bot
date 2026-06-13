from __future__ import annotations

import pytest

from app.backtesting.feature_analysis import BucketStat, FeatureHorizonAnalysis
from scripts.compare_feature_groups import _candidate_row, _group_summary_row


def _analysis(feature: str, sample_size: int, correlation: float | None, spread: float = 1.0) -> FeatureHorizonAnalysis:
    return FeatureHorizonAnalysis(
        feature=feature,
        horizon=3,
        sample_size=sample_size,
        correlation=correlation,
        buckets=[
            BucketStat(
                bucket=0,
                lower=0.0,
                upper=1.0,
                count=max(sample_size // 2, 1),
                avg_forward_return_pct=0.0,
                win_rate=0.5,
            ),
            BucketStat(
                bucket=1,
                lower=1.0,
                upper=2.0,
                count=max(sample_size // 2, 1),
                avg_forward_return_pct=spread,
                win_rate=0.6,
            ),
        ],
    )


def test_group_summary_ignores_low_sample_feature_when_picking_best() -> None:
    low_sample = _analysis("tiny_order_book", sample_size=46, correlation=0.99, spread=5.0)
    enough_sample = _analysis("trade_pressure", sample_size=250, correlation=0.12, spread=0.4)

    row = _group_summary_row(
        timeframe="5m",
        horizon=3,
        group="combined_available",
        group_analyses=[low_sample, enough_sample],
        min_feature_samples=100,
    )

    assert row["best_abs_correlation_feature"] == "trade_pressure"
    assert row["best_abs_correlation_sample_size"] == 250
    assert row["best_quantile_spread_feature"] == "trade_pressure"
    assert row["best_quantile_spread_sample_size"] == 250
    assert row["max_sample_size"] == 250
    assert row["min_feature_sample_size"] == 46
    assert row["warning"] == "mixed_sample_sizes"


def test_group_summary_marks_group_too_small_when_nothing_is_usable() -> None:
    row = _group_summary_row(
        timeframe="15m",
        horizon=6,
        group="live_order_book",
        group_analyses=[_analysis("spread_pct", sample_size=17, correlation=-0.8, spread=10.0)],
        min_feature_samples=100,
    )

    assert row["best_abs_correlation_feature"] is None
    assert row["best_quantile_spread_feature"] is None
    assert row["usable_feature_count"] == 0
    assert row["warning"] == "sample_too_small"


def test_candidate_row_explains_weak_and_small_features() -> None:
    small = _candidate_row(
        timeframe="15m",
        horizon=1,
        source_group="live_order_book",
        analysis=_analysis("spread_pct", sample_size=17, correlation=0.8, spread=5.0),
        min_feature_samples=100,
        min_abs_correlation=0.05,
        min_quantile_spread=0.02,
    )
    weak = _candidate_row(
        timeframe="1m",
        horizon=1,
        source_group="candle_only",
        analysis=_analysis("body_pct", sample_size=5000, correlation=0.001, spread=0.001),
        min_feature_samples=100,
        min_abs_correlation=0.05,
        min_quantile_spread=0.02,
    )
    candidate = _candidate_row(
        timeframe="1m",
        horizon=6,
        source_group="agg_trade_pressure",
        analysis=_analysis("taker_buy_quote_ratio_trades", sample_size=790, correlation=-0.12, spread=0.03),
        min_feature_samples=100,
        min_abs_correlation=0.05,
        min_quantile_spread=0.02,
    )

    assert small["reason"] == "not_enough_samples"
    assert small["usable_for_strategy_research"] is False
    assert weak["reason"] == "weak_signal"
    assert weak["usable_for_strategy_research"] is True
    assert candidate["reason"] == "candidate_research_only"
    assert candidate["usable_for_strategy_research"] is True
    assert candidate["abs_correlation"] == pytest.approx(0.12)
