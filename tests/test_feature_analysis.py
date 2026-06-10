from __future__ import annotations

import pytest

from app.backtesting.feature_analysis import (
    analyze_feature,
    forward_returns,
    pearson_correlation,
    quantile_buckets,
)


def test_forward_returns_basic() -> None:
    closes = [100.0, 110.0, 121.0]
    rets = forward_returns(closes, horizon=1)
    assert rets[0] == pytest.approx(10.0)
    assert rets[1] == pytest.approx(10.0)
    assert rets[2] is None  # no candle ahead


def test_forward_returns_multi_horizon() -> None:
    closes = [100.0, 105.0, 110.0, 121.0]
    rets = forward_returns(closes, horizon=3)
    assert rets[0] == pytest.approx(21.0)  # 100 -> 121
    assert rets[1] is None


def test_pearson_perfect_positive() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    assert pearson_correlation(xs, ys) == pytest.approx(1.0)


def test_pearson_perfect_negative() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [4.0, 3.0, 2.0, 1.0]
    assert pearson_correlation(xs, ys) == pytest.approx(-1.0)


def test_pearson_undefined_cases() -> None:
    assert pearson_correlation([1.0], [2.0]) is None  # too few
    assert pearson_correlation([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) is None  # zero variance in x


def test_quantile_buckets_split_and_stats() -> None:
    # feature ascending, returns mirror it: higher feature -> higher return.
    pairs = [(float(i), float(i)) for i in range(10)]
    buckets = quantile_buckets(pairs, num_buckets=5)
    assert len(buckets) == 5
    assert [b.count for b in buckets] == [2, 2, 2, 2, 2]
    # lowest bucket has the lowest avg return, highest bucket the highest.
    assert buckets[0].avg_forward_return_pct < buckets[-1].avg_forward_return_pct
    # win_rate counts strictly-positive returns; bucket 0 holds 0 and 1 -> one win.
    assert buckets[0].win_rate == 0.5


def test_quantile_buckets_too_few_returns_empty() -> None:
    assert quantile_buckets([(1.0, 1.0)], num_buckets=5) == []


def test_analyze_feature_drops_missing_and_reports_sample_size() -> None:
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    # feature present for some indices only; None entries must be dropped.
    feature_values: list[float | None] = [1.0, None, 3.0, 4.0, 5.0, 6.0]
    analysis = analyze_feature("demo", feature_values, closes, horizon=1, num_buckets=2)
    # horizon 1 makes index 5 return None; index 1 feature is None -> both dropped.
    assert analysis.sample_size == 4
    assert analysis.feature == "demo"
    assert analysis.horizon == 1


def test_analyze_feature_no_data_is_empty() -> None:
    closes = [100.0, 101.0, 102.0]
    analysis = analyze_feature("none", [None, None, None], closes, horizon=1)
    assert analysis.sample_size == 0
    assert analysis.correlation is None
    assert analysis.buckets == []
