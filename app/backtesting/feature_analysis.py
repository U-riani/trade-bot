"""Predictive-value analysis for market features.

Pure, testable statistics for answering one question: does a feature carry any
information about future price? Two complementary lenses:

1. Pearson correlation between the feature and forward return. A single number,
   easy to over-trust. A correlation near zero means no *linear* signal.
2. Quantile-bucket analysis: sort observations by feature value, split into N
   equal groups, and report average forward return + win rate per group. This
   catches monotonic and non-linear effects a single correlation hides, and the
   per-bucket sample size makes "this is just three lucky trades" obvious.

Honesty guard: every function reports the sample size it actually used after
dropping missing (None) observations. A feature with no historical data (e.g.
order-book imbalance) yields sample_size 0 and no buckets, instead of a
confident-looking number computed from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


def forward_returns(closes: list[float], horizon: int) -> list[float | None]:
    """Percent return from candle i to candle i+horizon; None when out of range."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    result: list[float | None] = []
    for index in range(len(closes)):
        future = index + horizon
        if future >= len(closes) or closes[index] <= 0:
            result.append(None)
        else:
            result.append((closes[future] - closes[index]) / closes[index] * 100.0)
    return result


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation of two equal-length series; None if undefined."""
    if len(xs) != len(ys):
        raise ValueError("series must be the same length")
    n = len(xs)
    if n < 2:
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / sqrt(var_x * var_y)


@dataclass(slots=True, frozen=True)
class BucketStat:
    bucket: int  # 0-based, lowest feature values first
    lower: float
    upper: float
    count: int
    avg_forward_return_pct: float
    win_rate: float  # fraction of forward returns strictly > 0


@dataclass(slots=True, frozen=True)
class FeatureHorizonAnalysis:
    feature: str
    horizon: int
    sample_size: int
    correlation: float | None
    buckets: list[BucketStat]


def _aligned_pairs(
    feature_values: list[float | None],
    returns: list[float | None],
) -> list[tuple[float, float]]:
    """Keep only (feature, return) pairs where both are present."""
    pairs: list[tuple[float, float]] = []
    for value, ret in zip(feature_values, returns, strict=True):
        if value is None or ret is None:
            continue
        pairs.append((float(value), float(ret)))
    return pairs


def quantile_buckets(pairs: list[tuple[float, float]], num_buckets: int) -> list[BucketStat]:
    """Split pairs into `num_buckets` equal-size groups sorted by feature value.

    Buckets are by rank (equal counts), not equal value-width, so a skewed
    feature still produces balanced sample sizes. Returns fewer buckets than
    requested when there are not enough observations to fill them.
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    if len(pairs) < num_buckets:
        return []

    ordered = sorted(pairs, key=lambda item: item[0])
    n = len(ordered)
    buckets: list[BucketStat] = []
    for bucket_index in range(num_buckets):
        start = (bucket_index * n) // num_buckets
        end = ((bucket_index + 1) * n) // num_buckets
        group = ordered[start:end]
        if not group:
            continue
        rets = [ret for _value, ret in group]
        wins = sum(1 for ret in rets if ret > 0)
        buckets.append(
            BucketStat(
                bucket=bucket_index,
                lower=group[0][0],
                upper=group[-1][0],
                count=len(group),
                avg_forward_return_pct=sum(rets) / len(rets),
                win_rate=wins / len(rets),
            )
        )
    return buckets


def analyze_feature(
    feature_name: str,
    feature_values: list[float | None],
    closes: list[float],
    horizon: int,
    *,
    num_buckets: int = 5,
) -> FeatureHorizonAnalysis:
    """Full analysis of one feature against one forward-return horizon."""
    returns = forward_returns(closes, horizon)
    pairs = _aligned_pairs(feature_values, returns)

    correlation = None
    if pairs:
        correlation = pearson_correlation([p[0] for p in pairs], [p[1] for p in pairs])

    return FeatureHorizonAnalysis(
        feature=feature_name,
        horizon=horizon,
        sample_size=len(pairs),
        correlation=correlation,
        buckets=quantile_buckets(pairs, num_buckets),
    )
