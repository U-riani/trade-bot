"""V26.1 sample-safe feature-group comparison lab.

Compares candle-only, kline taker, aggregate-trade pressure, live order-book, and
combined available feature groups. This is research tooling only: no model, no
strategy, no trading.

V26.1 fixes an important reporting trap from V26: a combined group must not select
a tiny-sample feature just because another feature in the same group has many
samples. Best features are now chosen only from features that meet
--min-feature-samples, and mixed-sample groups are explicitly flagged.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.backtesting.feature_analysis import FeatureHorizonAnalysis, analyze_feature
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.analyze_market_features import HORIZONS, _build_feature_series
from scripts.backtest_strategy import _load_candles, _resolve_market_data_source

logger = get_logger(__name__)

FEATURE_GROUPS = {
    "candle_only": ["volume_spike_ratio", "body_pct", "upper_wick_pct", "lower_wick_pct"],
    "kline_taker": ["taker_buy_ratio"],
    "agg_trade_pressure": [
        "trade_count_intensity",
        "quote_volume_intensity",
        "taker_buy_trade_ratio",
        "taker_buy_base_ratio_trades",
        "taker_buy_quote_ratio_trades",
        "taker_net_base_volume",
        "taker_net_quote_volume",
        "avg_trade_quote_size",
    ],
    "live_order_book": ["order_book_imbalance", "spread_pct", "imbalance_top_5", "imbalance_top_10", "imbalance_top_20"],
}
FEATURE_GROUPS["combined_available"] = sorted({feature for values in FEATURE_GROUPS.values() for feature in values})
ATOMIC_FEATURE_GROUPS = {key: values for key, values in FEATURE_GROUPS.items() if key != "combined_available"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare predictive value of feature groups (research only).")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--source-timeframe", default="1m")
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--volume-spike-lookback", type=int, default=20)
    parser.add_argument("--num-buckets", type=int, default=5)
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--candidate-min-abs-correlation", type=float, default=0.05)
    parser.add_argument("--candidate-min-quantile-spread", type=float, default=0.02)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    parser.add_argument("--export-candidates-json", type=Path, default=None)
    parser.add_argument("--export-candidates-csv", type=Path, default=None)
    return parser


def _parse_timeframes(value: str) -> list[str]:
    result: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in result:
            result.append(item)
    if not result:
        raise SystemExit("--timeframes must contain at least one timeframe")
    return result


def _quantile_spread(analysis: FeatureHorizonAnalysis) -> float | None:
    if not analysis.buckets:
        return None
    values = [bucket.avg_forward_return_pct for bucket in analysis.buckets]
    return max(values) - min(values)


def _edge_bucket_stats(analysis: FeatureHorizonAnalysis) -> dict[str, float | None]:
    if not analysis.buckets:
        return {
            "lowest_bucket_avg_return_pct": None,
            "highest_bucket_avg_return_pct": None,
            "lowest_bucket_win_rate": None,
            "highest_bucket_win_rate": None,
        }
    lowest = analysis.buckets[0]
    highest = analysis.buckets[-1]
    return {
        "lowest_bucket_avg_return_pct": lowest.avg_forward_return_pct,
        "highest_bucket_avg_return_pct": highest.avg_forward_return_pct,
        "lowest_bucket_win_rate": lowest.win_rate,
        "highest_bucket_win_rate": highest.win_rate,
    }


def _sample_warning(group_analyses: list[FeatureHorizonAnalysis], *, min_feature_samples: int) -> str:
    if not group_analyses:
        return "no_features"
    samples = [analysis.sample_size for analysis in group_analyses]
    if max(samples) < min_feature_samples:
        return "sample_too_small"
    if min(samples) < min_feature_samples:
        return "mixed_sample_sizes"
    return ""


def _group_summary_row(
    *,
    timeframe: str,
    horizon: int,
    group: str,
    group_analyses: list[FeatureHorizonAnalysis],
    min_feature_samples: int,
) -> dict[str, Any]:
    """Return a sample-safe group summary.

    V26 used all group analyses to choose "best" features and only checked the
    group's max sample size. That lets combined_available pick a 46-row
    order-book feature while displaying a 9,000+ candle sample size. V26.1 only
    ranks features that meet min_feature_samples and flags mixed-sample groups.
    """
    usable = [analysis for analysis in group_analyses if analysis.sample_size >= min_feature_samples]
    valid_corr = [analysis for analysis in usable if analysis.correlation is not None]
    best_corr = max(valid_corr, key=lambda analysis: abs(analysis.correlation or 0.0), default=None)

    with_spread = [(analysis, _quantile_spread(analysis)) for analysis in usable]
    with_spread = [(analysis, spread) for analysis, spread in with_spread if spread is not None]
    best_spread = max(with_spread, key=lambda item: item[1], default=None)

    sample_sizes = {analysis.feature: analysis.sample_size for analysis in group_analyses}
    max_sample = max(sample_sizes.values(), default=0)
    min_sample = min(sample_sizes.values(), default=0)
    warning = _sample_warning(group_analyses, min_feature_samples=min_feature_samples)

    return {
        "timeframe": timeframe,
        "horizon": horizon,
        "group": group,
        "feature_count": len(group_analyses),
        "usable_feature_count": len(usable),
        "feature_sample_sizes": sample_sizes,
        "min_feature_sample_size": min_sample,
        "max_sample_size": max_sample,
        "best_abs_correlation_feature": None if best_corr is None else best_corr.feature,
        "best_abs_correlation": None if best_corr is None else best_corr.correlation,
        "best_abs_correlation_sample_size": None if best_corr is None else best_corr.sample_size,
        "best_quantile_spread_feature": None if best_spread is None else best_spread[0].feature,
        "best_quantile_spread": None if best_spread is None else best_spread[1],
        "best_quantile_spread_sample_size": None if best_spread is None else best_spread[0].sample_size,
        "warning": warning,
    }


def _candidate_reason(
    *,
    analysis: FeatureHorizonAnalysis,
    quantile_spread: float | None,
    min_feature_samples: int,
    min_abs_correlation: float,
    min_quantile_spread: float,
) -> str:
    if analysis.sample_size < min_feature_samples:
        return "not_enough_samples"
    abs_corr = abs(analysis.correlation or 0.0)
    spread = abs(quantile_spread or 0.0)
    if abs_corr < min_abs_correlation and spread < min_quantile_spread:
        return "weak_signal"
    return "candidate_research_only"


def _candidate_row(
    *,
    timeframe: str,
    horizon: int,
    source_group: str,
    analysis: FeatureHorizonAnalysis,
    min_feature_samples: int,
    min_abs_correlation: float,
    min_quantile_spread: float,
) -> dict[str, Any]:
    quantile_spread = _quantile_spread(analysis)
    reason = _candidate_reason(
        analysis=analysis,
        quantile_spread=quantile_spread,
        min_feature_samples=min_feature_samples,
        min_abs_correlation=min_abs_correlation,
        min_quantile_spread=min_quantile_spread,
    )
    abs_corr = abs(analysis.correlation or 0.0) if analysis.correlation is not None else None
    edge_stats = _edge_bucket_stats(analysis)
    return {
        "timeframe": timeframe,
        "horizon": horizon,
        "source_group": source_group,
        "feature": analysis.feature,
        "sample_size": analysis.sample_size,
        "correlation": analysis.correlation,
        "abs_correlation": abs_corr,
        "quantile_spread": quantile_spread,
        **edge_stats,
        "usable_for_strategy_research": reason != "not_enough_samples",
        "reason": reason,
    }


def _sort_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[int, int, float, float]:
        usable = 1 if row["usable_for_strategy_research"] else 0
        candidate = 1 if row["reason"] == "candidate_research_only" else 0
        abs_corr = abs(float(row["abs_correlation"] or 0.0))
        spread = abs(float(row["quantile_spread"] or 0.0))
        return (usable, candidate, abs_corr, spread)

    return sorted(rows, key=key, reverse=True)


def _json_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def _export_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready_rows(rows), indent=2), encoding="utf-8")
    logger.info("feature_group_comparison_json_exported", path=str(path), rows=len(rows))


def _export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeframe",
        "horizon",
        "group",
        "feature_count",
        "usable_feature_count",
        "min_feature_sample_size",
        "max_sample_size",
        "best_abs_correlation_feature",
        "best_abs_correlation",
        "best_abs_correlation_sample_size",
        "best_quantile_spread_feature",
        "best_quantile_spread",
        "best_quantile_spread_sample_size",
        "warning",
        "feature_sample_sizes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["feature_sample_sizes"] = json.dumps(csv_row.get("feature_sample_sizes", {}), sort_keys=True)
            writer.writerow(csv_row)
    logger.info("feature_group_comparison_csv_exported", path=str(path), rows=len(rows))


def _export_candidates_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timeframe",
        "horizon",
        "source_group",
        "feature",
        "sample_size",
        "correlation",
        "abs_correlation",
        "quantile_spread",
        "lowest_bucket_avg_return_pct",
        "highest_bucket_avg_return_pct",
        "lowest_bucket_win_rate",
        "highest_bucket_win_rate",
        "usable_for_strategy_research",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("feature_candidates_csv_exported", path=str(path), rows=len(rows))


def _export_candidates_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("feature_candidates_json_exported", path=str(path), rows=len(rows))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    source_candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not source_candles:
        raise SystemExit("No candles available for feature group comparison")

    market_data_source, _use_testnet, exchange_id = _resolve_market_data_source(args.market_data_source)
    logger.info(
        "feature_group_comparison_started",
        market_data_source=market_data_source,
        exchange_id=exchange_id,
        source_candles=len(source_candles),
        timeframes=args.timeframes,
        min_feature_samples=args.min_feature_samples,
        note="V26.1 sample-safe comparison; research only",
    )

    db = Database(settings.database_url)
    await db.connect()
    repository = TradingRepository(db)
    payload: list[dict[str, Any]] = []
    candidate_payload: list[dict[str, Any]] = []
    try:
        for timeframe in _parse_timeframes(args.timeframes):
            candles = resample_candles(source_candles, target_timeframe=timeframe, source_timeframe=args.source_timeframe)
            if not candles:
                continue
            rows = await repository.load_market_features(
                exchange=exchange_id,
                symbol=settings.normalized_symbol,
                timeframe=timeframe,
                limit=len(candles) + 10,
            )
            features_by_close_time = {row.close_time: row for row in rows}
            series = _build_feature_series(
                candles, features_by_close_time, volume_spike_lookback=args.volume_spike_lookback
            )
            closes = [candle.close for candle in candles]

            for horizon in HORIZONS:
                analyses = {
                    feature: analyze_feature(feature, values, closes, horizon, num_buckets=args.num_buckets)
                    for feature, values in series.items()
                }

                for group, feature_names in FEATURE_GROUPS.items():
                    group_analyses = [analyses[name] for name in feature_names if name in analyses]
                    payload.append(
                        _group_summary_row(
                            timeframe=timeframe,
                            horizon=horizon,
                            group=group,
                            group_analyses=group_analyses,
                            min_feature_samples=args.min_feature_samples,
                        )
                    )

                for group, feature_names in ATOMIC_FEATURE_GROUPS.items():
                    for feature_name in feature_names:
                        analysis = analyses.get(feature_name)
                        if analysis is None:
                            continue
                        candidate_payload.append(
                            _candidate_row(
                                timeframe=timeframe,
                                horizon=horizon,
                                source_group=group,
                                analysis=analysis,
                                min_feature_samples=args.min_feature_samples,
                                min_abs_correlation=args.candidate_min_abs_correlation,
                                min_quantile_spread=args.candidate_min_quantile_spread,
                            )
                        )
    finally:
        await db.close()

    candidate_payload = _sort_candidates(candidate_payload)

    if args.export_json:
        _export_json(args.export_json, payload)
    if args.export_csv:
        _export_csv(args.export_csv, payload)
    if args.export_candidates_json:
        _export_candidates_json(args.export_candidates_json, candidate_payload)
    if args.export_candidates_csv:
        _export_candidates_csv(args.export_candidates_csv, candidate_payload)
    logger.info(
        "feature_group_comparison_finished",
        rows=len(payload),
        candidates=len(candidate_payload),
        note="research only; no strategy/trading/profit claim",
    )


if __name__ == "__main__":
    asyncio.run(main())
