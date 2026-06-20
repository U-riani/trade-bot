"""V30 research-only multi-timeframe pullback plus order-book-delta comparison.

Tests a fixed hypothesis:
    15m trend up -> 5m dip inside trend -> 1m order-book improvement.

The baseline uses the same 15m/5m price setup without the order-book condition.
Net return after modeled costs, not raw signal frequency, determines whether the
order-book gate adds value.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from math import ceil
from pathlib import Path
from typing import Any

from app.backtesting.analytics import bars_per_year_for_timeframe, compute_equity_statistics, compute_trade_statistics
from app.backtesting.multitimeframe_pullback_delta_strategy import (
    DeltaBacktestOutcome,
    MultiTimeframeDeltaConfig,
    as_price_setup_config,
    positive_feature_deltas,
    run_multitimeframe_pullback_delta_backtest,
)
from app.backtesting.multitimeframe_pullback_strategy import build_pullback_setup_cache
from app.backtesting.order_book_strategy import rows_with_feature
from app.backtesting.resample import resample_candles
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures
from app.market.models import Candle
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.backtest_strategy import _decimal_to_str, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class CoverageSplit:
    coverage_start: object
    split_time: object
    coverage_end: object
    full_rows: list[MarketFeatures]
    train_rows: list[MarketFeatures]
    validation_rows: list[MarketFeatures]
    train_feature_samples: int
    validation_feature_samples: int


@dataclass(slots=True, frozen=True)
class V30ComparisonRow:
    segment: str
    feature: str
    delta_quantile: Decimal
    delta_threshold: float
    positive_delta_train_samples: int
    horizon_bars: int
    coverage: CoverageSplit
    baseline: DeltaBacktestOutcome
    gated: DeltaBacktestOutcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V30 multi-timeframe pullback/order-book-delta research.")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--order-book-features", default="imbalance_top_20,imbalance_top_5")
    parser.add_argument(
        "--delta-quantiles",
        default="0.5,0.6,0.7",
        help="Train-only quantiles of strictly positive consecutive 1m imbalance deltas.",
    )
    parser.add_argument("--horizons", default="5,10,15", help="1m fixed holding periods in bars.")
    parser.add_argument("--min-current-imbalance", type=float, default=0.0)
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--train-ratio", type=Decimal, default=Decimal("0.7"))
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _parse_csv(value: str) -> list[str]:
    values: list[str] = []
    for raw in value.split(","):
        item = raw.strip()
        if item and item not in values:
            values.append(item)
    if not values:
        raise SystemExit("argument must contain at least one item")
    return values


def _parse_quantiles(value: str) -> list[Decimal]:
    quantiles = [Decimal(item) for item in _parse_csv(value)]
    if any(item <= 0 or item >= 1 for item in quantiles):
        raise SystemExit("delta quantiles must be between 0 and 1")
    return quantiles


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item) for item in _parse_csv(value)]
    if any(item <= 0 for item in horizons):
        raise SystemExit("horizons must be positive")
    return horizons


def _price_feature_row(candle: Candle, observation: MarketFeatures | None = None) -> MarketFeatures:
    """Represent a complete candle while preserving observed order-book fields."""

    if observation is not None:
        return replace(
            observation,
            timeframe=candle.timeframe,
            open_time=candle.open_time,
            close_time=candle.close_time,
            close_price=candle.close,
            volume=candle.volume,
        )
    return MarketFeatures(
        exchange=candle.exchange,
        symbol=candle.symbol,
        timeframe=candle.timeframe,
        open_time=candle.open_time,
        close_time=candle.close_time,
        close_price=candle.close,
        volume=candle.volume,
    )


def _merge_candles_with_observations(candles: list[Candle], observations: list[MarketFeatures]) -> list[MarketFeatures]:
    observed_by_close = {row.close_time: row for row in observations}
    return [_price_feature_row(candle, observed_by_close.get(candle.close_time)) for candle in candles]


async def _load_price_timelines(
    args: argparse.Namespace,
) -> tuple[list[MarketFeatures], list[MarketFeatures], list[MarketFeatures], str]:
    settings = get_settings()
    _source, _use_testnet, exchange = _resolve_market_data_source(args.market_data_source)
    db = Database(settings.database_url)
    await db.connect()
    try:
        repository = TradingRepository(db)
        candles_1m = await repository.load_recent_candles(
            exchange=exchange,
            symbol=settings.normalized_symbol,
            timeframe="1m",
            limit=args.limit,
        )
        observations_1m = await repository.load_market_features(
            exchange=exchange,
            symbol=settings.normalized_symbol,
            timeframe="1m",
            limit=args.limit,
        )
    finally:
        await db.close()

    if not candles_1m:
        raise SystemExit("No 1m candles available for V30")

    entry_rows = _merge_candles_with_observations(candles_1m, observations_1m)
    pullback_rows = [_price_feature_row(candle) for candle in resample_candles(candles_1m, target_timeframe="5m")]
    trend_rows = [_price_feature_row(candle) for candle in resample_candles(candles_1m, target_timeframe="15m")]
    observed_feature_rows = sum(1 for row in entry_rows if row.imbalance_top_20 is not None or row.imbalance_top_5 is not None)

    logger.info(
        "v30_price_timelines_loaded",
        symbol=settings.normalized_symbol,
        candles_1m=len(candles_1m),
        entry_rows=len(entry_rows),
        pullback_rows=len(pullback_rows),
        trend_rows=len(trend_rows),
        observed_order_book_rows=observed_feature_rows,
        note="price timelines come from complete candles; order-book values are joined only at 1m closes",
    )
    return entry_rows, pullback_rows, trend_rows, settings.normalized_symbol


def _coverage_split(rows: list[MarketFeatures], *, feature: str, train_ratio: Decimal) -> CoverageSplit:
    ordered = sorted(rows, key=lambda row: row.close_time)
    observed = rows_with_feature(ordered, feature)
    if len(observed) < 2:
        raise ValueError(f"not enough observed rows for {feature}")
    split_index = int(len(observed) * float(train_ratio))
    if split_index <= 0 or split_index >= len(observed):
        raise ValueError("not enough feature observations to split")

    coverage_start = observed[0].close_time
    coverage_end = observed[-1].close_time
    split_time = observed[split_index - 1].close_time
    full_rows = [row for row in ordered if coverage_start <= row.close_time <= coverage_end]
    train_rows = [row for row in full_rows if row.close_time <= split_time]
    validation_rows = [row for row in full_rows if row.close_time > split_time]
    return CoverageSplit(
        coverage_start=coverage_start,
        split_time=split_time,
        coverage_end=coverage_end,
        full_rows=full_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_feature_samples=len(rows_with_feature(train_rows, feature)),
        validation_feature_samples=len(rows_with_feature(validation_rows, feature)),
    )


def _quantile(values: list[float], quantile: Decimal) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile from no values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(len(ordered) * float(quantile)) - 1))
    return ordered[index]


def _metrics_payload(outcome: DeltaBacktestOutcome) -> dict[str, Any]:
    metrics = outcome.result.metrics
    trades = compute_trade_statistics(outcome.result.trades)
    equity = compute_equity_statistics(outcome.result.equity_curve, bars_per_year=bars_per_year_for_timeframe("1m"))
    diagnostics = outcome.diagnostics
    return {
        "return_pct": _decimal_to_str(metrics.return_pct),
        "final_equity": _decimal_to_str(metrics.final_equity),
        "round_trips": metrics.round_trips,
        "win_rate": metrics.win_rate,
        "profit_factor": _decimal_to_str(trades.profit_factor),
        "expectancy": _decimal_to_str(trades.expectancy),
        "trade_return_sharpe": _decimal_to_str(trades.trade_return_sharpe),
        "max_drawdown_pct": _decimal_to_str(equity.max_drawdown_pct),
        "total_fees": _decimal_to_str(metrics.total_fees),
        "diagnostics": {
            "total_entry_rows": diagnostics.total_entry_rows,
            "pullback_bar_checks": diagnostics.pullback_bar_checks,
            "missing_higher_timeframe_context": diagnostics.missing_higher_timeframe_context,
            "trend_passed": diagnostics.trend_passed,
            "pullback_passed": diagnostics.pullback_passed,
            "price_only_setup_candidates": diagnostics.price_only_setup_candidates,
            "usable_order_book_delta_at_setup": diagnostics.usable_order_book_delta_at_setup,
            "delta_gate_passed": diagnostics.delta_gate_passed,
            "delta_gate_rejected": diagnostics.delta_gate_rejected,
            "delta_gate_missing_feature": diagnostics.delta_gate_missing_feature,
            "skipped_gap_signals": diagnostics.skipped_gap_signals,
            "skipped_end_signals": diagnostics.skipped_end_signals,
            "entry_gap_count": diagnostics.entry_gap_count,
            "max_entry_gap_seconds": diagnostics.max_entry_gap_seconds,
        },
    }


def _verdict(row: V30ComparisonRow, *, min_trades: int) -> str:
    if row.segment == "full":
        return "descriptive_full_sample_only"
    baseline = row.baseline.result.metrics
    gated = row.gated.result.metrics
    gated_trade_stats = compute_trade_statistics(row.gated.result.trades)
    if gated.round_trips < min_trades:
        return "not_enough_gated_trades"
    if gated.return_pct <= 0:
        return "rejected_gated_negative_after_costs"
    if gated_trade_stats.profit_factor is None or gated_trade_stats.profit_factor <= Decimal("1"):
        return "rejected_gated_profit_factor_not_above_1"
    if gated.return_pct <= baseline.return_pct:
        return "rejected_not_better_than_price_only_baseline"
    return "promising_research_only"


def _payload(row: V30ComparisonRow, rank: int, *, min_trades: int, min_current_imbalance: float) -> dict[str, Any]:
    baseline = _metrics_payload(row.baseline)
    gated = _metrics_payload(row.gated)
    baseline_return = row.baseline.result.metrics.return_pct
    gated_return = row.gated.result.metrics.return_pct
    return {
        "rank": rank,
        "segment": row.segment,
        "strategy_name": "v30_15m_trend_5m_pullback_1m_order_book_delta",
        "entry_timeframe": "1m",
        "pullback_timeframe": "5m",
        "trend_timeframe": "15m",
        "order_book_feature": row.feature,
        "delta_quantile": str(row.delta_quantile),
        "delta_threshold": row.delta_threshold,
        "positive_delta_train_samples": row.positive_delta_train_samples,
        "min_current_imbalance": min_current_imbalance,
        "horizon_bars": row.horizon_bars,
        "coverage": {
            "start": row.coverage.coverage_start.isoformat(),
            "split_time": row.coverage.split_time.isoformat(),
            "end": row.coverage.coverage_end.isoformat(),
            "train_feature_samples": row.coverage.train_feature_samples,
            "validation_feature_samples": row.coverage.validation_feature_samples,
        },
        "verdict": _verdict(row, min_trades=min_trades),
        "notes": (
            "Research-only. 15m close above rising EMA50; 5m close below EMA9 but above EMA50 with RSI 30-50; "
            "gated variant additionally needs a consecutive observed 1m order-book improvement above a train-learned "
            "positive delta percentile, while current imbalance is not negative. Signals enter on the next contiguous 1m "
            "candle and use a fixed horizon."
        ),
        "price_only_baseline": baseline,
        "order_book_delta_gated": gated,
        "improvement": {
            "return_pct_delta": _decimal_to_str(gated_return - baseline_return),
            "gated_beats_baseline": gated_return > baseline_return,
        },
    }


def _sort_key(row: V30ComparisonRow) -> tuple[int, Decimal, Decimal, int]:
    gated = row.gated.result.metrics
    baseline = row.baseline.result.metrics
    priority = {"validation": 2, "full": 1, "train": 0}.get(row.segment, 0)
    profit_factor = compute_trade_statistics(row.gated.result.trades).profit_factor or Decimal("0")
    return (priority, gated.return_pct - baseline.return_pct, profit_factor, gated.round_trips)


def _export_json(path: Path, rows: list[V30ComparisonRow], *, min_trades: int, min_current_imbalance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        _payload(row, rank, min_trades=min_trades, min_current_imbalance=min_current_imbalance)
        for rank, row in enumerate(rows, start=1)
    ]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("v30_json_exported", path=str(path), rows=len(rows))


def _export_csv(path: Path, rows: list[V30ComparisonRow], *, min_trades: int, min_current_imbalance: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "segment", "order_book_feature", "delta_quantile", "delta_threshold", "positive_delta_train_samples",
        "horizon_bars", "verdict", "baseline_return_pct", "baseline_round_trips", "gated_return_pct",
        "gated_round_trips", "gated_profit_factor", "return_pct_delta", "gated_beats_baseline",
        "price_setup_candidates", "usable_delta_at_setup", "delta_gate_passed", "delta_gate_rejected",
        "delta_gate_missing_feature", "gated_skipped_gap_signals",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = _payload(row, rank, min_trades=min_trades, min_current_imbalance=min_current_imbalance)
            gated_diagnostics = payload["order_book_delta_gated"]["diagnostics"]
            writer.writerow({
                "rank": rank,
                "segment": row.segment,
                "order_book_feature": row.feature,
                "delta_quantile": str(row.delta_quantile),
                "delta_threshold": row.delta_threshold,
                "positive_delta_train_samples": row.positive_delta_train_samples,
                "horizon_bars": row.horizon_bars,
                "verdict": payload["verdict"],
                "baseline_return_pct": payload["price_only_baseline"]["return_pct"],
                "baseline_round_trips": payload["price_only_baseline"]["round_trips"],
                "gated_return_pct": payload["order_book_delta_gated"]["return_pct"],
                "gated_round_trips": payload["order_book_delta_gated"]["round_trips"],
                "gated_profit_factor": payload["order_book_delta_gated"]["profit_factor"],
                "return_pct_delta": payload["improvement"]["return_pct_delta"],
                "gated_beats_baseline": payload["improvement"]["gated_beats_baseline"],
                "price_setup_candidates": gated_diagnostics["price_only_setup_candidates"],
                "usable_delta_at_setup": gated_diagnostics["usable_order_book_delta_at_setup"],
                "delta_gate_passed": gated_diagnostics["delta_gate_passed"],
                "delta_gate_rejected": gated_diagnostics["delta_gate_rejected"],
                "delta_gate_missing_feature": gated_diagnostics["delta_gate_missing_feature"],
                "gated_skipped_gap_signals": gated_diagnostics["skipped_gap_signals"],
            })
    logger.info("v30_csv_exported", path=str(path), rows=len(rows))


def _build_rows(
    *,
    base_rows: list[MarketFeatures],
    pullback_rows: list[MarketFeatures],
    trend_rows: list[MarketFeatures],
    feature: str,
    delta_quantiles: list[Decimal],
    horizons: list[int],
    args: argparse.Namespace,
    symbol: str,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> list[V30ComparisonRow]:
    observed = rows_with_feature(base_rows, feature)
    if len(observed) < args.min_feature_samples:
        logger.info("v30_feature_skipped", feature=feature, reason="not_enough_feature_samples", samples=len(observed))
        return []

    coverage = _coverage_split(base_rows, feature=feature, train_ratio=args.train_ratio)
    if coverage.train_feature_samples < max(1, args.min_feature_samples // 2):
        logger.info("v30_feature_skipped", feature=feature, reason="not_enough_train_feature_samples", samples=coverage.train_feature_samples)
        return []

    train_positive_deltas = positive_feature_deltas(
        coverage.train_rows,
        feature_name=feature,
        timeframe="1m",
    )
    min_delta_samples = max(10, args.min_feature_samples // 4)
    if len(train_positive_deltas) < min_delta_samples:
        logger.info(
            "v30_feature_skipped",
            feature=feature,
            reason="not_enough_positive_train_deltas",
            positive_delta_samples=len(train_positive_deltas),
            required=min_delta_samples,
        )
        return []

    logger.info(
        "v30_coverage_split",
        feature=feature,
        coverage_start=coverage.coverage_start.isoformat(),
        split_time=coverage.split_time.isoformat(),
        coverage_end=coverage.coverage_end.isoformat(),
        train_feature_samples=coverage.train_feature_samples,
        validation_feature_samples=coverage.validation_feature_samples,
        positive_train_delta_samples=len(train_positive_deltas),
        full_entry_rows=len(coverage.full_rows),
        train_entry_rows=len(coverage.train_rows),
        validation_entry_rows=len(coverage.validation_rows),
    )

    segments = {
        "full": coverage.full_rows,
        "train": coverage.train_rows,
        "validation": coverage.validation_rows,
    }
    settings = get_settings()
    output: list[V30ComparisonRow] = []

    template = MultiTimeframeDeltaConfig(
        feature_name=feature,
        delta_threshold=0.0,
        horizon_bars=horizons[0],
        strategy_name="v30_setup_cache",
        require_order_book_delta=False,
        min_current_imbalance=args.min_current_imbalance,
    )
    price_setup_template = as_price_setup_config(template)
    setup_by_segment = {
        segment: build_pullback_setup_cache(
            entry_rows=entry_segment,
            pullback_rows=pullback_rows,
            trend_rows=trend_rows,
            config=price_setup_template,
        )
        for segment, entry_segment in segments.items()
    }

    thresholds = [(quantile, _quantile(train_positive_deltas, quantile)) for quantile in delta_quantiles]
    for horizon in horizons:
        baseline_by_segment: dict[str, DeltaBacktestOutcome] = {}
        for segment, entry_segment in segments.items():
            baseline_config = MultiTimeframeDeltaConfig(
                feature_name=feature,
                delta_threshold=0.0,
                horizon_bars=horizon,
                strategy_name=f"v30_{feature}_baseline_h{horizon}",
                require_order_book_delta=False,
                min_current_imbalance=args.min_current_imbalance,
            )
            baseline_by_segment[segment] = run_multitimeframe_pullback_delta_backtest(
                entry_rows=entry_segment,
                pullback_rows=pullback_rows,
                trend_rows=trend_rows,
                config=baseline_config,
                symbol=symbol,
                initial_quote_balance=settings.initial_quote_balance,
                quote_amount=settings.max_order_usdt,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
                setup_cache=setup_by_segment[segment],
            )

        for quantile, threshold in thresholds:
            for segment, entry_segment in segments.items():
                gated_config = MultiTimeframeDeltaConfig(
                    feature_name=feature,
                    delta_threshold=threshold,
                    horizon_bars=horizon,
                    strategy_name=f"v30_{feature}_delta_q{str(quantile).replace('.', 'p')}_h{horizon}",
                    require_order_book_delta=True,
                    min_current_imbalance=args.min_current_imbalance,
                )
                gated = run_multitimeframe_pullback_delta_backtest(
                    entry_rows=entry_segment,
                    pullback_rows=pullback_rows,
                    trend_rows=trend_rows,
                    config=gated_config,
                    symbol=symbol,
                    initial_quote_balance=settings.initial_quote_balance,
                    quote_amount=settings.max_order_usdt,
                    fee_rate_pct=fee_rate_pct,
                    slippage_pct=slippage_pct,
                    setup_cache=setup_by_segment[segment],
                )
                output.append(
                    V30ComparisonRow(
                        segment=segment,
                        feature=feature,
                        delta_quantile=quantile,
                        delta_threshold=threshold,
                        positive_delta_train_samples=len(train_positive_deltas),
                        horizon_bars=horizon,
                        coverage=coverage,
                        baseline=baseline_by_segment[segment],
                        gated=gated,
                    )
                )
    return output


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    if args.limit <= 0 or args.min_feature_samples <= 0 or args.min_trades <= 0 or args.top <= 0:
        raise SystemExit("limit, min-feature-samples, min-trades, and top must be positive")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise SystemExit("train-ratio must be between 0 and 1")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("slippage-pct cannot be negative")

    features = _parse_csv(args.order_book_features)
    delta_quantiles = _parse_quantiles(args.delta_quantiles)
    horizons = _parse_horizons(args.horizons)
    settings = get_settings()
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "v30_backtest_started",
        entry_timeframe="1m",
        pullback_timeframe="5m",
        trend_timeframe="15m",
        features=",".join(features),
        delta_quantiles=",".join(str(item) for item in delta_quantiles),
        horizons=",".join(str(item) for item in horizons),
        min_current_imbalance=args.min_current_imbalance,
        min_trades=args.min_trades,
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research-only; net profitability after modeled costs is the criterion",
    )

    base_rows, pullback_rows, trend_rows, symbol = await _load_price_timelines(args)
    comparison_rows: list[V30ComparisonRow] = []
    for feature in features:
        comparison_rows.extend(
            _build_rows(
                base_rows=base_rows,
                pullback_rows=pullback_rows,
                trend_rows=trend_rows,
                feature=feature,
                delta_quantiles=delta_quantiles,
                horizons=horizons,
                args=args,
                symbol=symbol,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )

    if not comparison_rows:
        raise SystemExit("No V30 rows produced. Check order-book feature coverage.")

    ranked = sorted(comparison_rows, key=_sort_key, reverse=True)
    logger.info("v30_backtest_finished", rows=len(ranked), note="research only")
    for rank, row in enumerate(ranked[: args.top], start=1):
        payload = _payload(
            row,
            rank,
            min_trades=args.min_trades,
            min_current_imbalance=args.min_current_imbalance,
        )
        diagnostics = payload["order_book_delta_gated"]["diagnostics"]
        logger.info(
            "v30_result",
            rank=rank,
            segment=row.segment,
            feature=row.feature,
            delta_quantile=str(row.delta_quantile),
            delta_threshold=round(row.delta_threshold, 6),
            horizon_bars=row.horizon_bars,
            verdict=payload["verdict"],
            baseline_return_pct=payload["price_only_baseline"]["return_pct"],
            gated_return_pct=payload["order_book_delta_gated"]["return_pct"],
            return_delta_pct=payload["improvement"]["return_pct_delta"],
            baseline_trades=payload["price_only_baseline"]["round_trips"],
            gated_trades=payload["order_book_delta_gated"]["round_trips"],
            gated_profit_factor=payload["order_book_delta_gated"]["profit_factor"],
            price_setup_candidates=diagnostics["price_only_setup_candidates"],
            usable_delta_at_setup=diagnostics["usable_order_book_delta_at_setup"],
            delta_gate_passed=diagnostics["delta_gate_passed"],
            delta_gate_rejected=diagnostics["delta_gate_rejected"],
            delta_gate_missing_feature=diagnostics["delta_gate_missing_feature"],
            skipped_gap_signals=diagnostics["skipped_gap_signals"],
        )

    if args.export_json:
        _export_json(
            args.export_json,
            ranked,
            min_trades=args.min_trades,
            min_current_imbalance=args.min_current_imbalance,
        )
    if args.export_csv:
        _export_csv(
            args.export_csv,
            ranked,
            min_trades=args.min_trades,
            min_current_imbalance=args.min_current_imbalance,
        )


if __name__ == "__main__":
    asyncio.run(main())
