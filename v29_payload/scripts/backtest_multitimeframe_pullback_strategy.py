"""V29 research-only multi-timeframe pullback plus order-book-reversal comparison.

Tests a fixed hypothesis:
    15m trend up -> 5m dip inside trend -> 1m order-book reversal.

For each configuration, compare the exact price-only setup against the same setup
with 1m order-book reversal confirmation.  Net returns after modeled fees and
slippage, not raw signal frequency, determine whether the gate adds value.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.analytics import bars_per_year_for_timeframe, compute_equity_statistics, compute_trade_statistics
from app.backtesting.multitimeframe_pullback_strategy import (
    MultiTimeframeBacktestOutcome,
    MultiTimeframePullbackConfig,
    build_pullback_setup_cache,
    run_multitimeframe_pullback_backtest,
)
from app.backtesting.order_book_strategy import quantile_threshold, rows_with_feature
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures
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
class V29ComparisonRow:
    segment: str
    feature: str
    entry_quantile: Decimal
    threshold: float
    horizon_bars: int
    coverage: CoverageSplit
    baseline: MultiTimeframeBacktestOutcome
    gated: MultiTimeframeBacktestOutcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V29 multi-timeframe pullback/order-book-reversal research.")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--order-book-features", default="imbalance_top_20,imbalance_top_5")
    parser.add_argument("--entry-quantiles", default="0.6,0.7,0.8")
    parser.add_argument("--horizons", default="5,10,15", help="1m fixed holding periods in bars.")
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
        raise SystemExit("entry quantiles must be between 0 and 1")
    return quantiles


def _parse_horizons(value: str) -> list[int]:
    horizons = [int(item) for item in _parse_csv(value)]
    if any(item <= 0 for item in horizons):
        raise SystemExit("horizons must be positive")
    return horizons


async def _load_rows(timeframe: str, args: argparse.Namespace) -> tuple[list[MarketFeatures], str]:
    settings = get_settings()
    _source, _use_testnet, exchange = _resolve_market_data_source(args.market_data_source)
    db = Database(settings.database_url)
    await db.connect()
    try:
        rows = await TradingRepository(db).load_market_features(
            exchange=exchange,
            symbol=settings.normalized_symbol,
            timeframe=timeframe,
            limit=args.limit,
        )
    finally:
        await db.close()
    logger.info("v29_feature_rows_loaded", timeframe=timeframe, rows=len(rows), symbol=settings.normalized_symbol)
    return rows, settings.normalized_symbol


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


def _metrics_payload(outcome: MultiTimeframeBacktestOutcome) -> dict[str, Any]:
    metrics = outcome.result.metrics
    trades = compute_trade_statistics(outcome.result.trades)
    equity = compute_equity_statistics(outcome.result.equity_curve, bars_per_year=bars_per_year_for_timeframe("1m"))
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
            "total_entry_rows": outcome.diagnostics.total_entry_rows,
            "pullback_bar_checks": outcome.diagnostics.pullback_bar_checks,
            "missing_higher_timeframe_context": outcome.diagnostics.missing_higher_timeframe_context,
            "trend_passed": outcome.diagnostics.trend_passed,
            "pullback_passed": outcome.diagnostics.pullback_passed,
            "baseline_signal_candidates": outcome.diagnostics.baseline_signal_candidates,
            "reversal_gate_passed": outcome.diagnostics.reversal_gate_passed,
            "reversal_gate_rejected": outcome.diagnostics.reversal_gate_rejected,
            "reversal_gate_missing_feature": outcome.diagnostics.reversal_gate_missing_feature,
            "skipped_gap_signals": outcome.diagnostics.skipped_gap_signals,
            "skipped_end_signals": outcome.diagnostics.skipped_end_signals,
            "entry_gap_count": outcome.diagnostics.entry_gap_count,
            "max_entry_gap_seconds": outcome.diagnostics.max_entry_gap_seconds,
        },
    }


def _verdict(row: V29ComparisonRow, *, min_trades: int) -> str:
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


def _payload(row: V29ComparisonRow, rank: int, *, min_trades: int) -> dict[str, Any]:
    baseline = _metrics_payload(row.baseline)
    gated = _metrics_payload(row.gated)
    baseline_return = row.baseline.result.metrics.return_pct
    gated_return = row.gated.result.metrics.return_pct
    return {
        "rank": rank,
        "segment": row.segment,
        "strategy_name": "v29_15m_trend_5m_pullback_1m_order_book_reversal",
        "entry_timeframe": "1m",
        "pullback_timeframe": "5m",
        "trend_timeframe": "15m",
        "order_book_feature": row.feature,
        "entry_quantile": str(row.entry_quantile),
        "reversal_threshold": row.threshold,
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
            "gated variant additionally needs 1m observed order-book reversal from <=0 to train-learned positive threshold. "
            "Signals enter on the next contiguous 1m candle and use a fixed horizon."
        ),
        "price_only_baseline": baseline,
        "order_book_reversal_gated": gated,
        "improvement": {
            "return_pct_delta": _decimal_to_str(gated_return - baseline_return),
            "gated_beats_baseline": gated_return > baseline_return,
        },
    }


def _sort_key(row: V29ComparisonRow) -> tuple[int, Decimal, Decimal, int]:
    gated = row.gated.result.metrics
    baseline = row.baseline.result.metrics
    priority = {"validation": 2, "full": 1, "train": 0}.get(row.segment, 0)
    profit_factor = compute_trade_statistics(row.gated.result.trades).profit_factor or Decimal("0")
    return (priority, gated.return_pct - baseline.return_pct, profit_factor, gated.round_trips)


def _export_json(path: Path, rows: list[V29ComparisonRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_payload(row, rank, min_trades=min_trades) for rank, row in enumerate(rows, start=1)], indent=2), encoding="utf-8")
    logger.info("v29_json_exported", path=str(path), rows=len(rows))


def _export_csv(path: Path, rows: list[V29ComparisonRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "segment", "order_book_feature", "entry_quantile", "reversal_threshold", "horizon_bars", "verdict",
        "baseline_return_pct", "baseline_round_trips", "gated_return_pct", "gated_round_trips", "gated_profit_factor",
        "return_pct_delta", "gated_beats_baseline", "gated_skipped_gap_signals",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = _payload(row, rank, min_trades=min_trades)
            writer.writerow({
                "rank": rank,
                "segment": row.segment,
                "order_book_feature": row.feature,
                "entry_quantile": str(row.entry_quantile),
                "reversal_threshold": row.threshold,
                "horizon_bars": row.horizon_bars,
                "verdict": payload["verdict"],
                "baseline_return_pct": payload["price_only_baseline"]["return_pct"],
                "baseline_round_trips": payload["price_only_baseline"]["round_trips"],
                "gated_return_pct": payload["order_book_reversal_gated"]["return_pct"],
                "gated_round_trips": payload["order_book_reversal_gated"]["round_trips"],
                "gated_profit_factor": payload["order_book_reversal_gated"]["profit_factor"],
                "return_pct_delta": payload["improvement"]["return_pct_delta"],
                "gated_beats_baseline": payload["improvement"]["gated_beats_baseline"],
                "gated_skipped_gap_signals": payload["order_book_reversal_gated"]["diagnostics"]["skipped_gap_signals"],
            })
    logger.info("v29_csv_exported", path=str(path), rows=len(rows))


def _build_rows(
    *,
    base_rows: list[MarketFeatures],
    pullback_rows: list[MarketFeatures],
    trend_rows: list[MarketFeatures],
    feature: str,
    quantiles: list[Decimal],
    horizons: list[int],
    args: argparse.Namespace,
    symbol: str,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> list[V29ComparisonRow]:
    observed = rows_with_feature(base_rows, feature)
    if len(observed) < args.min_feature_samples:
        logger.info("v29_feature_skipped", feature=feature, reason="not_enough_feature_samples", samples=len(observed))
        return []

    coverage = _coverage_split(base_rows, feature=feature, train_ratio=args.train_ratio)
    if coverage.train_feature_samples < max(1, args.min_feature_samples // 2):
        logger.info("v29_feature_skipped", feature=feature, reason="not_enough_train_feature_samples", samples=coverage.train_feature_samples)
        return []

    logger.info(
        "v29_coverage_split",
        feature=feature,
        coverage_start=coverage.coverage_start.isoformat(),
        split_time=coverage.split_time.isoformat(),
        coverage_end=coverage.coverage_end.isoformat(),
        train_feature_samples=coverage.train_feature_samples,
        validation_feature_samples=coverage.validation_feature_samples,
        full_entry_rows=len(coverage.full_rows),
        train_entry_rows=len(coverage.train_rows),
        validation_entry_rows=len(coverage.validation_rows),
    )

    segments = {
        "full": coverage.full_rows,
        "train": coverage.train_rows,
        "validation": coverage.validation_rows,
    }
    output: list[V29ComparisonRow] = []
    settings = get_settings()

    # The 15m/5m price setup does not depend on feature quantile or holding period.
    # Build it once per segment, then replay baseline/gated variants against the
    # same cached chronology. This is both faster and less error-prone than
    # recalculating rolling indicators for every threshold combination.
    setup_template = MultiTimeframePullbackConfig(
        feature_name=feature,
        reversal_threshold=0.0,
        horizon_bars=horizons[0],
        strategy_name="v29_setup_cache",
        require_order_book_reversal=False,
    )
    setup_by_segment = {
        segment: build_pullback_setup_cache(
            entry_rows=entry_segment,
            pullback_rows=pullback_rows,
            trend_rows=trend_rows,
            config=setup_template,
        )
        for segment, entry_segment in segments.items()
    }
    thresholds = [
        (quantile, quantile_threshold(coverage.train_rows, feature, float(quantile)))
        for quantile in quantiles
    ]

    for horizon in horizons:
        baseline_by_segment: dict[str, MultiTimeframeBacktestOutcome] = {}
        for segment, entry_segment in segments.items():
            baseline_config = MultiTimeframePullbackConfig(
                feature_name=feature,
                reversal_threshold=0.0,
                horizon_bars=horizon,
                strategy_name=f"v29_{feature}_baseline_h{horizon}",
                require_order_book_reversal=False,
            )
            baseline_by_segment[segment] = run_multitimeframe_pullback_backtest(
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
                gated_config = MultiTimeframePullbackConfig(
                    feature_name=feature,
                    reversal_threshold=threshold,
                    horizon_bars=horizon,
                    strategy_name=f"v29_{feature}_reversal_q{str(quantile).replace('.', 'p')}_h{horizon}",
                    require_order_book_reversal=True,
                )
                gated = run_multitimeframe_pullback_backtest(
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
                    V29ComparisonRow(
                        segment=segment,
                        feature=feature,
                        entry_quantile=quantile,
                        threshold=threshold,
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
    quantiles = _parse_quantiles(args.entry_quantiles)
    horizons = _parse_horizons(args.horizons)
    settings = get_settings()
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "v29_backtest_started",
        entry_timeframe="1m",
        pullback_timeframe="5m",
        trend_timeframe="15m",
        features=",".join(features),
        entry_quantiles=",".join(str(item) for item in quantiles),
        horizons=",".join(str(item) for item in horizons),
        min_trades=args.min_trades,
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research-only; net profitability after modeled costs is the criterion",
    )

    base_rows, symbol = await _load_rows("1m", args)
    pullback_rows, _ = await _load_rows("5m", args)
    trend_rows, _ = await _load_rows("15m", args)

    comparison_rows: list[V29ComparisonRow] = []
    for feature in features:
        comparison_rows.extend(
            _build_rows(
                base_rows=base_rows,
                pullback_rows=pullback_rows,
                trend_rows=trend_rows,
                feature=feature,
                quantiles=quantiles,
                horizons=horizons,
                args=args,
                symbol=symbol,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
        )

    if not comparison_rows:
        raise SystemExit("No V29 rows produced. Check order-book feature coverage.")

    ranked = sorted(comparison_rows, key=_sort_key, reverse=True)
    logger.info("v29_backtest_finished", rows=len(ranked), note="research only")
    for rank, row in enumerate(ranked[: args.top], start=1):
        payload = _payload(row, rank, min_trades=args.min_trades)
        logger.info(
            "v29_result",
            rank=rank,
            segment=row.segment,
            feature=row.feature,
            entry_quantile=str(row.entry_quantile),
            threshold=round(row.threshold, 6),
            horizon_bars=row.horizon_bars,
            verdict=payload["verdict"],
            baseline_return_pct=payload["price_only_baseline"]["return_pct"],
            gated_return_pct=payload["order_book_reversal_gated"]["return_pct"],
            return_delta_pct=payload["improvement"]["return_pct_delta"],
            baseline_trades=payload["price_only_baseline"]["round_trips"],
            gated_trades=payload["order_book_reversal_gated"]["round_trips"],
            gated_profit_factor=payload["order_book_reversal_gated"]["profit_factor"],
            skipped_gap_signals=payload["order_book_reversal_gated"]["diagnostics"]["skipped_gap_signals"],
        )

    if args.export_json:
        _export_json(args.export_json, ranked, min_trades=args.min_trades)
    if args.export_csv:
        _export_csv(args.export_csv, ranked, min_trades=args.min_trades)


if __name__ == "__main__":
    asyncio.run(main())
