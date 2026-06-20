"""V28 research-only test: does order-book gating improve price strategies?

The script compares a deterministic price-entry baseline with the exact same
entry rule filtered by a train-learned order-book condition. It is gap-safe,
coverage-aware, and uses only data known at each signal close.
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
from app.backtesting.order_book_gated_strategy import (
    PRICE_STRATEGIES,
    GatedBacktestOutcome,
    GatedStrategyConfig,
    OrderBookGateConfig,
    PriceRuleConfig,
    run_order_book_gated_backtest,
)
from app.backtesting.order_book_strategy import quantile_threshold, rows_with_feature, split_rows_by_feature_coverage
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.backtest_strategy import _decimal_to_str, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class V28ComparisonRow:
    timeframe: str
    segment: str
    price_strategy: str
    order_book_feature: str
    gate_tail: str
    configured_quantile: Decimal
    threshold_quantile: Decimal
    threshold: float
    horizon_bars: int
    sample_size: int
    coverage_start: object
    split_time: object
    coverage_end: object
    baseline: GatedBacktestOutcome
    gated: GatedBacktestOutcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "V28 research-only comparison of price-strategy baselines against "
            "train-learned order-book-gated versions. No live trading."
        )
    )
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--timeframes", default="1m,5m,15m")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--strategies", default=",".join(PRICE_STRATEGIES))
    parser.add_argument("--order-book-features", default="imbalance_top_20,imbalance_top_5")
    parser.add_argument("--entry-quantiles", default="0.6,0.7,0.8")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--train-ratio", type=Decimal, default=Decimal("0.7"))
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--ema-fast-period", type=int, default=12)
    parser.add_argument("--ema-slow-period", type=int, default=34)
    parser.add_argument("--trend-ema-period", type=int, default=50)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--rsi-buy-min", type=float, default=45.0)
    parser.add_argument("--rsi-buy-max", type=float, default=60.0)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--mean-reversion-lookback", type=int, default=20)
    parser.add_argument("--mean-reversion-entry-z", type=float, default=2.0)
    parser.add_argument("--mean-reversion-rsi-max", type=float, default=35.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _parse_csv(value: str) -> list[str]:
    items: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if item and item not in items:
            items.append(item)
    if not items:
        raise SystemExit("argument must contain at least one item")
    return items


def _parse_quantiles(value: str) -> list[Decimal]:
    values = [Decimal(item) for item in _parse_csv(value)]
    if any(value <= 0 or value >= 1 for value in values):
        raise SystemExit("entry quantiles must be strictly between 0 and 1")
    return values


def _parse_horizons(value: str) -> list[int]:
    values = [int(item) for item in _parse_csv(value)]
    if any(value <= 0 for value in values):
        raise SystemExit("horizons must be positive")
    return values


def _gate_tail_for(price_strategy: str) -> str:
    # Trend-following setups need supportive bid-side pressure; mean reversion
    # looks for a washout and therefore tests the lower order-book tail.
    return "low" if price_strategy == "mean_reversion" else "high"


def _threshold_quantile(configured: Decimal, *, tail: str) -> Decimal:
    return configured if tail == "high" else Decimal("1") - configured


def _build_price_rule(args: argparse.Namespace, strategy: str) -> PriceRuleConfig:
    return PriceRuleConfig(
        strategy_kind=strategy,
        fast_ema_period=args.ema_fast_period,
        slow_ema_period=args.ema_slow_period,
        trend_ema_period=args.trend_ema_period,
        rsi_period=args.rsi_period,
        rsi_buy_min=args.rsi_buy_min,
        rsi_buy_max=args.rsi_buy_max,
        breakout_lookback=args.breakout_lookback,
        mean_reversion_lookback=args.mean_reversion_lookback,
        mean_reversion_entry_z=args.mean_reversion_entry_z,
        mean_reversion_rsi_max=args.mean_reversion_rsi_max,
    )


def _config_name(
    *,
    strategy: str,
    feature: str,
    tail: str,
    configured_quantile: Decimal,
    horizon: int,
    gated: bool,
) -> str:
    q_label = str(configured_quantile).replace(".", "p")
    gate_label = "gated" if gated else "baseline"
    return f"v28_{strategy}_{gate_label}_{feature}_{tail}_q{q_label}_h{horizon}"


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
    logger.info("v28_feature_rows_loaded", timeframe=timeframe, rows=len(rows), symbol=settings.normalized_symbol)
    return rows, settings.normalized_symbol


def _metric_payload(outcome: GatedBacktestOutcome, *, timeframe: str) -> dict[str, Any]:
    metrics = outcome.result.metrics
    trade_stats = compute_trade_statistics(outcome.result.trades)
    equity_stats = compute_equity_statistics(
        outcome.result.equity_curve,
        bars_per_year=bars_per_year_for_timeframe(timeframe),
    )
    return {
        "return_pct": _decimal_to_str(metrics.return_pct),
        "final_equity": _decimal_to_str(metrics.final_equity),
        "round_trips": metrics.round_trips,
        "win_rate": metrics.win_rate,
        "profit_factor": _decimal_to_str(trade_stats.profit_factor),
        "expectancy": _decimal_to_str(trade_stats.expectancy),
        "avg_return_pct": _decimal_to_str(trade_stats.avg_return_pct),
        "total_fees": _decimal_to_str(metrics.total_fees),
        "max_drawdown": _decimal_to_str(metrics.max_drawdown),
        "max_drawdown_pct": _decimal_to_str(equity_stats.max_drawdown_pct),
        "diagnostics": {
            "price_signal_candidates": outcome.diagnostics.price_signal_candidates,
            "gate_passed_signals": outcome.diagnostics.gate_passed_signals,
            "gate_rejected_signals": outcome.diagnostics.gate_rejected_signals,
            "skipped_gap_signals": outcome.diagnostics.skipped_gap_signals,
            "skipped_end_signals": outcome.diagnostics.skipped_end_signals,
            "skipped_warmup_rows": outcome.diagnostics.skipped_warmup_rows,
            "gap_count": outcome.diagnostics.gap_count,
            "max_gap_seconds": outcome.diagnostics.max_gap_seconds,
        },
    }


def _verdict(row: V28ComparisonRow, *, min_trades: int) -> str:
    if row.segment == "full":
        return "descriptive_full_sample_only"
    gated_metrics = row.gated.result.metrics
    baseline_metrics = row.baseline.result.metrics
    gated_trade_stats = compute_trade_statistics(row.gated.result.trades)
    if gated_metrics.round_trips < min_trades:
        return "not_enough_gated_trades"
    if gated_metrics.return_pct <= 0:
        return "rejected_gated_negative_after_costs"
    if gated_trade_stats.profit_factor is None or gated_trade_stats.profit_factor <= Decimal("1"):
        return "rejected_gated_profit_factor_not_above_1"
    if gated_metrics.return_pct <= baseline_metrics.return_pct:
        return "rejected_gate_not_better_than_baseline"
    return "promising_research_only"


def _payload(row: V28ComparisonRow, rank: int, *, min_trades: int) -> dict[str, Any]:
    baseline = _metric_payload(row.baseline, timeframe=row.timeframe)
    gated = _metric_payload(row.gated, timeframe=row.timeframe)
    baseline_return = Decimal(str(row.baseline.result.metrics.return_pct))
    gated_return = Decimal(str(row.gated.result.metrics.return_pct))
    return {
        "rank": rank,
        "timeframe": row.timeframe,
        "segment": row.segment,
        "price_strategy": row.price_strategy,
        "order_book_feature": row.order_book_feature,
        "gate_tail": row.gate_tail,
        "configured_quantile": str(row.configured_quantile),
        "threshold_quantile": str(row.threshold_quantile),
        "entry_threshold": row.threshold,
        "horizon_bars": row.horizon_bars,
        "sample_size": row.sample_size,
        "coverage": {
            "start": row.coverage_start.isoformat(),
            "split_time": row.split_time.isoformat(),
            "end": row.coverage_end.isoformat(),
        },
        "verdict": _verdict(row, min_trades=min_trades),
        "improvement": {
            "return_pct_delta": _decimal_to_str(gated_return - baseline_return),
            "gated_beats_no_trade": gated_return > 0,
            "gated_beats_baseline": gated_return > baseline_return,
        },
        "baseline": baseline,
        "order_book_gated": gated,
        "notes": (
            "Research-only V28 comparison. Baseline and gated runs use the same "
            "price rule, coverage-aware split, costs, and gap-safe timeline. "
            "Only the observed, train-learned order-book gate differs."
        ),
    }


def _sort_key(row: V28ComparisonRow) -> tuple[int, Decimal, Decimal, int]:
    priority = {"validation": 2, "train": 1, "full": 0}.get(row.segment, 0)
    improvement = row.gated.result.metrics.return_pct - row.baseline.result.metrics.return_pct
    return (priority, improvement, row.gated.result.metrics.return_pct, row.gated.result.metrics.round_trips)


def _export_json(path: Path, rows: list[V28ComparisonRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_payload(row, rank, min_trades=min_trades) for rank, row in enumerate(rows, start=1)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("v28_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, rows: list[V28ComparisonRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank", "timeframe", "segment", "price_strategy", "order_book_feature", "gate_tail",
        "configured_quantile", "threshold_quantile", "entry_threshold", "horizon_bars", "sample_size",
        "verdict", "baseline_return_pct", "gated_return_pct", "return_pct_delta", "baseline_round_trips",
        "gated_round_trips", "baseline_profit_factor", "gated_profit_factor", "gated_beats_baseline",
        "gated_skipped_gap_signals", "gated_gate_rejected_signals",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            payload = _payload(row, rank, min_trades=min_trades)
            writer.writerow({
                "rank": rank,
                "timeframe": row.timeframe,
                "segment": row.segment,
                "price_strategy": row.price_strategy,
                "order_book_feature": row.order_book_feature,
                "gate_tail": row.gate_tail,
                "configured_quantile": str(row.configured_quantile),
                "threshold_quantile": str(row.threshold_quantile),
                "entry_threshold": row.threshold,
                "horizon_bars": row.horizon_bars,
                "sample_size": row.sample_size,
                "verdict": payload["verdict"],
                "baseline_return_pct": payload["baseline"]["return_pct"],
                "gated_return_pct": payload["order_book_gated"]["return_pct"],
                "return_pct_delta": payload["improvement"]["return_pct_delta"],
                "baseline_round_trips": payload["baseline"]["round_trips"],
                "gated_round_trips": payload["order_book_gated"]["round_trips"],
                "baseline_profit_factor": payload["baseline"]["profit_factor"],
                "gated_profit_factor": payload["order_book_gated"]["profit_factor"],
                "gated_beats_baseline": payload["improvement"]["gated_beats_baseline"],
                "gated_skipped_gap_signals": payload["order_book_gated"]["diagnostics"]["skipped_gap_signals"],
                "gated_gate_rejected_signals": payload["order_book_gated"]["diagnostics"]["gate_rejected_signals"],
            })
    logger.info("v28_csv_exported", path=str(path), rows=len(rows))


def _build_rows_for_feature(
    *,
    all_rows: list[MarketFeatures],
    timeframe: str,
    feature: str,
    strategies: list[str],
    quantiles: list[Decimal],
    horizons: list[int],
    args: argparse.Namespace,
    symbol: str,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> list[V28ComparisonRow]:
    ordered = sorted(all_rows, key=lambda row: row.close_time)
    observed = rows_with_feature(ordered, feature)
    if len(observed) < args.min_feature_samples:
        logger.info(
            "v28_feature_skipped",
            timeframe=timeframe,
            feature=feature,
            feature_samples=len(observed),
            reason="not_enough_feature_samples",
        )
        return []

    coverage = split_rows_by_feature_coverage(ordered, feature_name=feature, train_ratio=args.train_ratio)
    train_feature_samples = len(rows_with_feature(coverage.train_rows, feature))
    validation_feature_samples = len(rows_with_feature(coverage.validation_rows, feature))
    if train_feature_samples < max(1, args.min_feature_samples // 2):
        logger.info(
            "v28_feature_skipped",
            timeframe=timeframe,
            feature=feature,
            train_feature_samples=train_feature_samples,
            validation_feature_samples=validation_feature_samples,
            reason="not_enough_train_feature_samples",
        )
        return []

    logger.info(
        "v28_coverage_split",
        timeframe=timeframe,
        feature=feature,
        coverage_start=coverage.coverage_start.isoformat(),
        split_time=coverage.split_time.isoformat(),
        coverage_end=coverage.coverage_end.isoformat(),
        train_feature_samples=train_feature_samples,
        validation_feature_samples=validation_feature_samples,
        train_timeline_rows=len(coverage.train_rows),
        validation_timeline_rows=len(coverage.validation_rows),
    )

    result_rows: list[V28ComparisonRow] = []
    segments = {
        "full": coverage.full_rows,
        "train": coverage.train_rows,
        "validation": coverage.validation_rows,
    }
    settings = get_settings()

    for strategy in strategies:
        price_rule = _build_price_rule(args, strategy)
        gate_tail = _gate_tail_for(strategy)
        for configured_quantile in quantiles:
            threshold_quantile = _threshold_quantile(configured_quantile, tail=gate_tail)
            threshold = quantile_threshold(coverage.train_rows, feature, float(threshold_quantile))
            gate = OrderBookGateConfig(feature_name=feature, tail=gate_tail, threshold=threshold)
            for horizon in horizons:
                for segment, segment_rows in segments.items():
                    baseline_config = GatedStrategyConfig(
                        price_rule=price_rule,
                        horizon_bars=horizon,
                        timeframe=timeframe,
                        strategy_name=_config_name(
                            strategy=strategy,
                            feature=feature,
                            tail=gate_tail,
                            configured_quantile=configured_quantile,
                            horizon=horizon,
                            gated=False,
                        ),
                        order_book_gate=None,
                    )
                    gated_config = GatedStrategyConfig(
                        price_rule=price_rule,
                        horizon_bars=horizon,
                        timeframe=timeframe,
                        strategy_name=_config_name(
                            strategy=strategy,
                            feature=feature,
                            tail=gate_tail,
                            configured_quantile=configured_quantile,
                            horizon=horizon,
                            gated=True,
                        ),
                        order_book_gate=gate,
                    )
                    baseline = run_order_book_gated_backtest(
                        rows=segment_rows,
                        config=baseline_config,
                        symbol=symbol,
                        initial_quote_balance=settings.initial_quote_balance,
                        quote_amount=settings.max_order_usdt,
                        fee_rate_pct=fee_rate_pct,
                        slippage_pct=slippage_pct,
                    )
                    gated = run_order_book_gated_backtest(
                        rows=segment_rows,
                        config=gated_config,
                        symbol=symbol,
                        initial_quote_balance=settings.initial_quote_balance,
                        quote_amount=settings.max_order_usdt,
                        fee_rate_pct=fee_rate_pct,
                        slippage_pct=slippage_pct,
                    )
                    result_rows.append(
                        V28ComparisonRow(
                            timeframe=timeframe,
                            segment=segment,
                            price_strategy=strategy,
                            order_book_feature=feature,
                            gate_tail=gate_tail,
                            configured_quantile=configured_quantile,
                            threshold_quantile=threshold_quantile,
                            threshold=threshold,
                            horizon_bars=horizon,
                            sample_size=len(rows_with_feature(segment_rows, feature)),
                            coverage_start=coverage.coverage_start,
                            split_time=coverage.split_time,
                            coverage_end=coverage.coverage_end,
                            baseline=baseline,
                            gated=gated,
                        )
                    )
    return result_rows


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.limit <= 0 or args.min_feature_samples <= 0 or args.min_trades <= 0 or args.top <= 0:
        raise SystemExit("limit, min-feature-samples, min-trades, and top must be positive")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise SystemExit("train ratio must be between 0 and 1")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("fee rate cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("slippage cannot be negative")

    timeframes = _parse_csv(args.timeframes)
    strategies = _parse_csv(args.strategies)
    unknown_strategies = set(strategies) - set(PRICE_STRATEGIES)
    if unknown_strategies:
        raise SystemExit(f"unknown V28 strategies: {','.join(sorted(unknown_strategies))}")
    features = _parse_csv(args.order_book_features)
    quantiles = _parse_quantiles(args.entry_quantiles)
    horizons = _parse_horizons(args.horizons)
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "v28_backtest_started",
        timeframes=",".join(timeframes),
        strategies=",".join(strategies),
        order_book_features=",".join(features),
        horizons=",".join(str(item) for item in horizons),
        entry_quantiles=",".join(str(item) for item in quantiles),
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research-only; no execution; net profitability after modeled costs is the criterion",
    )

    comparison_rows: list[V28ComparisonRow] = []
    for timeframe in timeframes:
        rows, symbol = await _load_rows(timeframe, args)
        for feature in features:
            comparison_rows.extend(
                _build_rows_for_feature(
                    all_rows=rows,
                    timeframe=timeframe,
                    feature=feature,
                    strategies=strategies,
                    quantiles=quantiles,
                    horizons=horizons,
                    args=args,
                    symbol=symbol,
                    fee_rate_pct=fee_rate_pct,
                    slippage_pct=slippage_pct,
                )
            )

    if not comparison_rows:
        raise SystemExit("No V28 rows produced. Check feature coverage and configuration.")

    ranked = sorted(comparison_rows, key=_sort_key, reverse=True)
    logger.info("v28_backtest_finished", rows=len(ranked), note="research only")
    for rank, row in enumerate(ranked[: args.top], start=1):
        payload = _payload(row, rank, min_trades=args.min_trades)
        logger.info(
            "v28_result",
            rank=rank,
            timeframe=row.timeframe,
            segment=row.segment,
            price_strategy=row.price_strategy,
            order_book_feature=row.order_book_feature,
            gate_tail=row.gate_tail,
            threshold=round(row.threshold, 6),
            configured_quantile=str(row.configured_quantile),
            horizon_bars=row.horizon_bars,
            verdict=payload["verdict"],
            baseline_return_pct=payload["baseline"]["return_pct"],
            gated_return_pct=payload["order_book_gated"]["return_pct"],
            return_delta_pct=payload["improvement"]["return_pct_delta"],
            baseline_trades=payload["baseline"]["round_trips"],
            gated_trades=payload["order_book_gated"]["round_trips"],
            gated_profit_factor=payload["order_book_gated"]["profit_factor"],
            skipped_gap_signals=payload["order_book_gated"]["diagnostics"]["skipped_gap_signals"],
        )

    if args.export_json:
        _export_json(args.export_json, ranked, min_trades=args.min_trades)
    if args.export_csv:
        _export_csv(args.export_csv, ranked, min_trades=args.min_trades)


if __name__ == "__main__":
    asyncio.run(main())
