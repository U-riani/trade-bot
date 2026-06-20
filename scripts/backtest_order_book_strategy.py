"""V27.2 gap-safe order-book strategy validation.

This script replaces V27's compressed feature-only replay. It keeps the full
stored candle timeline, skips trades that would cross data gaps, and can test
both high-imbalance momentum entries and low-imbalance contrarian entries.
Research only: no live trading, no execution, no profitability claim.
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
from app.backtesting.order_book_strategy import (
    BacktestDiagnostics,
    OrderBookThresholdConfig,
    buy_and_hold_feature_rows,
    quantile_threshold,
    rows_with_feature,
    run_order_book_threshold_backtest_with_diagnostics,
    split_feature_rows,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.market.features import MarketFeatures
from app.storage.db import Database
from app.storage.repositories import TradingRepository
from scripts.backtest_strategy import _decimal_to_str, _resolve_market_data_source

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class OrderBookStrategyRow:
    timeframe: str
    segment: str
    strategy_name: str
    feature: str
    entry_tail: str
    horizon_bars: int
    entry_quantile: Decimal
    entry_threshold: float
    sample_size: int
    strategy_result: object
    buy_hold_result: object
    diagnostics: BacktestDiagnostics
    notes: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27.2 gap-safe order-book strategy research.")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default=None)
    parser.add_argument("--timeframes", default="1m,5m,15m", help="Comma-separated feature timeframes.")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--features", default="imbalance_top_20,imbalance_top_5")
    parser.add_argument("--horizons", default="1,3,6")
    parser.add_argument("--entry-quantiles", default="0.6,0.7,0.8", help="Upper-tail quantiles. Low tail uses 1-q.")
    parser.add_argument("--entry-tails", default="high,low", help="high, low, or both.")
    parser.add_argument("--min-feature-samples", type=int, default=100)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--train-ratio", type=Decimal, default=Decimal("0.7"))
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
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


def _parse_features(value: str) -> list[str]:
    return _parse_csv(value)


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in _parse_csv(value)]
    if any(item <= 0 for item in result):
        raise SystemExit("horizons must be positive")
    return result


def _parse_quantiles(value: str) -> list[Decimal]:
    result = [Decimal(item) for item in _parse_csv(value)]
    if any(item <= 0 or item >= 1 for item in result):
        raise SystemExit("entry quantiles must be between 0 and 1")
    return result


def _parse_tails(value: str) -> list[str]:
    tails = _parse_csv(value)
    if any(item not in {"high", "low"} for item in tails):
        raise SystemExit("entry tails must be high and/or low")
    return tails


def _strategy_name(feature: str, tail: str, quantile: Decimal, horizon: int) -> str:
    return f"order_book_gap_safe_{feature}_{tail}_q{str(quantile).replace('.', 'p')}_h{horizon}_v27_2"


def _verdict(row: OrderBookStrategyRow, *, min_trades: int) -> str:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    trade_stats = compute_trade_statistics(row.strategy_result.trades)  # type: ignore[attr-defined]
    buy_hold = row.buy_hold_result.metrics.return_pct  # type: ignore[attr-defined]
    if row.segment == "full":
        return "descriptive_full_sample_only"
    if metrics.round_trips < min_trades:
        return "not_enough_trades"
    if metrics.return_pct <= 0:
        return "rejected_validation_negative_after_costs"
    if trade_stats.profit_factor is None or trade_stats.profit_factor <= Decimal("1"):
        return "rejected_profit_factor_not_above_1"
    if metrics.return_pct <= buy_hold:
        return "beats_no_trade_only"
    return "promising_research_only"


def _payload(row: OrderBookStrategyRow, rank: int, *, min_trades: int) -> dict[str, Any]:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    benchmark = row.buy_hold_result.metrics  # type: ignore[attr-defined]
    trade = compute_trade_statistics(row.strategy_result.trades)  # type: ignore[attr-defined]
    equity = compute_equity_statistics(row.strategy_result.equity_curve, bars_per_year=bars_per_year_for_timeframe(row.timeframe))  # type: ignore[attr-defined]
    return {
        "rank": rank,
        "timeframe": row.timeframe,
        "segment": row.segment,
        "strategy_name": row.strategy_name,
        "feature": row.feature,
        "entry_tail": row.entry_tail,
        "horizon_bars": row.horizon_bars,
        "entry_quantile": str(row.entry_quantile),
        "entry_threshold": row.entry_threshold,
        "sample_size": row.sample_size,
        "verdict": _verdict(row, min_trades=min_trades),
        "notes": row.notes,
        "metrics": {
            "round_trips": metrics.round_trips,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "win_rate": metrics.win_rate,
            "return_pct": _decimal_to_str(metrics.return_pct),
            "final_equity": _decimal_to_str(metrics.final_equity),
            "total_fees": _decimal_to_str(metrics.total_fees),
            "max_drawdown": _decimal_to_str(metrics.max_drawdown),
        },
        "edge": {
            "profit_factor": _decimal_to_str(trade.profit_factor),
            "expectancy": _decimal_to_str(trade.expectancy),
            "payoff_ratio": _decimal_to_str(trade.payoff_ratio),
            "avg_return_pct": _decimal_to_str(trade.avg_return_pct),
            "trade_return_sharpe": _decimal_to_str(trade.trade_return_sharpe),
            "max_consecutive_losses": trade.max_consecutive_losses,
            "sharpe_annualized": _decimal_to_str(equity.sharpe),
            "max_drawdown_pct": _decimal_to_str(equity.max_drawdown_pct),
        },
        "benchmark": {
            "buy_hold_order_sized_return_pct": _decimal_to_str(benchmark.return_pct),
            "beats_no_trade": metrics.return_pct > 0,
            "beats_order_sized_buy_hold": metrics.return_pct > benchmark.return_pct,
        },
        "continuity": {
            "total_rows": row.diagnostics.total_rows,
            "feature_observations": row.diagnostics.feature_observations,
            "gap_count": row.diagnostics.gap_count,
            "max_gap_seconds": row.diagnostics.max_gap_seconds,
            "signal_candidates": row.diagnostics.signal_candidates,
            "skipped_gap_signals": row.diagnostics.skipped_gap_signals,
            "skipped_end_signals": row.diagnostics.skipped_end_signals,
        },
    }


def _sort_key(row: OrderBookStrategyRow) -> tuple[int, Decimal, Decimal, int]:
    metrics = row.strategy_result.metrics  # type: ignore[attr-defined]
    pf = compute_trade_statistics(row.strategy_result.trades).profit_factor or Decimal("0")  # type: ignore[attr-defined]
    # Always show held-out validation rows before train/full rows, even when ugly.
    priority = {"validation": 2, "full": 1, "train": 0}.get(row.segment, 0)
    return (priority, metrics.return_pct, pf, metrics.round_trips)


async def _load_rows(timeframe: str, args: argparse.Namespace) -> tuple[list[MarketFeatures], str]:
    settings = get_settings()
    _source, _use_testnet, exchange = _resolve_market_data_source(args.market_data_source)
    db = Database(settings.database_url)
    await db.connect()
    try:
        rows = await TradingRepository(db).load_market_features(
            exchange=exchange, symbol=settings.normalized_symbol, timeframe=timeframe, limit=args.limit
        )
    finally:
        await db.close()
    return rows, settings.normalized_symbol


def _build_rows_for_feature(
    *, rows: list[MarketFeatures], timeframe: str, feature: str, tails: list[str], horizons: list[int],
    quantiles: list[Decimal], args: argparse.Namespace, symbol: str, fee: Decimal, slippage: Decimal,
) -> list[OrderBookStrategyRow]:
    settings = get_settings()
    ordered = sorted(rows, key=lambda row: row.close_time)
    present = rows_with_feature(ordered, feature)
    if len(present) < args.min_feature_samples:
        logger.info("order_book_strategy_feature_skipped", timeframe=timeframe, feature=feature, sample_size=len(present), reason="not_enough_samples")
        return []

    train_rows, validation_rows = split_feature_rows(ordered, train_ratio=args.train_ratio)
    if len(rows_with_feature(train_rows, feature)) < max(1, args.min_feature_samples // 2):
        logger.info("order_book_strategy_feature_skipped", timeframe=timeframe, feature=feature, reason="not_enough_train_feature_samples")
        return []

    segments = {"full": ordered, "train": train_rows, "validation": validation_rows}
    output: list[OrderBookStrategyRow] = []
    for tail in tails:
        for base_quantile in quantiles:
            actual_quantile = base_quantile if tail == "high" else Decimal("1") - base_quantile
            threshold = quantile_threshold(train_rows, feature, float(actual_quantile))
            for horizon in horizons:
                config = OrderBookThresholdConfig(
                    feature_name=feature, entry_threshold=threshold, horizon_bars=horizon,
                    strategy_name=_strategy_name(feature, tail, actual_quantile, horizon),
                    timeframe=timeframe, entry_tail=tail,
                )
                for segment, segment_rows in segments.items():
                    outcome = run_order_book_threshold_backtest_with_diagnostics(
                        rows=segment_rows, config=config, symbol=symbol,
                        initial_quote_balance=settings.initial_quote_balance, quote_amount=settings.max_order_usdt,
                        fee_rate_pct=fee, slippage_pct=slippage,
                    )
                    benchmark = buy_and_hold_feature_rows(
                        rows=segment_rows, symbol=symbol, initial_quote_balance=settings.initial_quote_balance,
                        quote_amount=settings.max_order_usdt, fee_rate_pct=fee, slippage_pct=slippage,
                    )
                    output.append(OrderBookStrategyRow(
                        timeframe=timeframe, segment=segment, strategy_name=config.strategy_name, feature=feature,
                        entry_tail=tail, horizon_bars=horizon, entry_quantile=actual_quantile,
                        entry_threshold=threshold, sample_size=len(rows_with_feature(segment_rows, feature)),
                        strategy_result=outcome.result, buy_hold_result=benchmark, diagnostics=outcome.diagnostics,
                        notes=("Gap-safe research-only fixed-horizon long. Signal at candle i enters at i+1 only when every "
                               "timestamp through exit is contiguous; threshold learned from train rows only."),
                    ))
    return output


def _export_json(path: Path, rows: list[OrderBookStrategyRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [_payload(row, rank, min_trades=min_trades) for rank, row in enumerate(rows, start=1)]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("order_book_strategy_json_exported", path=str(path), rows=len(payload))


def _export_csv(path: Path, rows: list[OrderBookStrategyRow], *, min_trades: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "timeframe", "segment", "strategy_name", "feature", "entry_tail", "horizon_bars", "entry_quantile", "entry_threshold", "sample_size", "verdict", "return_pct", "round_trips", "profit_factor", "buy_hold_return_pct", "gap_count", "skipped_gap_signals", "skipped_end_signals"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            p = _payload(row, rank, min_trades=min_trades)
            writer.writerow({
                "rank": rank, "timeframe": row.timeframe, "segment": row.segment,
                "strategy_name": row.strategy_name, "feature": row.feature, "entry_tail": row.entry_tail,
                "horizon_bars": row.horizon_bars, "entry_quantile": str(row.entry_quantile),
                "entry_threshold": row.entry_threshold, "sample_size": row.sample_size, "verdict": p["verdict"],
                "return_pct": p["metrics"]["return_pct"], "round_trips": p["metrics"]["round_trips"],
                "profit_factor": p["edge"]["profit_factor"], "buy_hold_return_pct": p["benchmark"]["buy_hold_order_sized_return_pct"],
                "gap_count": p["continuity"]["gap_count"], "skipped_gap_signals": p["continuity"]["skipped_gap_signals"],
                "skipped_end_signals": p["continuity"]["skipped_end_signals"],
            })
    logger.info("order_book_strategy_csv_exported", path=str(path), rows=len(rows))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    if args.limit <= 0 or args.min_feature_samples <= 0 or args.min_trades <= 0 or args.top <= 0:
        raise SystemExit("limit, min-feature-samples, min-trades, and top must be positive")
    if args.train_ratio <= 0 or args.train_ratio >= 1:
        raise SystemExit("train-ratio must be between 0 and 1")
    features, timeframes = _parse_features(args.features), _parse_csv(args.timeframes)
    horizons, quantiles, tails = _parse_ints(args.horizons), _parse_quantiles(args.entry_quantiles), _parse_tails(args.entry_tails)
    settings = get_settings()
    fee = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    all_rows: list[OrderBookStrategyRow] = []
    for timeframe in timeframes:
        rows, symbol = await _load_rows(timeframe, args)
        logger.info("order_book_strategy_features_loaded", timeframe=timeframe, rows=len(rows), features=",".join(features))
        for feature in features:
            all_rows.extend(_build_rows_for_feature(
                rows=rows, timeframe=timeframe, feature=feature, tails=tails, horizons=horizons, quantiles=quantiles,
                args=args, symbol=symbol, fee=fee, slippage=slippage,
            ))

    if not all_rows:
        raise SystemExit("No gap-safe strategy rows produced. Check feature coverage.")
    ranked = sorted(all_rows, key=_sort_key, reverse=True)
    logger.info("order_book_strategy_backtest_finished", rows=len(ranked), note="gap-safe research only")
    for rank, row in enumerate(ranked[:args.top], start=1):
        p = _payload(row, rank, min_trades=args.min_trades)
        logger.info(
            "order_book_strategy_result", rank=rank, timeframe=row.timeframe, segment=row.segment,
            feature=row.feature, entry_tail=row.entry_tail, horizon_bars=row.horizon_bars,
            entry_quantile=str(row.entry_quantile), threshold=round(row.entry_threshold, 6), sample_size=row.sample_size,
            verdict=p["verdict"], return_pct=p["metrics"]["return_pct"], round_trips=p["metrics"]["round_trips"],
            profit_factor=p["edge"]["profit_factor"], gap_count=p["continuity"]["gap_count"],
            skipped_gap_signals=p["continuity"]["skipped_gap_signals"],
        )
    if args.export_json:
        _export_json(args.export_json, ranked, min_trades=args.min_trades)
    if args.export_csv:
        _export_csv(args.export_csv, ranked, min_trades=args.min_trades)


if __name__ == "__main__":
    asyncio.run(main())
