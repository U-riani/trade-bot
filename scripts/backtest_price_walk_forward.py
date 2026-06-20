"""Run V31 deterministic price-only walk-forward research.

V31 uses a small pre-registered candidate catalog.  For each rolling fold it
selects a candidate only from train performance, then measures it only on the
following validation window.  The purpose is cumulative net PnL after many
trades, not a fantasy in which every trade must win.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.walk_forward_price import (
    WalkForwardOutcome,
    build_walk_forward_folds,
    default_price_candidates,
    run_walk_forward,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from scripts.backtest_strategy import _load_candles

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V31 price-only walk-forward research.")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default="production")
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--validation-bars", type=int, default=10000)
    parser.add_argument("--step-bars", type=int, default=10000)
    parser.add_argument("--min-train-trades", type=int, default=5)
    parser.add_argument("--min-validation-trades", type=int, default=10)
    parser.add_argument("--min-selected-folds", type=int, default=3)
    parser.add_argument("--min-profitable-folds", type=int, default=2)
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _validate(args: argparse.Namespace) -> None:
    positive = (
        args.limit,
        args.train_bars,
        args.validation_bars,
        args.step_bars,
        args.min_train_trades,
        args.min_validation_trades,
        args.min_selected_folds,
        args.min_profitable_folds,
    )
    if any(value <= 0 for value in positive):
        raise SystemExit("all bar and trade-count arguments must be positive")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")


def _payload(outcome: WalkForwardOutcome, *, candidates: int, protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_name": "v31_price_only_walk_forward",
        "research_only": True,
        "nautilus_trader_reference": {
            "used_as": "event-driven research discipline reference",
            "runtime_dependency": False,
            "reason": (
                "V31 retains the project's existing deterministic DB/candle pipeline so results remain "
                "comparable. A separate NautilusTrader parity run is only appropriate after a candidate "
                "survives this walk-forward gate."
            ),
        },
        "protocol": protocol,
        "candidate_catalog_size": candidates,
        "outcome": outcome.payload(),
    }


def _export_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("v31_json_exported", path=str(path))


def _export_csv(path: Path, outcome: WalkForwardOutcome) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold_number",
        "selected_candidate",
        "family",
        "selection_reason",
        "train_return_pct",
        "train_round_trips",
        "train_profit_factor",
        "validation_return_pct",
        "validation_net_pnl",
        "validation_round_trips",
        "validation_profit_factor",
        "validation_max_drawdown",
        "buy_and_hold_return_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for fold in outcome.folds:
            train = fold.train_result
            validation = fold.validation_result
            writer.writerow(
                {
                    "fold_number": fold.fold.fold_number,
                    "selected_candidate": None if fold.selected is None else fold.selected.name,
                    "family": None if fold.selected is None else fold.selected.family,
                    "selection_reason": fold.selection_reason,
                    "train_return_pct": None if train is None else str(train.return_pct),
                    "train_round_trips": None if train is None else train.round_trips,
                    "train_profit_factor": None if train is None or train.profit_factor is None else str(train.profit_factor),
                    "validation_return_pct": None if validation is None else str(validation.return_pct),
                    "validation_net_pnl": None if validation is None else str(validation.net_pnl),
                    "validation_round_trips": None if validation is None else validation.round_trips,
                    "validation_profit_factor": (
                        None if validation is None or validation.profit_factor is None else str(validation.profit_factor)
                    ),
                    "validation_max_drawdown": None if validation is None else str(validation.max_drawdown),
                    "buy_and_hold_return_pct": (
                        None if fold.buy_and_hold_return_pct is None else str(fold.buy_and_hold_return_pct)
                    ),
                }
            )
    logger.info("v31_csv_exported", path=str(path), folds=len(outcome.folds))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    _validate(args)
    settings = get_settings()
    candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not candles:
        raise SystemExit("No candles available for V31")

    folds = build_walk_forward_folds(
        len(candles),
        train_bars=args.train_bars,
        validation_bars=args.validation_bars,
        step_bars=args.step_bars,
    )
    if not folds:
        raise SystemExit("Not enough candles for one complete train/validation fold")

    candidates = default_price_candidates()
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct
    protocol = {
        "signal_timing": "completed 5m/15m bars only",
        "entry_timing": "next contiguous 1m open",
        "exit_timing": "fixed horizon at 1m close",
        "cost_model": {"fee_rate_pct_per_side": str(fee_rate_pct), "slippage_pct_per_side": str(slippage_pct)},
        "walk_forward": {
            "train_bars": args.train_bars,
            "validation_bars": args.validation_bars,
            "step_bars": args.step_bars,
            "fold_count": len(folds),
        },
        "pass_criteria": {
            "minimum_validation_trades": args.min_validation_trades,
            "minimum_selected_folds": args.min_selected_folds,
            "minimum_profitable_validation_folds": args.min_profitable_folds,
            "net_pnl": "> 0",
            "profit_factor": "> 1",
            "largest_winner_share": "<= 50% of gross profit",
            "benchmark": "must beat cumulative buy-and-hold over validation folds",
        },
    }
    logger.info(
        "v31_backtest_started",
        candles=len(candles),
        candidates=len(candidates),
        folds=len(folds),
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research-only; candidate selection is train-only",
    )
    outcome = run_walk_forward(
        candles,
        candidates=candidates,
        folds=folds,
        initial_equity=settings.initial_quote_balance,
        quote_amount=settings.max_order_usdt,
        fee_rate_pct=fee_rate_pct,
        slippage_pct=slippage_pct,
        min_train_trades=args.min_train_trades,
        min_validation_trades=args.min_validation_trades,
        min_selected_folds=args.min_selected_folds,
        min_profitable_folds=args.min_profitable_folds,
    )
    aggregate = outcome.validation_result
    logger.info(
        "v31_walk_forward_finished",
        verdict=outcome.verdict,
        selected_folds=outcome.selected_fold_count,
        profitable_validation_folds=outcome.profitable_validation_folds,
        validation_trades=aggregate.round_trips,
        validation_net_pnl=str(aggregate.net_pnl),
        validation_return_pct=str(aggregate.return_pct),
        validation_profit_factor=None if aggregate.profit_factor is None else str(aggregate.profit_factor),
        cumulative_buy_and_hold_return_pct=str(outcome.cumulative_buy_and_hold_return_pct),
        note="research only; no execution decision is produced",
    )
    for fold in outcome.folds:
        logger.info(
            "v31_fold",
            fold=fold.fold.fold_number,
            selected=None if fold.selected is None else fold.selected.name,
            reason=fold.selection_reason,
            train_return_pct=None if fold.train_result is None else str(fold.train_result.return_pct),
            validation_return_pct=(
                None if fold.validation_result is None else str(fold.validation_result.return_pct)
            ),
            validation_trades=None if fold.validation_result is None else fold.validation_result.round_trips,
        )

    payload = _payload(outcome, candidates=len(candidates), protocol=protocol)
    if args.export_json:
        _export_json(args.export_json, payload)
    if args.export_csv:
        _export_csv(args.export_csv, outcome)


if __name__ == "__main__":
    asyncio.run(main())
