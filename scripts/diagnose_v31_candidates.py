"""V31.1: expose the train-window outcome of every V31 candidate.

This diagnostic does not perform validation trades and does not create a new
strategy. It explains why V31 selected no candidate, or why a selected candidate
was preferred, using the exact same completed-bar signal map and cost model.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import Counter
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.backtesting.walk_forward_diagnostics import (
    aggregate_candidate_diagnostics,
    diagnose_train_fold,
    rank_diagnostics,
)
from app.backtesting.walk_forward_price import (
    build_signal_index_map,
    build_walk_forward_folds,
    default_price_candidates,
)
from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from scripts.backtest_strategy import _load_candles

logger = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V31.1 candidate-level training diagnostics.")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--source", choices=("auto", "db", "rest"), default="db")
    parser.add_argument("--market-data-source", choices=("production", "testnet"), default="production")
    parser.add_argument("--train-bars", type=int, default=30000)
    parser.add_argument("--validation-bars", type=int, default=10000)
    parser.add_argument("--step-bars", type=int, default=10000)
    parser.add_argument("--min-train-trades", type=int, default=5)
    parser.add_argument("--fee-rate-pct", type=Decimal, default=None)
    parser.add_argument("--slippage-pct", type=Decimal, default=None)
    parser.add_argument("--top-per-fold", type=int, default=5)
    parser.add_argument("--export-json", type=Path, default=None)
    parser.add_argument("--export-csv", type=Path, default=None)
    return parser


def _validate(args: argparse.Namespace) -> None:
    integers = (args.limit, args.train_bars, args.validation_bars, args.step_bars, args.min_train_trades, args.top_per_fold)
    if any(value <= 0 for value in integers):
        raise SystemExit("all bar, trade, and top arguments must be positive")
    if args.fee_rate_pct is not None and args.fee_rate_pct < 0:
        raise SystemExit("--fee-rate-pct cannot be negative")
    if args.slippage_pct is not None and args.slippage_pct < 0:
        raise SystemExit("--slippage-pct cannot be negative")


def _fold_payload(fold, ranked, *, top_per_fold: int) -> dict[str, Any]:
    reason_counts = Counter(reason for item in ranked for reason in item.rejection_reasons)
    return {
        "fold_number": fold.fold_number,
        "train": {"start_index": fold.train_start, "end_index_exclusive": fold.train_end},
        "validation_reserved": {"start_index": fold.validation_start, "end_index_exclusive": fold.validation_end},
        "eligible_candidate_count": sum(item.eligible for item in ranked),
        "rejection_counts": dict(sorted(reason_counts.items())),
        "top_candidates": [item.payload(rank=index) for index, item in enumerate(ranked[:top_per_fold], start=1)],
        "all_candidates": [item.payload(rank=index) for index, item in enumerate(ranked, start=1)],
    }


def _export_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("v31_1_json_exported", path=str(path))


def _export_csv(path: Path, fold_payloads: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fold_number", "rank", "candidate", "family", "horizon_bars", "signal_count_in_window",
        "eligible", "rejection_reasons", "train_net_pnl", "train_return_pct", "train_round_trips",
        "train_profit_factor", "train_total_fees", "train_max_drawdown", "skipped_gap_signals",
        "skipped_end_signals", "skipped_overlap_signals",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for fold in fold_payloads:
            for item in fold["all_candidates"]:
                result = item["train_result"]
                candidate = item["candidate"]
                writer.writerow(
                    {
                        "fold_number": fold["fold_number"],
                        "rank": item["rank"],
                        "candidate": candidate["name"],
                        "family": candidate["family"],
                        "horizon_bars": candidate["horizon_bars"],
                        "signal_count_in_window": item["signal_count_in_window"],
                        "eligible": item["eligible"],
                        "rejection_reasons": ";".join(item["rejection_reasons"]),
                        "train_net_pnl": result["net_pnl"],
                        "train_return_pct": result["return_pct"],
                        "train_round_trips": result["round_trips"],
                        "train_profit_factor": result["profit_factor"],
                        "train_total_fees": result["total_fees"],
                        "train_max_drawdown": result["max_drawdown"],
                        "skipped_gap_signals": result["skipped_gap_signals"],
                        "skipped_end_signals": result["skipped_end_signals"],
                        "skipped_overlap_signals": result["skipped_overlap_signals"],
                    }
                )
    logger.info("v31_1_csv_exported", path=str(path), folds=len(fold_payloads))


async def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _parser().parse_args(argv)
    _validate(args)
    settings = get_settings()
    candles = await _load_candles(args.source, args.limit, args.market_data_source)
    if not candles:
        raise SystemExit("No candles available for V31.1")

    folds = build_walk_forward_folds(
        len(candles),
        train_bars=args.train_bars,
        validation_bars=args.validation_bars,
        step_bars=args.step_bars,
    )
    if not folds:
        raise SystemExit("Not enough candles for one complete V31.1 fold")

    candidates = default_price_candidates()
    signals = build_signal_index_map(candles, candidates)
    fee_rate_pct = args.fee_rate_pct if args.fee_rate_pct is not None else settings.backtest_fee_rate_pct
    slippage_pct = args.slippage_pct if args.slippage_pct is not None else settings.backtest_slippage_pct

    logger.info(
        "v31_1_diagnostics_started",
        candles=len(candles),
        candidates=len(candidates),
        folds=len(folds),
        fee_rate_pct=str(fee_rate_pct),
        slippage_pct=str(slippage_pct),
        note="research-only; reports all train candidates without modifying V31 selection",
    )

    fold_payloads: list[dict[str, Any]] = []
    all_diagnostics = []
    for fold in folds:
        diagnostics = diagnose_train_fold(
            candles,
            fold=fold,
            candidates=candidates,
            signal_index_map=signals,
            initial_equity=settings.initial_quote_balance,
            quote_amount=settings.max_order_usdt,
            fee_rate_pct=fee_rate_pct,
            slippage_pct=slippage_pct,
            min_train_trades=args.min_train_trades,
        )
        ranked = rank_diagnostics(diagnostics)
        all_diagnostics.extend(ranked)
        fold_data = _fold_payload(fold, ranked, top_per_fold=args.top_per_fold)
        fold_payloads.append(fold_data)
        top = ranked[0]
        logger.info(
            "v31_1_fold_top_candidate",
            fold=fold.fold_number,
            candidate=top.candidate.name,
            eligible=top.eligible,
            train_net_pnl=str(top.result.net_pnl),
            train_profit_factor=None if top.result.profit_factor is None else str(top.result.profit_factor),
            train_trades=top.result.round_trips,
            rejection_reasons=",".join(top.rejection_reasons) or "none",
        )

    aggregate = aggregate_candidate_diagnostics(all_diagnostics)
    payload: dict[str, Any] = {
        "strategy_name": "v31_1_candidate_training_diagnostics",
        "research_only": True,
        "purpose": "Explain V31 candidate failures before defining another strategy family.",
        "protocol": {
            "uses_same_candidate_catalog_as": "v31_price_only_walk_forward",
            "uses_same_signal_timing": "completed 5m/15m bars only; next contiguous 1m open entry",
            "uses_same_cost_model": {
                "fee_rate_pct_per_side": str(fee_rate_pct),
                "slippage_pct_per_side": str(slippage_pct),
            },
            "min_train_trades": args.min_train_trades,
            "fold_count": len(folds),
        },
        "folds": fold_payloads,
        "candidate_aggregate": aggregate,
    }
    logger.info(
        "v31_1_diagnostics_finished",
        eligible_fold_candidates=sum(item.eligible for item in all_diagnostics),
        best_candidate=aggregate[0]["candidate"]["name"] if aggregate else None,
        note="research only; no execution decision is produced",
    )
    if args.export_json:
        _export_json(args.export_json, payload)
    if args.export_csv:
        _export_csv(args.export_csv, fold_payloads)


if __name__ == "__main__":
    asyncio.run(main())
