"""V31.1 diagnostics for the V31 price-only walk-forward catalog.

V31 intentionally emits only the selected candidate in each fold. When no
candidate qualifies, that concise output is correct but not explanatory enough.
This module records every pre-registered candidate's train-window metrics and
rejection reason without changing the strategy catalog or selection rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from app.backtesting.walk_forward_price import (
    PriceCandidate,
    SimulationResult,
    WalkForwardFold,
    build_signal_index_map,
    simulate_fixed_horizon,
)
from app.market.models import Candle


@dataclass(frozen=True, slots=True)
class CandidateTrainDiagnostic:
    """One candidate evaluated on one training fold."""

    fold_number: int
    candidate: PriceCandidate
    signal_count_in_window: int
    result: SimulationResult
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def payload(self, *, rank: int) -> dict[str, object]:
        return {
            "rank": rank,
            "fold_number": self.fold_number,
            "candidate": self.candidate.payload(),
            "signal_count_in_window": self.signal_count_in_window,
            "eligible": self.eligible,
            "rejection_reasons": list(self.rejection_reasons),
            "train_result": self.result.payload(),
        }


def train_rejection_reasons(
    result: SimulationResult,
    *,
    min_train_trades: int,
) -> tuple[str, ...]:
    """Return every reason a train result cannot be selected by V31.

    Reasons are deliberately cumulative. A low-trade candidate can also be
    unprofitable, and hiding one of those facts leads people to "fix" the wrong
    problem with another parameter sweep.
    """

    if min_train_trades <= 0:
        raise ValueError("min_train_trades must be positive")

    reasons: list[str] = []
    if result.round_trips < min_train_trades:
        reasons.append("not_enough_train_trades")
    if result.net_pnl <= 0:
        reasons.append("non_positive_train_net_pnl_after_costs")
    profit_factor = result.profit_factor
    if profit_factor is None or profit_factor <= Decimal("1"):
        reasons.append("profit_factor_not_above_1")
    return tuple(reasons)


def diagnose_train_fold(
    candles: list[Candle],
    *,
    fold: WalkForwardFold,
    candidates: Iterable[PriceCandidate],
    signal_index_map: dict[str, tuple[int, ...]] | None,
    initial_equity: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
    min_train_trades: int,
) -> tuple[CandidateTrainDiagnostic, ...]:
    """Evaluate every fixed V31 candidate on one train period.

    ``signal_index_map`` can be provided by the caller and re-used across folds.
    This is important because it preserves identical completed-bar signals for
    every evaluation and avoids hidden per-fold feature recalculation.
    """

    ordered = sorted(candles, key=lambda candle: candle.open_time)
    candidate_list = tuple(candidates)
    signals = signal_index_map or build_signal_index_map(ordered, candidate_list)
    diagnostics: list[CandidateTrainDiagnostic] = []

    for candidate in candidate_list:
        indexes = signals.get(candidate.name, ())
        signal_count = sum(fold.train_start <= index < fold.train_end for index in indexes)
        result = simulate_fixed_horizon(
            ordered,
            candidate=candidate,
            signal_indexes=indexes,
            start_index=fold.train_start,
            end_index=fold.train_end,
            initial_equity=initial_equity,
            quote_amount=quote_amount,
            fee_rate_pct=fee_rate_pct,
            slippage_pct=slippage_pct,
        )
        reasons = train_rejection_reasons(result, min_train_trades=min_train_trades)
        diagnostics.append(
            CandidateTrainDiagnostic(
                fold_number=fold.fold_number,
                candidate=candidate,
                signal_count_in_window=signal_count,
                result=result,
                eligible=not reasons,
                rejection_reasons=reasons,
            )
        )

    return tuple(diagnostics)


def diagnostic_rank_key(item: CandidateTrainDiagnostic) -> tuple[int, Decimal, Decimal, int, int]:
    """Rank without changing V31's selection criteria.

    Eligible candidates appear first. Among all candidates, higher post-cost PnL,
    profit factor, completed trades, then lower signal overlap skips rank higher.
    """

    profit_factor = item.result.profit_factor or Decimal("0")
    return (
        1 if item.eligible else 0,
        item.result.net_pnl,
        profit_factor,
        item.result.round_trips,
        -item.result.skipped_overlap_signals,
    )


def rank_diagnostics(
    diagnostics: Iterable[CandidateTrainDiagnostic],
) -> tuple[CandidateTrainDiagnostic, ...]:
    return tuple(sorted(diagnostics, key=diagnostic_rank_key, reverse=True))


def aggregate_candidate_diagnostics(
    diagnostics: Iterable[CandidateTrainDiagnostic],
) -> list[dict[str, object]]:
    """Aggregate per-fold diagnostics by candidate for the report."""

    grouped: dict[str, list[CandidateTrainDiagnostic]] = {}
    for item in diagnostics:
        grouped.setdefault(item.candidate.name, []).append(item)

    output: list[dict[str, object]] = []
    for name, rows in grouped.items():
        candidate = rows[0].candidate
        folds = len(rows)
        rejection_counts: dict[str, int] = {}
        for row in rows:
            for reason in row.rejection_reasons:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

        total_net_pnl = sum((row.result.net_pnl for row in rows), Decimal("0"))
        total_trades = sum(row.result.round_trips for row in rows)
        total_fees = sum((row.result.total_fees for row in rows), Decimal("0"))
        positive_net_pnl_folds = sum(row.result.net_pnl > 0 for row in rows)
        eligible_folds = sum(row.eligible for row in rows)
        output.append(
            {
                "candidate": candidate.payload(),
                "folds_evaluated": folds,
                "eligible_folds": eligible_folds,
                "positive_net_pnl_folds": positive_net_pnl_folds,
                "total_train_net_pnl": str(total_net_pnl),
                "average_train_net_pnl": str(total_net_pnl / folds) if folds else "0",
                "total_train_trades": total_trades,
                "average_train_trades": total_trades / folds if folds else 0.0,
                "total_train_fees": str(total_fees),
                "rejection_counts": rejection_counts,
            }
        )

    return sorted(
        output,
        key=lambda row: (
            int(row["eligible_folds"]),
            Decimal(str(row["total_train_net_pnl"])),
            int(row["total_train_trades"]),
        ),
        reverse=True,
    )
