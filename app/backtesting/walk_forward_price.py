"""V31 deterministic price-only walk-forward research lab.

This module deliberately separates candidate selection (train windows) from
measurement (the following validation windows).  The design borrows the
research-to-execution discipline used by event-driven engines such as
NautilusTrader:

* signals are evaluated only after a bar closes;
* entries occur at the next 1m bar open;
* exits are deterministic fixed-horizon closes;
* fees, slippage, gaps and unavailable windows are explicit;
* no candidate sees validation performance while it is selected.

It is research-only.  A positive report is evidence to investigate further,
not a permission slip for live money.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
from decimal import Decimal
from math import sqrt
from typing import Iterable

from app.backtesting.resample import resample_candles
from app.market.indicators import ema, rsi
from app.market.models import Candle


@dataclass(frozen=True, slots=True)
class PriceCandidate:
    """One deliberately small, pre-registered price strategy candidate."""

    name: str
    family: str
    horizon_bars: int
    pullback_rsi_max: float | None = None
    mean_reversion_z: float | None = None
    mean_reversion_rsi_max: float | None = None
    breakout_lookback: int | None = None

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_number: int
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    entry_time: object
    exit_time: object
    entry_price: Decimal
    exit_price: Decimal
    quote_amount: Decimal
    quantity: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    pnl: Decimal
    candidate_name: str

    @property
    def return_pct(self) -> Decimal:
        denominator = self.quote_amount + self.entry_fee
        if denominator <= 0:
            return Decimal("0")
        return self.pnl / denominator * Decimal("100")


@dataclass(frozen=True, slots=True)
class SimulationResult:
    initial_equity: Decimal
    final_equity: Decimal
    trades: tuple[SimulatedTrade, ...]
    total_fees: Decimal
    max_drawdown: Decimal
    skipped_gap_signals: int
    skipped_end_signals: int
    skipped_overlap_signals: int

    @property
    def net_pnl(self) -> Decimal:
        return self.final_equity - self.initial_equity

    @property
    def return_pct(self) -> Decimal:
        if self.initial_equity <= 0:
            return Decimal("0")
        return self.net_pnl / self.initial_equity * Decimal("100")

    @property
    def round_trips(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for trade in self.trades if trade.pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for trade in self.trades if trade.pnl < 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.round_trips if self.round_trips else 0.0

    @property
    def gross_profit(self) -> Decimal:
        return sum((trade.pnl for trade in self.trades if trade.pnl > 0), Decimal("0"))

    @property
    def gross_loss(self) -> Decimal:
        return -sum((trade.pnl for trade in self.trades if trade.pnl < 0), Decimal("0"))

    @property
    def profit_factor(self) -> Decimal | None:
        if self.gross_loss <= 0:
            return None if self.gross_profit <= 0 else Decimal("Infinity")
        return self.gross_profit / self.gross_loss

    @property
    def largest_winner_share(self) -> Decimal | None:
        if self.gross_profit <= 0:
            return None
        largest = max((trade.pnl for trade in self.trades if trade.pnl > 0), default=Decimal("0"))
        return largest / self.gross_profit

    def payload(self) -> dict[str, object]:
        return {
            "initial_equity": str(self.initial_equity),
            "final_equity": str(self.final_equity),
            "net_pnl": str(self.net_pnl),
            "return_pct": str(self.return_pct),
            "round_trips": self.round_trips,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": None if self.profit_factor is None else str(self.profit_factor),
            "total_fees": str(self.total_fees),
            "max_drawdown": str(self.max_drawdown),
            "largest_winner_share": (
                None if self.largest_winner_share is None else str(self.largest_winner_share)
            ),
            "skipped_gap_signals": self.skipped_gap_signals,
            "skipped_end_signals": self.skipped_end_signals,
            "skipped_overlap_signals": self.skipped_overlap_signals,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate: PriceCandidate
    train_result: SimulationResult


@dataclass(frozen=True, slots=True)
class FoldOutcome:
    fold: WalkForwardFold
    selected: PriceCandidate | None
    selection_reason: str
    train_result: SimulationResult | None
    validation_result: SimulationResult | None
    buy_and_hold_return_pct: Decimal | None

    def payload(self) -> dict[str, object]:
        return {
            "fold_number": self.fold.fold_number,
            "train": {"start_index": self.fold.train_start, "end_index_exclusive": self.fold.train_end},
            "validation": {
                "start_index": self.fold.validation_start,
                "end_index_exclusive": self.fold.validation_end,
            },
            "selected_candidate": None if self.selected is None else self.selected.payload(),
            "selection_reason": self.selection_reason,
            "train_result": None if self.train_result is None else self.train_result.payload(),
            "validation_result": (
                None if self.validation_result is None else self.validation_result.payload()
            ),
            "buy_and_hold_return_pct": (
                None if self.buy_and_hold_return_pct is None else str(self.buy_and_hold_return_pct)
            ),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardOutcome:
    folds: tuple[FoldOutcome, ...]
    validation_result: SimulationResult
    selected_fold_count: int
    profitable_validation_folds: int
    cumulative_buy_and_hold_return_pct: Decimal
    verdict: str

    def payload(self) -> dict[str, object]:
        return {
            "selected_fold_count": self.selected_fold_count,
            "profitable_validation_folds": self.profitable_validation_folds,
            "cumulative_buy_and_hold_return_pct": str(self.cumulative_buy_and_hold_return_pct),
            "aggregate_validation": self.validation_result.payload(),
            "verdict": self.verdict,
            "folds": [fold.payload() for fold in self.folds],
        }


def default_price_candidates() -> tuple[PriceCandidate, ...]:
    """Return the fixed V31 catalog.

    The catalog is intentionally narrow.  It is not an optimizer that keeps
    trying knobs until a random chart says something flattering.
    """

    candidates: list[PriceCandidate] = []
    for rsi_max in (45.0, 50.0):
        for horizon in (15, 30, 60):
            candidates.append(
                PriceCandidate(
                    name=f"trend_pullback_rsi{int(rsi_max)}_h{horizon}",
                    family="trend_pullback",
                    horizon_bars=horizon,
                    pullback_rsi_max=rsi_max,
                )
            )
    for z_threshold in (1.5, 2.0):
        for rsi_max in (35.0, 40.0):
            for horizon in (15, 30):
                candidates.append(
                    PriceCandidate(
                        name=f"mean_reversion_z{str(z_threshold).replace('.', 'p')}_rsi{int(rsi_max)}_h{horizon}",
                        family="mean_reversion",
                        horizon_bars=horizon,
                        mean_reversion_z=z_threshold,
                        mean_reversion_rsi_max=rsi_max,
                    )
                )
    for lookback in (20, 40):
        for horizon in (30, 60):
            candidates.append(
                PriceCandidate(
                    name=f"breakout_lb{lookback}_h{horizon}",
                    family="breakout",
                    horizon_bars=horizon,
                    breakout_lookback=lookback,
                )
            )
    return tuple(candidates)


def build_walk_forward_folds(
    candle_count: int,
    *,
    train_bars: int,
    validation_bars: int,
    step_bars: int,
) -> tuple[WalkForwardFold, ...]:
    if min(train_bars, validation_bars, step_bars) <= 0:
        raise ValueError("train_bars, validation_bars and step_bars must be positive")

    folds: list[WalkForwardFold] = []
    start = 0
    fold_number = 1
    while start + train_bars + validation_bars <= candle_count:
        train_end = start + train_bars
        folds.append(
            WalkForwardFold(
                fold_number=fold_number,
                train_start=start,
                train_end=train_end,
                validation_start=train_end,
                validation_end=train_end + validation_bars,
            )
        )
        fold_number += 1
        start += step_bars
    return tuple(folds)


def build_signal_index_map(
    candles: list[Candle], candidates: Iterable[PriceCandidate]
) -> dict[str, tuple[int, ...]]:
    """Build each candidate's signals from completed bars only.

    Signals use a completed 5m or 15m bar and map to its matching 1m close.  A
    simulator subsequently enters at the NEXT 1m bar open, so this function
    cannot manufacture a same-bar fill.
    """

    ordered = sorted(candles, key=lambda candle: candle.open_time)
    if not ordered:
        return {candidate.name: () for candidate in candidates}

    five = resample_candles(ordered, target_timeframe="5m")
    fifteen = resample_candles(ordered, target_timeframe="15m")
    by_close_time = {candle.close_time: index for index, candle in enumerate(ordered)}

    trend_ok = _trend_flags(fifteen)
    five_trend_index = _asof_indexes(
        [candle.close_time for candle in five], [candle.close_time for candle in fifteen]
    )
    five_signals = _five_minute_signal_features(five, trend_ok, five_trend_index)
    breakout_signals = _breakout_signal_features(fifteen, trend_ok)

    result: dict[str, tuple[int, ...]] = {}
    for candidate in candidates:
        if candidate.family == "trend_pullback":
            signal_times = [
                item.close_time
                for item in five_signals
                if item.trend_ok
                and item.close_below_fast
                and item.close_above_slow
                and item.rsi_value is not None
                and 35.0 <= item.rsi_value <= (candidate.pullback_rsi_max or 0.0)
            ]
        elif candidate.family == "mean_reversion":
            signal_times = [
                item.close_time
                for item in five_signals
                if item.trend_ok
                and item.close_above_slow
                and item.rsi_value is not None
                and item.z_score is not None
                and item.rsi_value <= (candidate.mean_reversion_rsi_max or 0.0)
                and item.z_score <= -(candidate.mean_reversion_z or 0.0)
            ]
        elif candidate.family == "breakout":
            lookback = candidate.breakout_lookback or 0
            signal_times = [
                item.close_time
                for item in breakout_signals
                if item.trend_ok and item.breakout_windows.get(lookback, False)
            ]
        else:
            raise ValueError(f"Unknown V31 candidate family: {candidate.family}")

        result[candidate.name] = tuple(
            sorted(by_close_time[time] for time in signal_times if time in by_close_time)
        )
    return result


@dataclass(frozen=True, slots=True)
class _FiveSignalFeature:
    close_time: object
    trend_ok: bool
    close_below_fast: bool
    close_above_slow: bool
    rsi_value: float | None
    z_score: float | None


@dataclass(frozen=True, slots=True)
class _BreakoutSignalFeature:
    close_time: object
    trend_ok: bool
    breakout_windows: dict[int, bool]


def _asof_indexes(source_times: list[object], target_times: list[object]) -> list[int | None]:
    """Return latest target index closed at each source timestamp."""

    result: list[int | None] = []
    for timestamp in source_times:
        index = bisect_right(target_times, timestamp) - 1
        result.append(index if index >= 0 else None)
    return result


def _trend_flags(fifteen: list[Candle]) -> list[bool]:
    closes = [float(candle.close) for candle in fifteen]
    trend = ema(closes, 50)
    return [
        index >= 50 and closes[index] > trend[index] and trend[index] > trend[index - 1]
        for index in range(len(fifteen))
    ]


def _five_minute_signal_features(
    five: list[Candle], trend_ok: list[bool], five_trend_index: list[int | None]
) -> tuple[_FiveSignalFeature, ...]:
    closes = [float(candle.close) for candle in five]
    fast = ema(closes, 20)
    slow = ema(closes, 50)
    rolling_rsi = _rolling_rsi(closes, period=14)
    rolling_z = _rolling_zscore(closes, period=20)

    output: list[_FiveSignalFeature] = []
    for index, candle in enumerate(five):
        trend_index = five_trend_index[index]
        bullish = trend_index is not None and trend_ok[trend_index]
        output.append(
            _FiveSignalFeature(
                close_time=candle.close_time,
                trend_ok=bullish,
                close_below_fast=index >= 20 and closes[index] < fast[index],
                close_above_slow=index >= 50 and closes[index] > slow[index],
                rsi_value=rolling_rsi[index],
                z_score=rolling_z[index],
            )
        )
    return tuple(output)


def _breakout_signal_features(
    fifteen: list[Candle], trend_ok: list[bool]
) -> tuple[_BreakoutSignalFeature, ...]:
    lookbacks = (20, 40)
    output: list[_BreakoutSignalFeature] = []
    for index, candle in enumerate(fifteen):
        windows: dict[int, bool] = {}
        for lookback in lookbacks:
            if index < lookback:
                windows[lookback] = False
                continue
            previous_high = max(item.high for item in fifteen[index - lookback : index])
            windows[lookback] = candle.close > previous_high
        output.append(
            _BreakoutSignalFeature(
                close_time=candle.close_time,
                trend_ok=trend_ok[index],
                breakout_windows=windows,
            )
        )
    return tuple(output)


def _rolling_rsi(values: list[float], *, period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    for index in range(period, len(values)):
        output[index] = rsi(values[index - period : index + 1], period)
    return output


def _rolling_zscore(values: list[float], *, period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if period <= 1:
        raise ValueError("period must exceed 1")
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        mean = sum(window) / period
        variance = sum((value - mean) ** 2 for value in window) / period
        stddev = sqrt(variance)
        output[index] = 0.0 if stddev == 0 else (values[index] - mean) / stddev
    return output


def simulate_fixed_horizon(
    candles: list[Candle],
    *,
    candidate: PriceCandidate,
    signal_indexes: Iterable[int],
    start_index: int,
    end_index: int,
    initial_equity: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> SimulationResult:
    """Simulate one no-overlap long-only candidate over a bounded window.

    Signal is seen at a bar close. Entry is the next 1m OPEN. Exit is the
    fixed-horizon 1m CLOSE.  This intentional latency is the small, unglamorous
    difference between a usable backtest and a time machine.
    """

    ordered = sorted(candles, key=lambda candle: candle.open_time)
    if initial_equity <= 0 or quote_amount <= 0:
        raise ValueError("initial_equity and quote_amount must be positive")
    if fee_rate_pct < 0 or slippage_pct < 0:
        raise ValueError("fee_rate_pct and slippage_pct cannot be negative")
    if not 0 <= start_index < end_index <= len(ordered):
        raise ValueError("window indexes are outside candle range")

    fee_rate = fee_rate_pct / Decimal("100")
    slippage_rate = slippage_pct / Decimal("100")
    balance = initial_equity
    peak = initial_equity
    max_drawdown = Decimal("0")
    total_fees = Decimal("0")
    trades: list[SimulatedTrade] = []
    skipped_gap = 0
    skipped_end = 0
    skipped_overlap = 0
    last_exit_index = start_index - 1

    for signal_index in sorted(set(signal_indexes)):
        if signal_index < start_index or signal_index >= end_index:
            continue
        if signal_index <= last_exit_index:
            skipped_overlap += 1
            continue

        entry_index = signal_index + 1
        exit_index = signal_index + candidate.horizon_bars
        if entry_index >= end_index or exit_index >= end_index:
            skipped_end += 1
            continue
        if not _path_is_contiguous(ordered, signal_index, exit_index):
            skipped_gap += 1
            continue

        spend = min(quote_amount, balance / (Decimal("1") + fee_rate))
        if spend <= 0:
            break
        raw_entry = Decimal(str(ordered[entry_index].open))
        raw_exit = Decimal(str(ordered[exit_index].close))
        if raw_entry <= 0 or raw_exit <= 0:
            continue

        entry_price = raw_entry * (Decimal("1") + slippage_rate)
        entry_fee = spend * fee_rate
        quantity = spend / entry_price
        gross_exit = quantity * raw_exit * (Decimal("1") - slippage_rate)
        exit_fee = gross_exit * fee_rate
        net_exit = gross_exit - exit_fee
        pnl = net_exit - spend - entry_fee

        balance += pnl
        total_fees += entry_fee + exit_fee
        peak = max(peak, balance)
        max_drawdown = max(max_drawdown, peak - balance)
        trades.append(
            SimulatedTrade(
                entry_time=ordered[entry_index].open_time,
                exit_time=ordered[exit_index].close_time,
                entry_price=entry_price,
                exit_price=raw_exit * (Decimal("1") - slippage_rate),
                quote_amount=spend,
                quantity=quantity,
                entry_fee=entry_fee,
                exit_fee=exit_fee,
                pnl=pnl,
                candidate_name=candidate.name,
            )
        )
        last_exit_index = exit_index

    return SimulationResult(
        initial_equity=initial_equity,
        final_equity=balance,
        trades=tuple(trades),
        total_fees=total_fees,
        max_drawdown=max_drawdown,
        skipped_gap_signals=skipped_gap,
        skipped_end_signals=skipped_end,
        skipped_overlap_signals=skipped_overlap,
    )


def run_walk_forward(
    candles: list[Candle],
    *,
    candidates: Iterable[PriceCandidate],
    folds: Iterable[WalkForwardFold],
    initial_equity: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
    min_train_trades: int,
    min_validation_trades: int,
    min_selected_folds: int,
    min_profitable_folds: int,
) -> WalkForwardOutcome:
    if min(train_value for train_value in (min_train_trades, min_validation_trades, min_selected_folds, min_profitable_folds)) <= 0:
        raise ValueError("minimum thresholds must be positive")

    ordered = sorted(candles, key=lambda candle: candle.open_time)
    candidate_list = tuple(candidates)
    signals = build_signal_index_map(ordered, candidate_list)
    carried_equity = initial_equity
    all_validation_trades: list[SimulatedTrade] = []
    all_fees = Decimal("0")
    max_drawdown = Decimal("0")
    skipped_gap = 0
    skipped_end = 0
    skipped_overlap = 0
    selected_fold_count = 0
    profitable_folds = 0
    benchmark_compound = Decimal("1")
    outcomes: list[FoldOutcome] = []

    for fold in folds:
        evaluations: list[CandidateEvaluation] = []
        for candidate in candidate_list:
            train = simulate_fixed_horizon(
                ordered,
                candidate=candidate,
                signal_indexes=signals[candidate.name],
                start_index=fold.train_start,
                end_index=fold.train_end,
                initial_equity=initial_equity,
                quote_amount=quote_amount,
                fee_rate_pct=fee_rate_pct,
                slippage_pct=slippage_pct,
            )
            if _train_eligible(train, min_train_trades=min_train_trades):
                evaluations.append(CandidateEvaluation(candidate=candidate, train_result=train))

        benchmark = buy_and_hold_return_pct(
            ordered,
            start_index=fold.validation_start,
            end_index=fold.validation_end,
            fee_rate_pct=fee_rate_pct,
            slippage_pct=slippage_pct,
        )
        benchmark_compound *= Decimal("1") + benchmark / Decimal("100")

        if not evaluations:
            outcomes.append(
                FoldOutcome(
                    fold=fold,
                    selected=None,
                    selection_reason="no_train_candidate_met_minimum_profit_and_trade_requirements",
                    train_result=None,
                    validation_result=None,
                    buy_and_hold_return_pct=benchmark,
                )
            )
            continue

        chosen = max(evaluations, key=_candidate_rank_key)
        validation = simulate_fixed_horizon(
            ordered,
            candidate=chosen.candidate,
            signal_indexes=signals[chosen.candidate.name],
            start_index=fold.validation_start,
            end_index=fold.validation_end,
            initial_equity=carried_equity,
            quote_amount=quote_amount,
            fee_rate_pct=fee_rate_pct,
            slippage_pct=slippage_pct,
        )
        carried_equity = validation.final_equity
        selected_fold_count += 1
        if validation.net_pnl > 0:
            profitable_folds += 1
        all_validation_trades.extend(validation.trades)
        all_fees += validation.total_fees
        max_drawdown = max(max_drawdown, validation.max_drawdown)
        skipped_gap += validation.skipped_gap_signals
        skipped_end += validation.skipped_end_signals
        skipped_overlap += validation.skipped_overlap_signals
        outcomes.append(
            FoldOutcome(
                fold=fold,
                selected=chosen.candidate,
                selection_reason="highest_train_return_with_positive_profit_factor_and_minimum_trades",
                train_result=chosen.train_result,
                validation_result=validation,
                buy_and_hold_return_pct=benchmark,
            )
        )

    aggregate = SimulationResult(
        initial_equity=initial_equity,
        final_equity=carried_equity,
        trades=tuple(all_validation_trades),
        total_fees=all_fees,
        max_drawdown=max_drawdown,
        skipped_gap_signals=skipped_gap,
        skipped_end_signals=skipped_end,
        skipped_overlap_signals=skipped_overlap,
    )
    verdict = _verdict(
        aggregate,
        selected_fold_count=selected_fold_count,
        profitable_folds=profitable_folds,
        min_validation_trades=min_validation_trades,
        min_selected_folds=min_selected_folds,
        min_profitable_folds=min_profitable_folds,
        benchmark_return_pct=(benchmark_compound - Decimal("1")) * Decimal("100"),
    )
    return WalkForwardOutcome(
        folds=tuple(outcomes),
        validation_result=aggregate,
        selected_fold_count=selected_fold_count,
        profitable_validation_folds=profitable_folds,
        cumulative_buy_and_hold_return_pct=(benchmark_compound - Decimal("1")) * Decimal("100"),
        verdict=verdict,
    )


def buy_and_hold_return_pct(
    candles: list[Candle],
    *,
    start_index: int,
    end_index: int,
    fee_rate_pct: Decimal,
    slippage_pct: Decimal,
) -> Decimal:
    if end_index - start_index < 2:
        return Decimal("0")
    fee_rate = fee_rate_pct / Decimal("100")
    slippage_rate = slippage_pct / Decimal("100")
    entry = Decimal(str(candles[start_index].open)) * (Decimal("1") + slippage_rate)
    exit_price = Decimal(str(candles[end_index - 1].close)) * (Decimal("1") - slippage_rate)
    if entry <= 0:
        return Decimal("0")
    gross = exit_price / entry
    net = gross * (Decimal("1") - fee_rate) * (Decimal("1") - fee_rate)
    return (net - Decimal("1")) * Decimal("100")


def _train_eligible(result: SimulationResult, *, min_train_trades: int) -> bool:
    profit_factor = result.profit_factor
    return (
        result.round_trips >= min_train_trades
        and result.net_pnl > 0
        and profit_factor is not None
        and profit_factor > Decimal("1")
    )


def _candidate_rank_key(evaluation: CandidateEvaluation) -> tuple[Decimal, Decimal, Decimal, int]:
    result = evaluation.train_result
    profit_factor = result.profit_factor or Decimal("0")
    return (result.return_pct, profit_factor, -result.max_drawdown, result.round_trips)


def _verdict(
    result: SimulationResult,
    *,
    selected_fold_count: int,
    profitable_folds: int,
    min_validation_trades: int,
    min_selected_folds: int,
    min_profitable_folds: int,
    benchmark_return_pct: Decimal,
) -> str:
    if selected_fold_count < min_selected_folds:
        return "not_enough_train_qualified_folds"
    if result.round_trips < min_validation_trades:
        return "not_enough_validation_trades"
    if result.net_pnl <= 0:
        return "rejected_validation_negative_after_costs"
    profit_factor = result.profit_factor
    if profit_factor is None or profit_factor <= Decimal("1"):
        return "rejected_validation_profit_factor_not_above_1"
    if profitable_folds < min_profitable_folds:
        return "rejected_not_enough_profitable_validation_folds"
    largest_share = result.largest_winner_share
    if largest_share is not None and largest_share > Decimal("0.5"):
        return "rejected_single_trade_dominance"
    if result.return_pct <= benchmark_return_pct:
        return "rejected_not_better_than_buy_and_hold"
    return "promising_research_only"


def _path_is_contiguous(candles: list[Candle], start_index: int, end_index: int) -> bool:
    for index in range(start_index, end_index):
        delta = (candles[index + 1].open_time - candles[index].open_time).total_seconds()
        if abs(delta - 60.0) > 1.0:
            return False
    return True
