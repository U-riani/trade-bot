"""V27.2 gap-safe order-book threshold strategy research backtester.

V27 compressed the timeline by filtering out rows where an order-book feature was
missing. That is unsafe for sparse forward-collected data: the next stored
feature may be minutes or hours later, yet the old backtest treated it as the
next candle. V27.2 keeps every stored market-feature row, only uses present
feature values to create signals, and skips any trade whose entry/exit path
crosses a timestamp gap.

Research only. No live signals, no execution, no profit claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import ceil

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.market.features import MarketFeatures
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class OrderBookThresholdConfig:
    """One fixed-horizon long research configuration."""

    feature_name: str
    entry_threshold: float
    horizon_bars: int
    strategy_name: str
    timeframe: str = "5m"
    entry_tail: str = "high"  # high => feature >= threshold; low => feature <= threshold


@dataclass(slots=True, frozen=True)
class BacktestDiagnostics:
    total_rows: int
    feature_observations: int
    gap_count: int
    max_gap_seconds: float
    signal_candidates: int
    skipped_gap_signals: int
    skipped_end_signals: int


@dataclass(slots=True, frozen=True)
class OrderBookBacktestOutcome:
    result: BacktestResult
    diagnostics: BacktestDiagnostics


@dataclass(slots=True, frozen=True)
class FeatureCoverageSplit:
    """Chronological split inside the real order-book observation period."""

    full_rows: list[MarketFeatures]
    train_rows: list[MarketFeatures]
    validation_rows: list[MarketFeatures]
    coverage_start: object
    split_time: object
    coverage_end: object


@dataclass(slots=True)
class _OpenPosition:
    exit_index: int
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


def feature_value(row: MarketFeatures, feature_name: str) -> float | None:
    if not hasattr(row, feature_name):
        raise ValueError(f"Unknown market feature: {feature_name}")
    value = getattr(row, feature_name)
    return None if value is None else float(value)


def rows_with_feature(rows: list[MarketFeatures], feature_name: str) -> list[MarketFeatures]:
    """Present observations, sorted, for threshold estimation only.

    Do not pass this filtered list into the gap-safe backtester. Missing feature
    values must remain in the price timeline so a one-bar horizon still means one
    actual candle, not one arbitrary later observation.
    """

    return sorted((row for row in rows if feature_value(row, feature_name) is not None), key=lambda row: row.close_time)


def quantile_threshold(rows: list[MarketFeatures], feature_name: str, quantile: float) -> float:
    if quantile <= 0 or quantile >= 1:
        raise ValueError("quantile must be between 0 and 1")
    values = sorted(value for row in rows if (value := feature_value(row, feature_name)) is not None)
    if not values:
        raise ValueError(f"No rows with feature: {feature_name}")
    index = max(0, min(len(values) - 1, ceil(len(values) * quantile) - 1))
    return values[index]


def split_feature_rows(rows: list[MarketFeatures], *, train_ratio: Decimal) -> tuple[list[MarketFeatures], list[MarketFeatures]]:
    """Chronological split preserving rows with missing feature values."""

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be greater than 0 and smaller than 1")
    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    split_index = int(len(sorted_rows) * float(train_ratio))
    if split_index <= 0 or split_index >= len(sorted_rows):
        raise ValueError("not enough rows to split train/validation sets")
    return sorted_rows[:split_index], sorted_rows[split_index:]


def split_rows_by_feature_coverage(
    rows: list[MarketFeatures],
    *,
    feature_name: str,
    train_ratio: Decimal,
) -> FeatureCoverageSplit:
    """Split chronologically inside actual feature coverage, not old candle history.

    The replay timeline retains every candle row from first observed feature through
    last observed feature. Threshold learning uses the earliest feature observations
    only, while gap-safe execution still rejects paths crossing missing candles.
    """

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be greater than 0 and smaller than 1")

    ordered = sorted(rows, key=lambda row: row.close_time)
    observed = rows_with_feature(ordered, feature_name)
    split_index = int(len(observed) * float(train_ratio))
    if split_index <= 0 or split_index >= len(observed):
        raise ValueError("not enough feature observations to split train/validation sets")

    coverage_start = observed[0].close_time
    split_time = observed[split_index].close_time
    coverage_end = observed[-1].close_time
    full_rows = [row for row in ordered if coverage_start <= row.close_time <= coverage_end]
    train_rows = [row for row in full_rows if row.close_time < split_time]
    validation_rows = [row for row in full_rows if row.close_time >= split_time]
    if not train_rows or not validation_rows:
        raise ValueError("feature coverage split produced an empty train or validation timeline")

    return FeatureCoverageSplit(
        full_rows=full_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        coverage_start=coverage_start,
        split_time=split_time,
        coverage_end=coverage_end,
    )


def _expected_seconds(timeframe: str) -> int:
    seconds = timeframe_to_seconds(timeframe)
    if seconds <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return seconds


def _gap_seconds(previous: MarketFeatures, current: MarketFeatures) -> float:
    return max(0.0, (current.close_time - previous.close_time).total_seconds())


def _is_contiguous(previous: MarketFeatures, current: MarketFeatures, *, timeframe: str) -> bool:
    # Exact exchange candle timestamps should differ by the timeframe. One second
    # tolerance protects against timestamp serialization quirks, not missing bars.
    return abs(_gap_seconds(previous, current) - _expected_seconds(timeframe)) <= 1.0


def continuity_diagnostics(rows: list[MarketFeatures], *, timeframe: str) -> tuple[int, float]:
    ordered = sorted(rows, key=lambda row: row.close_time)
    gaps = 0
    max_gap = 0.0
    for previous, current in zip(ordered, ordered[1:]):
        seconds = _gap_seconds(previous, current)
        max_gap = max(max_gap, seconds)
        if not _is_contiguous(previous, current, timeframe=timeframe):
            gaps += 1
    return gaps, max_gap


def _signal_matches(value: float, config: OrderBookThresholdConfig) -> bool:
    if config.entry_tail == "high":
        return value >= config.entry_threshold
    if config.entry_tail == "low":
        return value <= config.entry_threshold
    raise ValueError("entry_tail must be 'high' or 'low'")


def _can_complete_trade_path(rows: list[MarketFeatures], *, signal_index: int, horizon_bars: int, timeframe: str) -> str | None:
    """Return a skip reason when next-candle entry plus fixed exit is unavailable."""

    entry_index = signal_index + 1
    exit_index = entry_index + horizon_bars
    if exit_index >= len(rows):
        return "end"
    for index in range(signal_index, exit_index):
        if not _is_contiguous(rows[index], rows[index + 1], timeframe=timeframe):
            return "gap"
    return None


def run_order_book_threshold_backtest_with_diagnostics(
    *,
    rows: list[MarketFeatures],
    config: OrderBookThresholdConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> OrderBookBacktestOutcome:
    """Gap-safe fixed-horizon long threshold backtest.

    A feature at row i may schedule an entry at i+1 only when every timestamp
    through its planned exit is contiguous. Signals that would bridge a gap or
    run beyond the available sample are explicitly skipped and reported.
    """

    if config.horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if initial_quote_balance <= 0:
        raise ValueError("initial_quote_balance must be positive")
    if quote_amount <= 0:
        raise ValueError("quote_amount must be positive")
    if fee_rate_pct < 0 or slippage_pct < 0:
        raise ValueError("fee_rate_pct and slippage_pct cannot be negative")

    ordered = sorted(rows, key=lambda row: row.close_time)
    if not ordered:
        return OrderBookBacktestOutcome(
            result=_empty_result(initial_quote_balance=initial_quote_balance),
            diagnostics=BacktestDiagnostics(0, 0, 0, 0.0, 0, 0, 0),
        )

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    gap_count, max_gap_seconds = continuity_diagnostics(ordered, timeframe=config.timeframe)

    quote_balance = initial_quote_balance
    realized_pnl = Decimal("0")
    total_fees = Decimal("0")
    position: _OpenPosition | None = None
    trades: list[BacktestTrade] = []
    equity_curve: list[Decimal] = []
    equity_peak = initial_quote_balance
    max_drawdown = Decimal("0")
    scheduled_buy_index: int | None = None
    scheduled_exit_index: int | None = None
    scheduled_reason = ""
    executed_orders = 0
    signal_candidates = 0
    skipped_gap_signals = 0
    skipped_end_signals = 0

    for index, row in enumerate(ordered):
        close_price = Decimal(str(row.close_price))

        if position is None and scheduled_buy_index == index:
            spend = _spend_amount(quote_balance=quote_balance, requested=quote_amount, fee_rate=fee_rate)
            if spend > 0 and close_price > 0 and scheduled_exit_index is not None:
                entry_price = close_price * (Decimal("1") + slippage_rate)
                entry_fee = spend * fee_rate
                quantity = spend / entry_price
                quote_balance -= spend + entry_fee
                realized_pnl -= entry_fee
                total_fees += entry_fee
                executed_orders += 1
                position = _OpenPosition(
                    exit_index=scheduled_exit_index,
                    entry_time=row.close_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    quote_amount=spend,
                    entry_fee=entry_fee,
                    entry_reason=scheduled_reason,
                )
            scheduled_buy_index = None
            scheduled_exit_index = None
            scheduled_reason = ""

        if position is not None and index == position.exit_index:
            exit_price = close_price * (Decimal("1") - slippage_rate)
            gross_quote = position.quantity * exit_price
            exit_fee = gross_quote * fee_rate
            net_quote = gross_quote - exit_fee
            cost_basis = position.quantity * position.entry_price
            pnl = net_quote - cost_basis - position.entry_fee

            quote_balance += net_quote
            realized_pnl += gross_quote - cost_basis - exit_fee
            total_fees += exit_fee
            executed_orders += 1
            trades.append(
                BacktestTrade(
                    symbol=symbol,
                    entry_time=position.entry_time,  # type: ignore[arg-type]
                    exit_time=row.close_time,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    quantity=position.quantity,
                    quote_amount=position.quote_amount,
                    entry_fee=position.entry_fee,
                    exit_fee=exit_fee,
                    pnl=pnl,
                    entry_reason=position.entry_reason,
                    exit_reason=f"gap_safe_fixed_horizon_{config.horizon_bars}_bars",
                )
            )
            position = None

        if position is None and scheduled_buy_index is None:
            value = feature_value(row, config.feature_name)
            if value is not None and _signal_matches(value, config):
                signal_candidates += 1
                reason = _can_complete_trade_path(
                    ordered,
                    signal_index=index,
                    horizon_bars=config.horizon_bars,
                    timeframe=config.timeframe,
                )
                if reason == "gap":
                    skipped_gap_signals += 1
                elif reason == "end":
                    skipped_end_signals += 1
                else:
                    scheduled_buy_index = index + 1
                    scheduled_exit_index = index + 1 + config.horizon_bars
                    operator = ">=" if config.entry_tail == "high" else "<="
                    scheduled_reason = (
                        f"{config.feature_name}={value:.6f} {operator} threshold={config.entry_threshold:.6f}; "
                        f"tail={config.entry_tail}"
                    )

        current_equity = quote_balance + ((position.quantity * close_price) if position is not None else Decimal("0"))
        equity_curve.append(current_equity)
        if current_equity > equity_peak:
            equity_peak = current_equity
        max_drawdown = max(max_drawdown, equity_peak - current_equity)

    last_price = Decimal(str(ordered[-1].close_price))
    open_qty = position.quantity if position is not None else Decimal("0")
    open_avg = position.entry_price if position is not None else Decimal("0")
    unrealized_pnl = (last_price - open_avg) * open_qty if position is not None else Decimal("0")
    final_equity = quote_balance + (open_qty * last_price)

    metrics = BacktestMetrics(
        candles_processed=len(ordered),
        executed_orders=executed_orders,
        round_trips=len(trades),
        winning_trades=sum(1 for trade in trades if trade.pnl > 0),
        losing_trades=sum(1 for trade in trades if trade.pnl < 0),
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        total_fees=total_fees,
        max_drawdown=max_drawdown,
        initial_equity=initial_quote_balance,
        final_equity=final_equity,
        open_position_quantity=open_qty,
        open_position_avg_entry_price=open_avg,
        last_price=last_price,
    )
    diagnostics = BacktestDiagnostics(
        total_rows=len(ordered),
        feature_observations=len(rows_with_feature(ordered, config.feature_name)),
        gap_count=gap_count,
        max_gap_seconds=max_gap_seconds,
        signal_candidates=signal_candidates,
        skipped_gap_signals=skipped_gap_signals,
        skipped_end_signals=skipped_end_signals,
    )
    return OrderBookBacktestOutcome(result=BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve), diagnostics=diagnostics)


def run_order_book_threshold_backtest(**kwargs: object) -> BacktestResult:
    """Compatibility wrapper for callers that only need the result."""

    return run_order_book_threshold_backtest_with_diagnostics(**kwargs).result  # type: ignore[arg-type]


def buy_and_hold_feature_rows(
    *,
    rows: list[MarketFeatures],
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    ordered = sorted(rows, key=lambda row: row.close_time)
    if not ordered:
        return _empty_result(initial_quote_balance=initial_quote_balance)

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    spend = _spend_amount(quote_balance=initial_quote_balance, requested=quote_amount, fee_rate=fee_rate)
    first, last = ordered[0], ordered[-1]
    entry_price = Decimal(str(first.close_price)) * (Decimal("1") + slippage_rate)
    last_price = Decimal(str(last.close_price))
    entry_fee = spend * fee_rate
    quantity = spend / entry_price if entry_price > 0 else Decimal("0")
    quote_balance = initial_quote_balance - spend - entry_fee
    final_equity = quote_balance + (quantity * last_price)
    unrealized = (last_price - entry_price) * quantity
    equity_curve = [quote_balance + (quantity * Decimal(str(row.close_price))) for row in ordered]
    max_drawdown = _absolute_max_drawdown(equity_curve, initial_quote_balance)
    metrics = BacktestMetrics(
        candles_processed=len(ordered), executed_orders=1 if quantity > 0 else 0, round_trips=0,
        winning_trades=0, losing_trades=0, realized_pnl=-entry_fee, unrealized_pnl=unrealized,
        total_fees=entry_fee, max_drawdown=max_drawdown, initial_equity=initial_quote_balance,
        final_equity=final_equity, open_position_quantity=quantity,
        open_position_avg_entry_price=entry_price, last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=equity_curve)


def _spend_amount(*, quote_balance: Decimal, requested: Decimal, fee_rate: Decimal) -> Decimal:
    max_spend = quote_balance / (Decimal("1") + fee_rate) if fee_rate > 0 else quote_balance
    return min(requested, max_spend)


def _empty_result(*, initial_quote_balance: Decimal) -> BacktestResult:
    metrics = BacktestMetrics(
        candles_processed=0, executed_orders=0, round_trips=0, winning_trades=0, losing_trades=0,
        realized_pnl=Decimal("0"), unrealized_pnl=Decimal("0"), total_fees=Decimal("0"),
        max_drawdown=Decimal("0"), initial_equity=initial_quote_balance,
        final_equity=initial_quote_balance, open_position_quantity=Decimal("0"),
        open_position_avg_entry_price=Decimal("0"), last_price=None,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=[])


def _absolute_max_drawdown(equity_curve: list[Decimal], fallback_peak: Decimal) -> Decimal:
    if not equity_curve:
        return Decimal("0")
    peak = max(fallback_peak, equity_curve[0])
    max_drawdown = Decimal("0")
    for equity in equity_curve:
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    return max_drawdown
