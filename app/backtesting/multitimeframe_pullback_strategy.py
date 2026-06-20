"""V29 multi-timeframe pullback plus order-book-reversal research backtester.

Hypothesis, deliberately narrow and testable:

    15m uptrend -> 5m controlled pullback -> 1m order-book reversal -> long entry.

The matching baseline uses the same 15m/5m price setup but does not require the
1m order-book reversal.  Both variants enter on the following 1m candle and use
the same fixed horizon, fees, slippage and gap checks.

Research only.  No exchange execution, no live signals and no profit claim.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.backtesting.order_book_strategy import feature_value
from app.market.features import MarketFeatures
from app.market.indicators import ema, rsi
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class MultiTimeframePullbackConfig:
    """One V29 fixed-horizon replay configuration.

    ``require_order_book_reversal=False`` is the exact price-only baseline.  The
    gated version changes only that flag and the train-learned threshold.
    """

    feature_name: str
    reversal_threshold: float
    horizon_bars: int
    strategy_name: str
    require_order_book_reversal: bool = True
    entry_timeframe: str = "1m"
    pullback_timeframe: str = "5m"
    trend_timeframe: str = "15m"
    trend_ema_period: int = 50
    pullback_fast_ema_period: int = 9
    pullback_trend_ema_period: int = 50
    pullback_rsi_period: int = 14
    pullback_rsi_min: float = 30.0
    pullback_rsi_max: float = 50.0
    reversal_previous_max: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")
        if self.trend_ema_period <= 1:
            raise ValueError("trend_ema_period must be greater than 1")
        if self.pullback_fast_ema_period <= 1:
            raise ValueError("pullback_fast_ema_period must be greater than 1")
        if self.pullback_trend_ema_period <= self.pullback_fast_ema_period:
            raise ValueError("pullback_trend_ema_period must exceed pullback_fast_ema_period")
        if self.pullback_rsi_period <= 0:
            raise ValueError("pullback_rsi_period must be positive")
        if not 0 <= self.pullback_rsi_min <= self.pullback_rsi_max <= 100:
            raise ValueError("pullback RSI range must be between 0 and 100")


@dataclass(slots=True, frozen=True)
class PullbackSetup:
    """Price-only setup state for one 1m candle close.

    This cache is independent of threshold and holding horizon, allowing the
    runner to compare baseline and gated variants without recalculating EMA/RSI
    thousands of times.  It also ensures the two variants see exactly the same
    price setup chronology.
    """

    is_new_pullback_bar: bool
    price_setup: bool
    reason: str


@dataclass(slots=True, frozen=True)
class MultiTimeframeDiagnostics:
    total_entry_rows: int
    pullback_bar_checks: int
    missing_higher_timeframe_context: int
    trend_passed: int
    pullback_passed: int
    baseline_signal_candidates: int
    reversal_gate_passed: int
    reversal_gate_rejected: int
    reversal_gate_missing_feature: int
    skipped_gap_signals: int
    skipped_end_signals: int
    entry_gap_count: int
    max_entry_gap_seconds: float


@dataclass(slots=True, frozen=True)
class MultiTimeframeBacktestOutcome:
    result: BacktestResult
    diagnostics: MultiTimeframeDiagnostics


@dataclass(slots=True)
class _OpenPosition:
    exit_index: int
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


def expected_seconds(timeframe: str) -> int:
    seconds = timeframe_to_seconds(timeframe)
    if seconds <= 0:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return seconds


def is_contiguous(previous: MarketFeatures, current: MarketFeatures, *, timeframe: str) -> bool:
    delta = (current.close_time - previous.close_time).total_seconds()
    return abs(delta - expected_seconds(timeframe)) <= 1.0


def continuity_summary(rows: list[MarketFeatures], *, timeframe: str) -> tuple[int, float]:
    ordered = sorted(rows, key=lambda row: row.close_time)
    gaps = 0
    max_gap_seconds = 0.0
    for previous, current in zip(ordered, ordered[1:]):
        delta = max(0.0, (current.close_time - previous.close_time).total_seconds())
        max_gap_seconds = max(max_gap_seconds, delta)
        if not is_contiguous(previous, current, timeframe=timeframe):
            gaps += 1
    return gaps, max_gap_seconds


def _contiguous_history(rows: list[MarketFeatures], *, index: int, timeframe: str) -> list[MarketFeatures]:
    start = index
    while start > 0 and is_contiguous(rows[start - 1], rows[start], timeframe=timeframe):
        start -= 1
    return rows[start : index + 1]


def _asof_index(close_times: list[object], timestamp: object) -> int | None:
    # ``datetime`` is orderable.  Using bisect keeps higher-timeframe joins
    # explicitly backward-looking: never read a 5m/15m candle that has not closed.
    index = bisect_right(close_times, timestamp) - 1
    return index if index >= 0 else None


def _trend_is_bullish(rows: list[MarketFeatures], *, index: int, config: MultiTimeframePullbackConfig) -> bool | None:
    history = _contiguous_history(rows, index=index, timeframe=config.trend_timeframe)
    if len(history) < config.trend_ema_period + 1:
        return None
    closes = [float(row.close_price) for row in history]
    trend = ema(closes, config.trend_ema_period)
    return closes[-1] > trend[-1] and trend[-1] > trend[-2]


def _five_minute_pullback(rows: list[MarketFeatures], *, index: int, config: MultiTimeframePullbackConfig) -> bool | None:
    minimum = max(config.pullback_trend_ema_period, config.pullback_rsi_period + 1)
    history = _contiguous_history(rows, index=index, timeframe=config.pullback_timeframe)
    if len(history) < minimum:
        return None
    closes = [float(row.close_price) for row in history]
    current_rsi = rsi(closes, config.pullback_rsi_period)
    if current_rsi is None:
        return None
    fast = ema(closes, config.pullback_fast_ema_period)[-1]
    trend = ema(closes, config.pullback_trend_ema_period)[-1]
    current_close = closes[-1]
    # A dip, not a breakdown: below the short EMA, still above the slower 5m trend,
    # and neither completely flat nor already deeply oversold.
    return (
        current_close < fast
        and current_close > trend
        and config.pullback_rsi_min <= current_rsi <= config.pullback_rsi_max
    )


def build_pullback_setup_cache(
    *,
    entry_rows: list[MarketFeatures],
    pullback_rows: list[MarketFeatures],
    trend_rows: list[MarketFeatures],
    config: MultiTimeframePullbackConfig,
) -> list[PullbackSetup]:
    """Build price-only setup decisions using only completed 5m/15m candles."""

    entry = sorted(entry_rows, key=lambda row: row.close_time)
    pullback = sorted(pullback_rows, key=lambda row: row.close_time)
    trend = sorted(trend_rows, key=lambda row: row.close_time)
    pullback_times = [row.close_time for row in pullback]
    trend_times = [row.close_time for row in trend]

    previous_pullback_index: int | None = None
    setups: list[PullbackSetup] = []
    for row in entry:
        pullback_index = _asof_index(pullback_times, row.close_time)
        trend_index = _asof_index(trend_times, row.close_time)
        if pullback_index is None or trend_index is None:
            setups.append(PullbackSetup(False, False, "missing_higher_timeframe_candle"))
            continue

        # A 5m candle can be used only at its own close.  Without this condition,
        # the first 1m row of a segment could reuse a 5m bar that actually closed
        # before the segment started.  That would turn delayed availability into a
        # synthetic signal.
        if pullback[pullback_index].close_time != row.close_time:
            previous_pullback_index = pullback_index
            setups.append(PullbackSetup(False, False, "awaiting_5m_close"))
            continue

        # Evaluate only once per newly closed 5m candle.  Repeating the same pullback
        # setup on every one-minute row would manufacture extra trades.
        is_new_pullback_bar = pullback_index != previous_pullback_index
        previous_pullback_index = pullback_index
        if not is_new_pullback_bar:
            setups.append(PullbackSetup(False, False, "same_5m_bar"))
            continue

        bullish = _trend_is_bullish(trend, index=trend_index, config=config)
        pullback_signal = _five_minute_pullback(pullback, index=pullback_index, config=config)
        if bullish is None or pullback_signal is None:
            setups.append(PullbackSetup(True, False, "higher_timeframe_warmup_or_gap"))
        elif not bullish:
            setups.append(PullbackSetup(True, False, "15m_trend_not_bullish"))
        elif not pullback_signal:
            setups.append(PullbackSetup(True, False, "5m_pullback_not_present"))
        else:
            setups.append(PullbackSetup(True, True, "15m_trend_plus_5m_pullback"))
    return setups


def order_book_reversal_matches(
    rows: list[MarketFeatures], *, index: int, config: MultiTimeframePullbackConfig
) -> bool | None:
    """Check a 1m negative/neutral-to-positive order-book reversal.

    ``None`` means the reversal cannot be evaluated because the prior 1m candle or
    either feature value is unavailable.  It is not silently treated as bearish.
    """

    if index <= 0 or not is_contiguous(rows[index - 1], rows[index], timeframe=config.entry_timeframe):
        return None
    previous = feature_value(rows[index - 1], config.feature_name)
    current = feature_value(rows[index], config.feature_name)
    if previous is None or current is None:
        return None
    return previous <= config.reversal_previous_max and current >= config.reversal_threshold


def _trade_path_reason(rows: list[MarketFeatures], *, signal_index: int, horizon_bars: int, timeframe: str) -> str | None:
    entry_index = signal_index + 1
    exit_index = entry_index + horizon_bars
    if exit_index >= len(rows):
        return "end"
    for current_index in range(signal_index, exit_index):
        if not is_contiguous(rows[current_index], rows[current_index + 1], timeframe=timeframe):
            return "gap"
    return None


def run_multitimeframe_pullback_backtest(
    *,
    entry_rows: list[MarketFeatures],
    pullback_rows: list[MarketFeatures],
    trend_rows: list[MarketFeatures],
    config: MultiTimeframePullbackConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
    setup_cache: list[PullbackSetup] | None = None,
) -> MultiTimeframeBacktestOutcome:
    """Replay V29 baseline or 1m-reversal-gated entries on a gap-safe 1m timeline."""

    if initial_quote_balance <= 0 or quote_amount <= 0:
        raise ValueError("initial_quote_balance and quote_amount must be positive")
    if fee_rate_pct < 0 or slippage_pct < 0:
        raise ValueError("fee_rate_pct and slippage_pct cannot be negative")

    ordered = sorted(entry_rows, key=lambda row: row.close_time)
    if not ordered:
        return MultiTimeframeBacktestOutcome(
            result=_empty_result(initial_quote_balance),
            diagnostics=MultiTimeframeDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0),
        )

    setups = setup_cache or build_pullback_setup_cache(
        entry_rows=ordered,
        pullback_rows=pullback_rows,
        trend_rows=trend_rows,
        config=config,
    )
    if len(setups) != len(ordered):
        raise ValueError("setup_cache must align one-to-one with entry_rows")

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    gap_count, max_gap_seconds = continuity_summary(ordered, timeframe=config.entry_timeframe)

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

    pullback_bar_checks = 0
    missing_higher_timeframe_context = 0
    trend_passed = 0
    pullback_passed = 0
    baseline_signal_candidates = 0
    reversal_gate_passed = 0
    reversal_gate_rejected = 0
    reversal_gate_missing_feature = 0
    skipped_gap_signals = 0
    skipped_end_signals = 0

    for index, row in enumerate(ordered):
        close_price = Decimal(str(row.close_price))

        if position is None and scheduled_buy_index == index:
            spend = _spend_amount(quote_balance, quote_amount, fee_rate)
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
                    exit_reason=f"v29_fixed_horizon_{config.horizon_bars}_bars",
                )
            )
            position = None

        if position is None and scheduled_buy_index is None:
            setup = setups[index]
            if setup.is_new_pullback_bar:
                pullback_bar_checks += 1
            if setup.reason in {"missing_higher_timeframe_candle", "higher_timeframe_warmup_or_gap"}:
                missing_higher_timeframe_context += 1
            elif setup.reason == "15m_trend_plus_5m_pullback":
                trend_passed += 1
                pullback_passed += 1

            if setup.price_setup:
                baseline_signal_candidates += 1
                permitted = True
                if config.require_order_book_reversal:
                    reversal = order_book_reversal_matches(ordered, index=index, config=config)
                    if reversal is None:
                        reversal_gate_missing_feature += 1
                        permitted = False
                    elif reversal:
                        reversal_gate_passed += 1
                    else:
                        reversal_gate_rejected += 1
                        permitted = False

                if permitted:
                    path_reason = _trade_path_reason(
                        ordered,
                        signal_index=index,
                        horizon_bars=config.horizon_bars,
                        timeframe=config.entry_timeframe,
                    )
                    if path_reason == "gap":
                        skipped_gap_signals += 1
                    elif path_reason == "end":
                        skipped_end_signals += 1
                    else:
                        scheduled_buy_index = index + 1
                        scheduled_exit_index = index + 1 + config.horizon_bars
                        gate_note = "price_only_baseline"
                        if config.require_order_book_reversal:
                            previous = feature_value(ordered[index - 1], config.feature_name) if index > 0 else None
                            current = feature_value(row, config.feature_name)
                            gate_note = (
                                f"{config.feature_name}_reversal: previous={previous:.6f} <= {config.reversal_previous_max:.6f}; "
                                f"current={current:.6f} >= {config.reversal_threshold:.6f}"
                            )
                        scheduled_reason = f"15m_trend_plus_5m_pullback; {gate_note}"

        current_equity = quote_balance + ((position.quantity * close_price) if position is not None else Decimal("0"))
        equity_curve.append(current_equity)
        if current_equity > equity_peak:
            equity_peak = current_equity
        max_drawdown = max(max_drawdown, equity_peak - current_equity)

    last_price = Decimal(str(ordered[-1].close_price))
    open_quantity = position.quantity if position is not None else Decimal("0")
    open_average = position.entry_price if position is not None else Decimal("0")
    unrealized_pnl = (last_price - open_average) * open_quantity if position is not None else Decimal("0")
    final_equity = quote_balance + (open_quantity * last_price)

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
        open_position_quantity=open_quantity,
        open_position_avg_entry_price=open_average,
        last_price=last_price,
    )
    diagnostics = MultiTimeframeDiagnostics(
        total_entry_rows=len(ordered),
        pullback_bar_checks=pullback_bar_checks,
        missing_higher_timeframe_context=missing_higher_timeframe_context,
        trend_passed=trend_passed,
        pullback_passed=pullback_passed,
        baseline_signal_candidates=baseline_signal_candidates,
        reversal_gate_passed=reversal_gate_passed,
        reversal_gate_rejected=reversal_gate_rejected,
        reversal_gate_missing_feature=reversal_gate_missing_feature,
        skipped_gap_signals=skipped_gap_signals,
        skipped_end_signals=skipped_end_signals,
        entry_gap_count=gap_count,
        max_entry_gap_seconds=max_gap_seconds,
    )
    return MultiTimeframeBacktestOutcome(
        result=BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve),
        diagnostics=diagnostics,
    )


def _spend_amount(quote_balance: Decimal, requested: Decimal, fee_rate: Decimal) -> Decimal:
    maximum = quote_balance / (Decimal("1") + fee_rate) if fee_rate > 0 else quote_balance
    return min(requested, maximum)


def _empty_result(initial_quote_balance: Decimal) -> BacktestResult:
    metrics = BacktestMetrics(
        candles_processed=0,
        executed_orders=0,
        round_trips=0,
        winning_trades=0,
        losing_trades=0,
        realized_pnl=Decimal("0"),
        unrealized_pnl=Decimal("0"),
        total_fees=Decimal("0"),
        max_drawdown=Decimal("0"),
        initial_equity=initial_quote_balance,
        final_equity=initial_quote_balance,
        open_position_quantity=Decimal("0"),
        open_position_avg_entry_price=Decimal("0"),
        last_price=None,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=[])
