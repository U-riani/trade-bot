"""V30 multi-timeframe pullback plus order-book-improvement research backtester.

Hypothesis, deliberately narrow and testable:

    15m uptrend -> 5m controlled pullback -> 1m order-book improvement -> long entry.

V29 required a very specific zero-to-high-positive imbalance jump.  V30 instead
requires a train-learned positive *delta* in observed order-book imbalance while
current imbalance is not negative.  The matching baseline uses the exact same
15m/5m price setup without the order-book condition.

Research only.  No exchange execution, no live signals and no profitability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.backtesting.multitimeframe_pullback_strategy import (
    MultiTimeframePullbackConfig,
    PullbackSetup,
    build_pullback_setup_cache,
    continuity_summary,
    is_contiguous,
)
from app.backtesting.order_book_strategy import feature_value
from app.market.features import MarketFeatures


@dataclass(slots=True, frozen=True)
class MultiTimeframeDeltaConfig:
    """One V30 fixed-horizon replay configuration.

    ``require_order_book_delta=False`` is the exact price-only baseline. The
    gated version changes only the delta gate, keeping the same trend, pullback,
    timeline, costs and exit horizon.
    """

    feature_name: str
    delta_threshold: float
    horizon_bars: int
    strategy_name: str
    require_order_book_delta: bool = True
    min_current_imbalance: float = 0.0
    entry_timeframe: str = "1m"
    pullback_timeframe: str = "5m"
    trend_timeframe: str = "15m"
    trend_ema_period: int = 50
    pullback_fast_ema_period: int = 9
    pullback_trend_ema_period: int = 50
    pullback_rsi_period: int = 14
    pullback_rsi_min: float = 30.0
    pullback_rsi_max: float = 50.0

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
class DeltaBacktestDiagnostics:
    total_entry_rows: int
    pullback_bar_checks: int
    missing_higher_timeframe_context: int
    trend_passed: int
    pullback_passed: int
    price_only_setup_candidates: int
    usable_order_book_delta_at_setup: int
    delta_gate_passed: int
    delta_gate_rejected: int
    delta_gate_missing_feature: int
    skipped_gap_signals: int
    skipped_end_signals: int
    entry_gap_count: int
    max_entry_gap_seconds: float


@dataclass(slots=True, frozen=True)
class DeltaBacktestOutcome:
    result: BacktestResult
    diagnostics: DeltaBacktestDiagnostics


@dataclass(slots=True)
class _OpenPosition:
    exit_index: int
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


def as_price_setup_config(config: MultiTimeframeDeltaConfig) -> MultiTimeframePullbackConfig:
    """Convert V30 config into the price-only V29 setup definition.

    The V29 cache is intentionally reused because V30 changes only the 1m gate.
    """

    return MultiTimeframePullbackConfig(
        feature_name=config.feature_name,
        reversal_threshold=0.0,
        horizon_bars=config.horizon_bars,
        strategy_name=f"{config.strategy_name}_price_setup",
        require_order_book_reversal=False,
        entry_timeframe=config.entry_timeframe,
        pullback_timeframe=config.pullback_timeframe,
        trend_timeframe=config.trend_timeframe,
        trend_ema_period=config.trend_ema_period,
        pullback_fast_ema_period=config.pullback_fast_ema_period,
        pullback_trend_ema_period=config.pullback_trend_ema_period,
        pullback_rsi_period=config.pullback_rsi_period,
        pullback_rsi_min=config.pullback_rsi_min,
        pullback_rsi_max=config.pullback_rsi_max,
    )


def observed_feature_delta(
    rows: list[MarketFeatures],
    *,
    index: int,
    feature_name: str,
    timeframe: str = "1m",
) -> tuple[float, float, float] | None:
    """Return ``(previous, current, delta)`` for one contiguous observed pair.

    A missing feature value means unknown, not zero.  A timestamp gap means the
    pair is unusable, because its change is not a one-bar order-book movement.
    """

    if index <= 0 or not is_contiguous(rows[index - 1], rows[index], timeframe=timeframe):
        return None
    previous = feature_value(rows[index - 1], feature_name)
    current = feature_value(rows[index], feature_name)
    if previous is None or current is None:
        return None
    return previous, current, current - previous


def positive_feature_deltas(
    rows: list[MarketFeatures],
    *,
    feature_name: str,
    timeframe: str = "1m",
) -> list[float]:
    """Return strictly positive one-bar order-book improvements only.

    Thresholds are learned from these values in the training segment.  This keeps
    V30 focused on improvement rather than allowing a negative change to pass a
    low percentile merely because the recent book was generally deteriorating.
    """

    ordered = sorted(rows, key=lambda row: row.close_time)
    values: list[float] = []
    for index in range(1, len(ordered)):
        pair = observed_feature_delta(
            ordered,
            index=index,
            feature_name=feature_name,
            timeframe=timeframe,
        )
        if pair is None:
            continue
        _previous, _current, delta = pair
        if delta > 0:
            values.append(delta)
    return values


def order_book_delta_matches(
    rows: list[MarketFeatures],
    *,
    index: int,
    config: MultiTimeframeDeltaConfig,
) -> tuple[bool, tuple[float, float, float] | None]:
    """Evaluate the V30 gate and return its observed pair for diagnostics."""

    pair = observed_feature_delta(
        rows,
        index=index,
        feature_name=config.feature_name,
        timeframe=config.entry_timeframe,
    )
    if pair is None:
        return False, None
    _previous, current, delta = pair
    return current >= config.min_current_imbalance and delta >= config.delta_threshold, pair


def _trade_path_reason(
    rows: list[MarketFeatures],
    *,
    signal_index: int,
    horizon_bars: int,
    timeframe: str,
) -> str | None:
    entry_index = signal_index + 1
    exit_index = entry_index + horizon_bars
    if exit_index >= len(rows):
        return "end"
    for current_index in range(signal_index, exit_index):
        if not is_contiguous(rows[current_index], rows[current_index + 1], timeframe=timeframe):
            return "gap"
    return None


def run_multitimeframe_pullback_delta_backtest(
    *,
    entry_rows: list[MarketFeatures],
    pullback_rows: list[MarketFeatures],
    trend_rows: list[MarketFeatures],
    config: MultiTimeframeDeltaConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
    setup_cache: list[PullbackSetup] | None = None,
) -> DeltaBacktestOutcome:
    """Replay V30 baseline or order-book-delta-gated entries on a gap-safe 1m timeline."""

    if initial_quote_balance <= 0 or quote_amount <= 0:
        raise ValueError("initial_quote_balance and quote_amount must be positive")
    if fee_rate_pct < 0 or slippage_pct < 0:
        raise ValueError("fee_rate_pct and slippage_pct cannot be negative")

    ordered = sorted(entry_rows, key=lambda row: row.close_time)
    if not ordered:
        return DeltaBacktestOutcome(
            result=_empty_result(initial_quote_balance),
            diagnostics=DeltaBacktestDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0),
        )

    setups = setup_cache or build_pullback_setup_cache(
        entry_rows=ordered,
        pullback_rows=pullback_rows,
        trend_rows=trend_rows,
        config=as_price_setup_config(config),
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
    price_only_setup_candidates = 0
    usable_order_book_delta_at_setup = 0
    delta_gate_passed = 0
    delta_gate_rejected = 0
    delta_gate_missing_feature = 0
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
                    exit_reason=f"v30_fixed_horizon_{config.horizon_bars}_bars",
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
                price_only_setup_candidates += 1
                permitted = True
                gate_note = "price_only_baseline"
                if config.require_order_book_delta:
                    matched, pair = order_book_delta_matches(ordered, index=index, config=config)
                    if pair is None:
                        delta_gate_missing_feature += 1
                        permitted = False
                    else:
                        usable_order_book_delta_at_setup += 1
                        previous, current, delta = pair
                        if matched:
                            delta_gate_passed += 1
                            gate_note = (
                                f"{config.feature_name}_delta: previous={previous:.6f}; current={current:.6f}; "
                                f"delta={delta:.6f} >= threshold={config.delta_threshold:.6f}; "
                                f"current >= {config.min_current_imbalance:.6f}"
                            )
                        else:
                            delta_gate_rejected += 1
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
    diagnostics = DeltaBacktestDiagnostics(
        total_entry_rows=len(ordered),
        pullback_bar_checks=pullback_bar_checks,
        missing_higher_timeframe_context=missing_higher_timeframe_context,
        trend_passed=trend_passed,
        pullback_passed=pullback_passed,
        price_only_setup_candidates=price_only_setup_candidates,
        usable_order_book_delta_at_setup=usable_order_book_delta_at_setup,
        delta_gate_passed=delta_gate_passed,
        delta_gate_rejected=delta_gate_rejected,
        delta_gate_missing_feature=delta_gate_missing_feature,
        skipped_gap_signals=skipped_gap_signals,
        skipped_end_signals=skipped_end_signals,
        entry_gap_count=gap_count,
        max_entry_gap_seconds=max_gap_seconds,
    )
    return DeltaBacktestOutcome(
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
