"""V28 research-only order-book gated price-strategy backtester.

V27.3 answered an important negative question: order-book imbalance by itself is
not a validated long-entry strategy after costs.  V28 tests a narrower and more
useful hypothesis: can observed order-book imbalance improve a conventional
price-based entry rule?

This module is deliberately research-only:
- no exchange execution
- no live signals
- no profitability promises
- train-only feature thresholds
- gap-safe entry/exit paths

A baseline and its gated counterpart are replayed on the same chronology.  The
only difference is the order-book gate, so the result answers whether the gate
adds value rather than merely finding a different strategy by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.backtesting.order_book_strategy import (
    feature_value,
    rows_with_feature,
)
from app.market.features import MarketFeatures
from app.market.indicators import ema, rsi
from app.utils.timeframe import timeframe_to_seconds


PRICE_STRATEGIES = ("ema_rsi_momentum", "breakout_momentum", "mean_reversion")


@dataclass(slots=True, frozen=True)
class PriceRuleConfig:
    """Fixed, intentionally small V28 price-rule family configuration."""

    strategy_kind: str
    fast_ema_period: int = 12
    slow_ema_period: int = 34
    trend_ema_period: int = 50
    rsi_period: int = 14
    rsi_buy_min: float = 45.0
    rsi_buy_max: float = 60.0
    breakout_lookback: int = 20
    mean_reversion_lookback: int = 20
    mean_reversion_entry_z: float = 2.0
    mean_reversion_rsi_max: float = 35.0

    def __post_init__(self) -> None:
        if self.strategy_kind not in PRICE_STRATEGIES:
            raise ValueError(f"Unknown V28 price strategy: {self.strategy_kind}")
        if self.fast_ema_period <= 0 or self.slow_ema_period <= self.fast_ema_period:
            raise ValueError("slow EMA period must be greater than fast EMA period")
        if self.trend_ema_period <= 1:
            raise ValueError("trend EMA period must be greater than 1")
        if self.rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if self.breakout_lookback <= 1:
            raise ValueError("breakout lookback must be greater than 1")
        if self.mean_reversion_lookback <= 1:
            raise ValueError("mean-reversion lookback must be greater than 1")
        if self.mean_reversion_entry_z <= 0:
            raise ValueError("mean-reversion entry z must be positive")


@dataclass(slots=True, frozen=True)
class OrderBookGateConfig:
    """Train-learned observed order-book gate applied to a price rule."""

    feature_name: str
    tail: str  # high or low
    threshold: float

    def __post_init__(self) -> None:
        if self.tail not in {"high", "low"}:
            raise ValueError("order-book gate tail must be 'high' or 'low'")


@dataclass(slots=True, frozen=True)
class GatedStrategyConfig:
    """One baseline-or-gated fixed-horizon V28 replay configuration."""

    price_rule: PriceRuleConfig
    horizon_bars: int
    timeframe: str
    strategy_name: str
    order_book_gate: OrderBookGateConfig | None = None

    def __post_init__(self) -> None:
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars must be positive")


@dataclass(slots=True, frozen=True)
class GatedBacktestDiagnostics:
    total_rows: int
    price_signal_candidates: int
    gate_passed_signals: int
    gate_rejected_signals: int
    skipped_gap_signals: int
    skipped_end_signals: int
    skipped_warmup_rows: int
    gap_count: int
    max_gap_seconds: float


@dataclass(slots=True, frozen=True)
class GatedBacktestOutcome:
    result: BacktestResult
    diagnostics: GatedBacktestDiagnostics


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


def gate_matches(row: MarketFeatures, gate: OrderBookGateConfig | None) -> bool:
    """Return whether the observed order-book value passes a gate.

    A baseline has no gate and therefore always passes.  Missing feature values
    fail a gated setup, not because absence means bearishness but because making
    up market depth is not a valid research method.
    """

    if gate is None:
        return True
    value = feature_value(row, gate.feature_name)
    if value is None:
        return False
    return value >= gate.threshold if gate.tail == "high" else value <= gate.threshold


def _contiguous_history(rows: list[MarketFeatures], *, index: int, timeframe: str) -> list[MarketFeatures]:
    """Return the contiguous candle run ending at ``index``."""

    start = index
    while start > 0 and is_contiguous(rows[start - 1], rows[start], timeframe=timeframe):
        start -= 1
    return rows[start : index + 1]


def _price_rule_minimum(rule: PriceRuleConfig) -> int:
    if rule.strategy_kind == "ema_rsi_momentum":
        return max(rule.slow_ema_period, rule.trend_ema_period, rule.rsi_period + 1) + 1
    if rule.strategy_kind == "breakout_momentum":
        return max(rule.breakout_lookback + 1, rule.trend_ema_period)
    if rule.strategy_kind == "mean_reversion":
        return max(rule.mean_reversion_lookback, rule.trend_ema_period, rule.rsi_period + 1)
    raise ValueError(f"Unknown V28 price strategy: {rule.strategy_kind}")


def price_rule_signal(rows: list[MarketFeatures], *, index: int, config: GatedStrategyConfig) -> bool | None:
    """Evaluate a price rule at candle close without reading future candles.

    Returns ``None`` when the current contiguous history is not long enough.
    This lets diagnostics distinguish a normal warm-up from a rejected signal.
    """

    history = _contiguous_history(rows, index=index, timeframe=config.timeframe)
    rule = config.price_rule
    if len(history) < _price_rule_minimum(rule):
        return None

    closes = [float(row.close_price) for row in history]
    current_close = closes[-1]

    if rule.strategy_kind == "ema_rsi_momentum":
        fast_series = ema(closes, rule.fast_ema_period)
        slow_series = ema(closes, rule.slow_ema_period)
        current_rsi = rsi(closes, rule.rsi_period)
        if current_rsi is None:
            return None
        crossed_above = fast_series[-2] <= slow_series[-2] and fast_series[-1] > slow_series[-1]
        in_bull_trend = current_close > ema(closes, rule.trend_ema_period)[-1]
        rsi_ok = rule.rsi_buy_min <= current_rsi <= rule.rsi_buy_max
        return crossed_above and in_bull_trend and rsi_ok

    if rule.strategy_kind == "breakout_momentum":
        previous_closes = closes[-(rule.breakout_lookback + 1) : -1]
        previous_high = max(previous_closes)
        in_bull_trend = current_close > ema(closes, rule.trend_ema_period)[-1]
        return current_close > previous_high and in_bull_trend

    if rule.strategy_kind == "mean_reversion":
        window = closes[-rule.mean_reversion_lookback :]
        mean = sum(window) / len(window)
        variance = sum((value - mean) ** 2 for value in window) / (len(window) - 1)
        standard_deviation = sqrt(variance)
        current_rsi = rsi(closes, rule.rsi_period)
        if standard_deviation <= 0 or current_rsi is None:
            return False
        z_score = (current_close - mean) / standard_deviation
        # Buy a statistically stretched dip only within a broader uptrend.
        in_bull_trend = current_close > ema(closes, rule.trend_ema_period)[-1]
        return z_score <= -rule.mean_reversion_entry_z and current_rsi <= rule.mean_reversion_rsi_max and in_bull_trend

    raise ValueError(f"Unknown V28 price strategy: {rule.strategy_kind}")


def _trade_path_reason(rows: list[MarketFeatures], *, signal_index: int, horizon_bars: int, timeframe: str) -> str | None:
    entry_index = signal_index + 1
    exit_index = entry_index + horizon_bars
    if exit_index >= len(rows):
        return "end"
    for current_index in range(signal_index, exit_index):
        if not is_contiguous(rows[current_index], rows[current_index + 1], timeframe=timeframe):
            return "gap"
    return None


def run_order_book_gated_backtest(
    *,
    rows: list[MarketFeatures],
    config: GatedStrategyConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> GatedBacktestOutcome:
    """Replay a price strategy with an optional observed order-book gate.

    Price signal at candle ``i`` schedules entry at ``i + 1``. A trade only
    executes when every timestamp through the planned fixed-horizon exit is
    contiguous. A gated setup must additionally have an observed feature value
    that passes the train-learned gate at candle ``i``.
    """

    if initial_quote_balance <= 0 or quote_amount <= 0:
        raise ValueError("initial_quote_balance and quote_amount must be positive")
    if fee_rate_pct < 0 or slippage_pct < 0:
        raise ValueError("fee_rate_pct and slippage_pct cannot be negative")

    ordered = sorted(rows, key=lambda row: row.close_time)
    if not ordered:
        return GatedBacktestOutcome(
            result=_empty_result(initial_quote_balance),
            diagnostics=GatedBacktestDiagnostics(0, 0, 0, 0, 0, 0, 0, 0.0),
        )

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    gap_count, max_gap_seconds = continuity_summary(ordered, timeframe=config.timeframe)

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

    price_signal_candidates = 0
    gate_passed_signals = 0
    gate_rejected_signals = 0
    skipped_gap_signals = 0
    skipped_end_signals = 0
    skipped_warmup_rows = 0

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
                    exit_reason=f"v28_fixed_horizon_{config.horizon_bars}_bars",
                )
            )
            position = None

        if position is None and scheduled_buy_index is None:
            signal = price_rule_signal(ordered, index=index, config=config)
            if signal is None:
                skipped_warmup_rows += 1
            elif signal:
                price_signal_candidates += 1
                if not gate_matches(row, config.order_book_gate):
                    gate_rejected_signals += 1
                else:
                    gate_passed_signals += 1
                    path_reason = _trade_path_reason(
                        ordered,
                        signal_index=index,
                        horizon_bars=config.horizon_bars,
                        timeframe=config.timeframe,
                    )
                    if path_reason == "gap":
                        skipped_gap_signals += 1
                    elif path_reason == "end":
                        skipped_end_signals += 1
                    else:
                        scheduled_buy_index = index + 1
                        scheduled_exit_index = index + 1 + config.horizon_bars
                        gate_description = "no_order_book_gate"
                        if config.order_book_gate is not None:
                            feature = config.order_book_gate.feature_name
                            value = feature_value(row, feature)
                            operator = ">=" if config.order_book_gate.tail == "high" else "<="
                            gate_description = f"{feature}={value:.6f} {operator} {config.order_book_gate.threshold:.6f}"
                        scheduled_reason = f"price_rule={config.price_rule.strategy_kind}; {gate_description}"

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
    diagnostics = GatedBacktestDiagnostics(
        total_rows=len(ordered),
        price_signal_candidates=price_signal_candidates,
        gate_passed_signals=gate_passed_signals,
        gate_rejected_signals=gate_rejected_signals,
        skipped_gap_signals=skipped_gap_signals,
        skipped_end_signals=skipped_end_signals,
        skipped_warmup_rows=skipped_warmup_rows,
        gap_count=gap_count,
        max_gap_seconds=max_gap_seconds,
    )
    return GatedBacktestOutcome(
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
