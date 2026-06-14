"""V27 research-only order-book threshold strategy backtester.

This module intentionally does not place orders, emit live signals, or claim an
edge. It turns the V26/V26.1 feature research into a small, brutally simple
question:

    If a live order-book imbalance feature is high at a candle close, does a
    fixed-horizon long trade beat no-trade / order-sized buy-and-hold after fees
    and slippage?

The design is deliberately conservative about time:
- Feature value at row i is considered known only after that candle closes.
- Entry happens on row i+1, never on the same row that produced the signal.
- Exit happens after a fixed number of future bars.

So yes, we do the boring no-lookahead thing, because reality already charges
fees and does not need help cheating.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import ceil

from app.backtesting.metrics import BacktestMetrics, BacktestResult, BacktestTrade
from app.market.features import MarketFeatures


@dataclass(slots=True, frozen=True)
class OrderBookThresholdConfig:
    """Config for one research-only long threshold strategy."""

    feature_name: str
    entry_threshold: float
    horizon_bars: int
    strategy_name: str


@dataclass(slots=True)
class _OpenPosition:
    entry_index: int
    exit_index: int
    entry_time: object
    entry_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    entry_reason: str


def feature_value(row: MarketFeatures, feature_name: str) -> float | None:
    """Return a numeric feature value from a MarketFeatures row.

    Unknown feature names raise early instead of quietly returning None and making
    a strategy look like it responsibly decided not to trade. Computers love that
    kind of passive-aggressive failure.
    """

    if not hasattr(row, feature_name):
        raise ValueError(f"Unknown market feature: {feature_name}")
    value = getattr(row, feature_name)
    if value is None:
        return None
    return float(value)


def rows_with_feature(rows: list[MarketFeatures], feature_name: str) -> list[MarketFeatures]:
    """Rows sorted by close_time where the requested feature is present."""

    return sorted((row for row in rows if feature_value(row, feature_name) is not None), key=lambda row: row.close_time)


def quantile_threshold(rows: list[MarketFeatures], feature_name: str, quantile: float) -> float:
    """Nearest-rank quantile threshold for a feature.

    The threshold is intended to be learned from train rows and reused for
    validation/full runs. That prevents the classic "I optimized on the future"
    backtest circus.
    """

    if quantile <= 0 or quantile >= 1:
        raise ValueError("quantile must be between 0 and 1")
    values = sorted(value for row in rows if (value := feature_value(row, feature_name)) is not None)
    if not values:
        raise ValueError(f"No rows with feature: {feature_name}")
    index = max(0, min(len(values) - 1, ceil(len(values) * quantile) - 1))
    return values[index]


def split_feature_rows(rows: list[MarketFeatures], *, train_ratio: Decimal) -> tuple[list[MarketFeatures], list[MarketFeatures]]:
    """Chronological train/validation split for feature rows."""

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be greater than 0 and smaller than 1")
    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    split_index = int(len(sorted_rows) * float(train_ratio))
    if split_index <= 0 or split_index >= len(sorted_rows):
        raise ValueError("not enough rows to split train/validation sets")
    return sorted_rows[:split_index], sorted_rows[split_index:]


def run_order_book_threshold_backtest(
    *,
    rows: list[MarketFeatures],
    config: OrderBookThresholdConfig,
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    """Run one fixed-horizon long threshold strategy on feature rows.

    Feature row i triggers a BUY scheduled for row i+1. This is important: the
    feature is known only after row i closes, so buying at the same row's close is
    lookahead-ish optimism with nicer shoes.
    """

    if config.horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")
    if initial_quote_balance <= 0:
        raise ValueError("initial_quote_balance must be positive")
    if quote_amount <= 0:
        raise ValueError("quote_amount must be positive")
    if fee_rate_pct < 0:
        raise ValueError("fee_rate_pct cannot be negative")
    if slippage_pct < 0:
        raise ValueError("slippage_pct cannot be negative")

    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    if not sorted_rows:
        return _empty_result(initial_quote_balance=initial_quote_balance)

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")

    quote_balance = initial_quote_balance
    realized_pnl = Decimal("0")
    total_fees = Decimal("0")
    position: _OpenPosition | None = None
    trades: list[BacktestTrade] = []
    equity_curve: list[Decimal] = []
    equity_peak = initial_quote_balance
    max_drawdown = Decimal("0")
    scheduled_buy_index: int | None = None
    scheduled_reason = ""
    executed_orders = 0

    for index, row in enumerate(sorted_rows):
        close_price = Decimal(str(row.close_price))

        if position is None and scheduled_buy_index == index:
            spend = _spend_amount(quote_balance=quote_balance, requested=quote_amount, fee_rate=fee_rate)
            if spend > 0 and close_price > 0:
                entry_price = close_price * (Decimal("1") + slippage_rate)
                entry_fee = spend * fee_rate
                quantity = spend / entry_price
                quote_balance -= spend + entry_fee
                realized_pnl -= entry_fee
                total_fees += entry_fee
                executed_orders += 1
                position = _OpenPosition(
                    entry_index=index,
                    exit_index=min(index + config.horizon_bars, len(sorted_rows) - 1),
                    entry_time=row.close_time,
                    entry_price=entry_price,
                    quantity=quantity,
                    quote_amount=spend,
                    entry_fee=entry_fee,
                    entry_reason=scheduled_reason or f"{config.feature_name}>={config.entry_threshold}",
                )
            scheduled_buy_index = None
            scheduled_reason = ""

        if position is not None and index >= position.exit_index and index > position.entry_index:
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
                    exit_reason=f"fixed_horizon_{config.horizon_bars}_bars",
                )
            )
            position = None

        if position is None and scheduled_buy_index is None and index + 1 < len(sorted_rows):
            value = feature_value(row, config.feature_name)
            if value is not None and value >= config.entry_threshold:
                scheduled_buy_index = index + 1
                scheduled_reason = f"{config.feature_name}={value:.6f} >= threshold={config.entry_threshold:.6f}"

        current_equity = quote_balance + ((position.quantity * close_price) if position is not None else Decimal("0"))
        equity_curve.append(current_equity)
        if current_equity > equity_peak:
            equity_peak = current_equity
        drawdown = equity_peak - current_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    last_price = Decimal(str(sorted_rows[-1].close_price))
    open_qty = position.quantity if position is not None else Decimal("0")
    open_avg = position.entry_price if position is not None else Decimal("0")
    unrealized_pnl = (last_price - open_avg) * open_qty if position is not None else Decimal("0")
    final_equity = quote_balance + (open_qty * last_price)

    metrics = BacktestMetrics(
        candles_processed=len(sorted_rows),
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
    return BacktestResult(metrics=metrics, trades=trades, equity_curve=equity_curve)


def buy_and_hold_feature_rows(
    *,
    rows: list[MarketFeatures],
    symbol: str,
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    """Order-sized buy-and-hold benchmark over the same feature-row window."""

    sorted_rows = sorted(rows, key=lambda row: row.close_time)
    if not sorted_rows:
        return _empty_result(initial_quote_balance=initial_quote_balance)

    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    spend = _spend_amount(quote_balance=initial_quote_balance, requested=quote_amount, fee_rate=fee_rate)
    first = sorted_rows[0]
    last = sorted_rows[-1]
    entry_price = Decimal(str(first.close_price)) * (Decimal("1") + slippage_rate)
    last_price = Decimal(str(last.close_price))
    entry_fee = spend * fee_rate
    quantity = spend / entry_price if entry_price > 0 else Decimal("0")
    quote_balance = initial_quote_balance - spend - entry_fee
    final_equity = quote_balance + (quantity * last_price)
    unrealized = (last_price - entry_price) * quantity
    equity_curve = [quote_balance + (quantity * Decimal(str(row.close_price))) for row in sorted_rows]
    max_drawdown = _absolute_max_drawdown(equity_curve, initial_quote_balance)

    metrics = BacktestMetrics(
        candles_processed=len(sorted_rows),
        executed_orders=1 if quantity > 0 else 0,
        round_trips=0,
        winning_trades=0,
        losing_trades=0,
        realized_pnl=-entry_fee,
        unrealized_pnl=unrealized,
        total_fees=entry_fee,
        max_drawdown=max_drawdown,
        initial_equity=initial_quote_balance,
        final_equity=final_equity,
        open_position_quantity=quantity,
        open_position_avg_entry_price=entry_price,
        last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=[], equity_curve=equity_curve)


def _spend_amount(*, quote_balance: Decimal, requested: Decimal, fee_rate: Decimal) -> Decimal:
    max_spend = quote_balance / (Decimal("1") + fee_rate) if fee_rate > 0 else quote_balance
    return min(requested, max_spend)


def _empty_result(*, initial_quote_balance: Decimal) -> BacktestResult:
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


def _absolute_max_drawdown(equity_curve: list[Decimal], fallback_peak: Decimal) -> Decimal:
    if not equity_curve:
        return Decimal("0")
    peak = max(fallback_peak, equity_curve[0])
    max_drawdown = Decimal("0")
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown
