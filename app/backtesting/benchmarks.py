from __future__ import annotations

from decimal import Decimal

from app.backtesting.metrics import BacktestMetrics, BacktestResult
from app.market.models import Candle


def no_trade_benchmark(*, candles: list[Candle], initial_quote_balance: Decimal) -> BacktestResult:
    sorted_candles = sorted(candles, key=lambda item: item.open_time)
    last_price = Decimal(str(sorted_candles[-1].close)) if sorted_candles else None
    metrics = BacktestMetrics(
        candles_processed=len(sorted_candles),
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
        last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=[])


def buy_and_hold_order_sized_benchmark(
    *,
    candles: list[Candle],
    initial_quote_balance: Decimal,
    quote_amount: Decimal,
    fee_rate_pct: Decimal = Decimal("0"),
    slippage_pct: Decimal = Decimal("0"),
) -> BacktestResult:
    """Buy once with strategy-sized capital and hold until the final candle.

    This keeps the benchmark comparable with our bot, which currently risks a
    small fixed notional instead of going all-in like a caffeinated casino pigeon.
    """
    sorted_candles = sorted(candles, key=lambda item: item.open_time)
    if not sorted_candles:
        return no_trade_benchmark(candles=[], initial_quote_balance=initial_quote_balance)

    first_price = Decimal(str(sorted_candles[0].close))
    last_price = Decimal(str(sorted_candles[-1].close))
    quote_to_use = min(quote_amount, initial_quote_balance)
    fee_rate = fee_rate_pct / Decimal("100") if fee_rate_pct > 0 else Decimal("0")
    slippage_rate = slippage_pct / Decimal("100") if slippage_pct > 0 else Decimal("0")
    entry_price = first_price * (Decimal("1") + slippage_rate)
    entry_fee = quote_to_use * fee_rate

    if quote_to_use + entry_fee > initial_quote_balance:
        quote_to_use = initial_quote_balance / (Decimal("1") + fee_rate)
        entry_fee = quote_to_use * fee_rate

    quantity = quote_to_use / entry_price if entry_price > 0 else Decimal("0")
    quote_balance = initial_quote_balance - quote_to_use - entry_fee
    final_equity = quote_balance + (quantity * last_price)
    unrealized_pnl = (last_price - entry_price) * quantity

    equity_peak = initial_quote_balance
    max_drawdown = Decimal("0")
    for candle in sorted_candles:
        mark_price = Decimal(str(candle.close))
        current_equity = quote_balance + (quantity * mark_price)
        if current_equity > equity_peak:
            equity_peak = current_equity
        drawdown = equity_peak - current_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    metrics = BacktestMetrics(
        candles_processed=len(sorted_candles),
        executed_orders=1 if quantity > 0 else 0,
        round_trips=0,
        winning_trades=0,
        losing_trades=0,
        realized_pnl=-entry_fee,
        unrealized_pnl=unrealized_pnl,
        total_fees=entry_fee,
        max_drawdown=max_drawdown,
        initial_equity=initial_quote_balance,
        final_equity=final_equity,
        open_position_quantity=quantity,
        open_position_avg_entry_price=entry_price,
        last_price=last_price,
    )
    return BacktestResult(metrics=metrics, trades=[])


def split_walk_forward(candles: list[Candle], *, train_ratio: Decimal) -> tuple[list[Candle], list[Candle]]:
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be greater than 0 and smaller than 1")

    sorted_candles = sorted(candles, key=lambda item: item.open_time)
    split_index = int(len(sorted_candles) * float(train_ratio))
    if split_index <= 0 or split_index >= len(sorted_candles):
        raise ValueError("not enough candles to split train/validation sets")
    return sorted_candles[:split_index], sorted_candles[split_index:]
