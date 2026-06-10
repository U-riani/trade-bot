from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.backtesting.benchmarks import (
    buy_and_hold_order_sized_benchmark,
    no_trade_benchmark,
    split_walk_forward,
)
from app.market.models import Candle


def make_candle(index: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1,
    )


def test_no_trade_benchmark_keeps_initial_equity() -> None:
    result = no_trade_benchmark(
        candles=[make_candle(0, 100), make_candle(1, 110)],
        initial_quote_balance=Decimal("1000"),
    )

    assert result.metrics.final_equity == Decimal("1000")
    assert result.metrics.executed_orders == 0
    assert result.metrics.round_trips == 0


def test_buy_and_hold_order_sized_benchmark_marks_to_market() -> None:
    result = buy_and_hold_order_sized_benchmark(
        candles=[make_candle(0, 100), make_candle(1, 110)],
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("10"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )

    assert result.metrics.executed_orders == 1
    assert result.metrics.has_open_position
    assert result.metrics.final_equity == Decimal("1001.0")


def test_split_walk_forward_returns_train_and_validation() -> None:
    candles = [make_candle(index, 100 + index) for index in range(10)]

    train, validation = split_walk_forward(candles, train_ratio=Decimal("0.7"))

    assert len(train) == 7
    assert len(validation) == 3
    assert train[-1].close == 106
    assert validation[0].close == 107


def test_split_walk_forward_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError):
        split_walk_forward([make_candle(0, 100)], train_ratio=Decimal("1"))
