from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.market.indicators import atr, ema, rsi
from app.market.models import Candle


def make_candle(index: int, high: float, low: float, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=close,
        high=high,
        low=low,
        close=close,
        volume=1,
    )


def test_ema_returns_same_length():
    values = [1, 2, 3, 4, 5]
    result = ema(values, 3)
    assert len(result) == len(values)
    assert result[-1] > result[0]


def test_rsi_returns_value_after_period():
    values = [100, 101, 102, 101, 103, 104, 102, 105, 106, 107, 106, 108, 109, 110, 111]
    result = rsi(values, 14)
    assert result is not None
    assert 0 <= result <= 100


def test_atr_uses_true_range():
    candles = [
        make_candle(0, 10, 9, 9.5),
        make_candle(1, 12, 10, 11),
        make_candle(2, 13, 11, 12),
        make_candle(3, 12, 10, 11),
    ]

    result = atr(candles, 3)

    assert result == pytest.approx((2.5 + 2 + 2) / 3)
