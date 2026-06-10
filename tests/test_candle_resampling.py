from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.backtesting.resample import resample_candles
from app.market.models import Candle


def _candle(index: int, *, minute_offset: int = 0) -> Candle:
    open_time = datetime(2026, 1, 1, 0, minute_offset + index, tzinfo=timezone.utc)
    close_time = open_time + timedelta(minutes=1) - timedelta(milliseconds=1)
    base = 100 + index
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=open_time,
        close_time=close_time,
        open=float(base),
        high=float(base + 2),
        low=float(base - 2),
        close=float(base + 1),
        volume=float(index + 1),
        is_closed=True,
    )


def test_resample_1m_to_5m_aggregates_ohlcv() -> None:
    candles = [_candle(index) for index in range(5)]

    result = resample_candles(candles, target_timeframe="5m", source_timeframe="1m")

    assert len(result) == 1
    candle = result[0]
    assert candle.timeframe == "5m"
    assert candle.open == candles[0].open
    assert candle.high == max(item.high for item in candles)
    assert candle.low == min(item.low for item in candles)
    assert candle.close == candles[-1].close
    assert candle.volume == sum(item.volume for item in candles)
    assert candle.open_time == candles[0].open_time
    assert candle.close_time == candles[-1].close_time


def test_resample_skips_partial_natural_buckets() -> None:
    candles = [_candle(index, minute_offset=2) for index in range(8)]

    result = resample_candles(candles, target_timeframe="5m", source_timeframe="1m")

    assert len(result) == 1
    assert result[0].open_time.minute == 5
    assert result[0].close_time.minute == 9


def test_resample_rejects_non_multiple_timeframe() -> None:
    with pytest.raises(ValueError, match="clean multiple"):
        resample_candles([_candle(0)], target_timeframe="7m", source_timeframe="5m")
