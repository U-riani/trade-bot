from __future__ import annotations

from datetime import timedelta

from app.market.bootstrap import validate_startup_candles
from app.market.models import Candle
from app.market.state import MarketState
from app.utils.time import utc_now
from app.utils.timeframe import timeframe_to_seconds


def make_candle(index: int, close: float, minutes_ago: int = 0) -> Candle:
    close_time = utc_now() - timedelta(minutes=minutes_ago) + timedelta(minutes=index)
    open_time = close_time - timedelta(minutes=1) + timedelta(milliseconds=1)
    return Candle(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=open_time,
        close_time=close_time,
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        volume=1.0,
    )


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("5m") == 300
    assert timeframe_to_seconds("1h") == 3600
    assert timeframe_to_seconds("1d") == 86400


def test_startup_candles_rejected_when_empty():
    result = validate_startup_candles(
        [],
        timeframe="1m",
        max_age_seconds=180,
        gap_tolerance_seconds=2,
    )
    assert not result.can_use
    assert result.reason == "no_candles_found"


def test_startup_candles_rejected_when_stale():
    candles = [make_candle(index=0, close=100, minutes_ago=60)]
    result = validate_startup_candles(
        candles,
        timeframe="1m",
        max_age_seconds=180,
        gap_tolerance_seconds=2,
    )
    assert not result.can_use
    assert result.reason.startswith("stale_candles")


def test_startup_candles_rejected_when_gap_exists():
    first = make_candle(index=0, close=100, minutes_ago=2)
    second = make_candle(index=3, close=103, minutes_ago=2)
    result = validate_startup_candles(
        [first, second],
        timeframe="1m",
        max_age_seconds=300,
        gap_tolerance_seconds=2,
    )
    assert not result.can_use
    assert result.reason.startswith("candle_gap_detected")


def test_startup_candles_accepted_when_fresh_and_continuous():
    candles = [make_candle(index=index, close=100 + index, minutes_ago=2) for index in range(3)]
    result = validate_startup_candles(
        candles,
        timeframe="1m",
        max_age_seconds=300,
        gap_tolerance_seconds=2,
    )
    assert result.can_use
    assert result.reason == "candles_fresh_and_continuous"


def test_market_state_loads_historical_candles_and_skips_duplicate_live_candle():
    candles = [make_candle(index=index, close=100 + index, minutes_ago=2) for index in range(3)]
    state = MarketState(symbol="BTCUSDT")

    loaded_count = state.load_historical_candles(candles)

    assert loaded_count == 3
    assert state.latest_price == candles[-1].close
    assert len(state.candles) == 3

    duplicate_added = state.add_candle(candles[-1])

    assert duplicate_added is False
    assert len(state.candles) == 3
