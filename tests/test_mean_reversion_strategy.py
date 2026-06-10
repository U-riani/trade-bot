from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.mean_reversion import MeanReversionStrategy
from app.strategy.models import SignalSide


def make_candle(index: int, *, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1.0,
    )


def _state_with(closes: list[float]) -> MarketState:
    state = MarketState(symbol="BTCUSDT")
    for index, close in enumerate(closes):
        state.add_candle(make_candle(index, close=close))
    return state


def test_buys_on_stretched_dip_with_filters_off() -> None:
    strategy = MeanReversionStrategy(
        lookback=20,
        entry_z=Decimal("2.0"),
        rsi_period=None,
        rsi_buy_max=None,
        trend_ema_period=None,
        atr_period=None,
    )
    # 19 calm candles then a sharp drop => deep negative z-score.
    state = _state_with([100.0] * 19 + [97.0])

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.BUY
    assert "mean_reversion_buy" in signal.reason


def test_default_rsi_filter_allows_oversold_dip() -> None:
    # Defaults keep the RSI oversold filter on; a steep drop is oversold, so the
    # filter should let the buy through rather than block it.
    strategy = MeanReversionStrategy(lookback=20, entry_z=Decimal("2.0"))
    state = _state_with([100.0] * 19 + [97.0])

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.BUY


def test_sells_on_reversion_above_mean() -> None:
    strategy = MeanReversionStrategy(
        lookback=20,
        exit_z=Decimal("0.0"),
        rsi_period=None,
        rsi_buy_max=None,
        trend_ema_period=None,
        atr_period=None,
    )
    # Price pops above the mean => positive z-score => reversion exit.
    state = _state_with([100.0] * 19 + [101.0])

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.SELL
    assert "mean_reversion_exit" in signal.reason


def test_trend_filter_blocks_dip_below_trend() -> None:
    strategy = MeanReversionStrategy(
        lookback=20,
        entry_z=Decimal("2.0"),
        rsi_period=None,
        rsi_buy_max=None,
        trend_ema_period=10,
        atr_period=None,
    )
    # The dip is below the trend EMA, so "buy dips in an uptrend" must refuse it.
    state = _state_with([100.0] * 19 + [97.0])

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.HOLD
    assert "buy_filtered_trend" in signal.reason


def test_flat_window_holds() -> None:
    strategy = MeanReversionStrategy(lookback=20, rsi_period=None, rsi_buy_max=None, atr_period=None)
    state = _state_with([100.0] * 20)

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.HOLD
    assert "flat_window_no_volatility" in signal.reason


def test_not_enough_candles_holds() -> None:
    strategy = MeanReversionStrategy(lookback=20)
    state = _state_with([100.0] * 5)

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.HOLD
    assert "not_enough_candles" in signal.reason


def test_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError):
        MeanReversionStrategy(lookback=1)
    with pytest.raises(ValueError):
        MeanReversionStrategy(entry_z=Decimal("0"))
    with pytest.raises(ValueError):
        MeanReversionStrategy(rsi_buy_max=150.0)
