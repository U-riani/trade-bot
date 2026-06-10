from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.breakout_momentum import BreakoutMomentumStrategy
from app.strategy.models import SignalSide


def make_candle(index: int, *, close: float, high: float | None = None, low: float | None = None) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=close,
        high=high if high is not None else close,
        low=low if low is not None else close,
        close=close,
        volume=1.0,
    )


def test_breakout_strategy_buys_above_previous_high() -> None:
    state = MarketState(symbol="BTCUSDT")
    strategy = BreakoutMomentumStrategy(
        breakout_lookback=5,
        exit_lookback=3,
        trend_ema_period=None,
        atr_period=None,
        min_breakout_pct=Decimal("0"),
    )

    for index in range(6):
        state.add_candle(make_candle(index, close=100, high=100, low=99))
    state.add_candle(make_candle(6, close=101, high=101, low=100))

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.BUY
    assert "breakout_buy" in signal.reason


def test_breakout_strategy_sells_below_previous_low() -> None:
    state = MarketState(symbol="BTCUSDT")
    strategy = BreakoutMomentumStrategy(
        breakout_lookback=5,
        exit_lookback=3,
        trend_ema_period=None,
        atr_period=None,
    )

    for index in range(6):
        state.add_candle(make_candle(index, close=100, high=101, low=99))
    state.add_candle(make_candle(6, close=98, high=100, low=98))

    signal = strategy.on_market_state(state)

    assert signal.side == SignalSide.SELL
    assert "breakout_exit_low_break" in signal.reason
