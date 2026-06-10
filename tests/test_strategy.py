from datetime import UTC, datetime, timedelta

from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.ema_rsi import EmaRsiStrategy
from app.strategy.models import SignalSide


def make_candle(index: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=10,
    )


def test_strategy_returns_hold_when_not_enough_data():
    state = MarketState(symbol="BTCUSDT")
    strategy = EmaRsiStrategy()
    state.add_candle(make_candle(0, 100))
    signal = strategy.on_market_state(state)
    assert signal.side == SignalSide.HOLD


def test_strategy_returns_valid_side_after_enough_data():
    state = MarketState(symbol="BTCUSDT")
    strategy = EmaRsiStrategy()
    prices = [100 + i * 0.1 for i in range(30)]
    for index, price in enumerate(prices):
        state.add_candle(make_candle(index, price))
    signal = strategy.on_market_state(state)
    assert signal.side in {SignalSide.BUY, SignalSide.SELL, SignalSide.HOLD}
