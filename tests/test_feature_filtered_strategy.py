from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.market.features import MarketFeatures
from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.feature_filtered import FeatureFilteredStrategy, FeatureUnavailableError
from app.strategy.models import SignalSide, TradeSignal

START = datetime(2026, 1, 1, tzinfo=UTC)


class StubStrategy(Strategy):
    name = "stub"

    def __init__(self, side: SignalSide) -> None:
        self._side = side

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        if self._side == SignalSide.HOLD:
            return TradeSignal.hold(strategy_name=self.name, symbol=market_state.symbol, reason="stub_hold")
        return TradeSignal(
            strategy_name=self.name,
            symbol=market_state.symbol,
            side=self._side,
            confidence=0.7,
            reason="stub_signal",
            created_at=START,
        )


def make_candle(index: int, volume: float) -> Candle:
    start = START + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=volume,
    )


def state_with_volumes(volumes: list[float]) -> MarketState:
    state = MarketState(symbol="BTCUSDT")
    for index, volume in enumerate(volumes):
        state.add_candle(make_candle(index, volume))
    return state


def latest_close_time(state: MarketState) -> datetime:
    return state.candles[-1].close_time


def feature_row(close_time: datetime, **kwargs) -> MarketFeatures:
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=close_time - timedelta(minutes=1),
        close_time=close_time,
        close_price=100.0,
        volume=10.0,
        **kwargs,
    )


# --- volume spike (always available historically) ---

def test_volume_spike_passes_buy() -> None:
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        volume_spike_lookback=5,
        min_volume_spike_ratio=2.0,
    )
    state = state_with_volumes([10.0] * 5 + [30.0])  # ratio 3.0
    signal = strat.on_market_state(state)
    assert signal.side == SignalSide.BUY
    assert "feature_filters_passed" in signal.reason


def test_volume_spike_blocks_buy() -> None:
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        volume_spike_lookback=5,
        min_volume_spike_ratio=2.0,
    )
    state = state_with_volumes([10.0] * 5 + [10.0])  # ratio 1.0
    signal = strat.on_market_state(state)
    assert signal.side == SignalSide.HOLD
    assert "volume_spike_too_low" in signal.reason


def test_volume_spike_insufficient_history_holds() -> None:
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        volume_spike_lookback=5,
        min_volume_spike_ratio=2.0,
    )
    state = state_with_volumes([10.0, 10.0, 10.0])
    signal = strat.on_market_state(state)
    assert signal.side == SignalSide.HOLD
    assert "insufficient_history" in signal.reason


# --- taker filter (historically available) ---

def test_taker_filter_blocks_when_below_threshold() -> None:
    state = state_with_volumes([10.0] * 6)
    features = {latest_close_time(state): feature_row(latest_close_time(state), taker_buy_ratio=0.3)}
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        min_taker_buy_ratio=0.5,
        features_by_close_time=features,
    )
    signal = strat.on_market_state(state)
    assert signal.side == SignalSide.HOLD
    assert "taker_buy_ratio_too_low" in signal.reason


def test_taker_filter_passes_when_above_threshold() -> None:
    state = state_with_volumes([10.0] * 6)
    features = {latest_close_time(state): feature_row(latest_close_time(state), taker_buy_ratio=0.7)}
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        min_taker_buy_ratio=0.5,
        features_by_close_time=features,
    )
    assert strat.on_market_state(state).side == SignalSide.BUY


def test_optional_taker_missing_data_is_skipped() -> None:
    state = state_with_volumes([10.0] * 6)
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        min_taker_buy_ratio=0.5,
        require_taker=False,
        features_by_close_time={},  # no row for the candle
    )
    # Optional filter with missing data must not block and must not pretend.
    assert strat.on_market_state(state).side == SignalSide.BUY


# --- order book (NOT available historically) ---

def test_optional_order_book_missing_is_skipped() -> None:
    state = state_with_volumes([10.0] * 6)
    # Feature row exists but order_book_imbalance is None (historical reality).
    features = {latest_close_time(state): feature_row(latest_close_time(state), order_book_imbalance=None)}
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        min_order_book_imbalance=0.1,
        require_order_book=False,
        features_by_close_time=features,
    )
    assert strat.on_market_state(state).side == SignalSide.BUY


def test_required_order_book_missing_raises() -> None:
    state = state_with_volumes([10.0] * 6)
    features = {latest_close_time(state): feature_row(latest_close_time(state), order_book_imbalance=None)}
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.BUY),
        min_order_book_imbalance=0.1,
        require_order_book=True,
        features_by_close_time=features,
    )
    with pytest.raises(FeatureUnavailableError):
        strat.on_market_state(state)


def test_required_filter_without_feature_source_raises_at_construction() -> None:
    with pytest.raises(ValueError):
        FeatureFilteredStrategy(
            base_strategy=StubStrategy(SignalSide.BUY),
            min_order_book_imbalance=0.1,
            require_order_book=True,
            features_by_close_time=None,
        )


# --- passthrough behavior ---

def test_sell_passes_through_unfiltered() -> None:
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.SELL),
        min_volume_spike_ratio=99.0,  # would block a BUY, must not touch SELL
    )
    signal = strat.on_market_state(state_with_volumes([10.0] * 6))
    assert signal.side == SignalSide.SELL
    assert "base_exit_passthrough" in signal.reason


def test_hold_passes_through() -> None:
    strat = FeatureFilteredStrategy(
        base_strategy=StubStrategy(SignalSide.HOLD),
        min_volume_spike_ratio=1.0,
    )
    signal = strat.on_market_state(state_with_volumes([10.0] * 6))
    assert signal.side == SignalSide.HOLD
    assert "base_hold" in signal.reason
