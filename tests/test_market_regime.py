from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.market_regime import (
    MarketRegime,
    MarketRegimeFilteredStrategy,
    classify_market_regime,
)
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


def _candles_from_closes(closes: list[float]) -> list[Candle]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles: list[Candle] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        open_price = previous
        high = max(open_price, close) + 1
        low = min(open_price, close) - 1
        open_time = start + timedelta(minutes=index)
        candles.append(
            Candle(
                exchange="binance_spot",
                symbol="BTCUSDT",
                timeframe="1m",
                open_time=open_time,
                close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=1.0,
                is_closed=True,
            )
        )
        previous = close
    return candles


def _state_from_closes(closes: list[float]) -> MarketState:
    state = MarketState(symbol="BTCUSDT", max_candles=1000)
    for candle in _candles_from_closes(closes):
        state.add_candle(candle)
    return state


class _AlwaysBuyStrategy(Strategy):
    name = "always_buy"

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        return TradeSignal(
            strategy_name=self.name,
            symbol=market_state.symbol,
            side=SignalSide.BUY,
            confidence=0.5,
            reason="test_buy",
            created_at=utc_now(),
            suggested_quote_amount=Decimal("10"),
        )


class _AlwaysSellStrategy(Strategy):
    name = "always_sell"

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        return TradeSignal(
            strategy_name=self.name,
            symbol=market_state.symbol,
            side=SignalSide.SELL,
            confidence=0.5,
            reason="test_sell",
            created_at=utc_now(),
        )


def test_classify_market_regime_detects_bullish_trend() -> None:
    closes = [100 + (index * 0.2) for index in range(260)]
    snapshot = classify_market_regime(
        _state_from_closes(closes),
        fast_ema_period=20,
        slow_ema_period=50,
        slope_lookback=10,
        min_slope_pct=Decimal("0.01"),
        min_ema_gap_pct=Decimal("0.01"),
    )

    assert snapshot.regime == MarketRegime.BULLISH
    assert snapshot.ema_gap_pct is not None
    assert snapshot.ema_gap_pct > 0
    assert snapshot.slow_slope_pct is not None
    assert snapshot.slow_slope_pct > 0


def test_classify_market_regime_detects_bearish_trend() -> None:
    closes = [200 - (index * 0.2) for index in range(260)]
    snapshot = classify_market_regime(
        _state_from_closes(closes),
        fast_ema_period=20,
        slow_ema_period=50,
        slope_lookback=10,
        min_slope_pct=Decimal("0.01"),
        min_ema_gap_pct=Decimal("0.01"),
    )

    assert snapshot.regime == MarketRegime.BEARISH
    assert snapshot.ema_gap_pct is not None
    assert snapshot.ema_gap_pct < 0
    assert snapshot.slow_slope_pct is not None
    assert snapshot.slow_slope_pct < 0


def test_regime_filtered_strategy_blocks_buy_outside_bullish_regime() -> None:
    closes = [200 - (index * 0.2) for index in range(260)]
    strategy = MarketRegimeFilteredStrategy(
        base_strategy=_AlwaysBuyStrategy(),
        fast_ema_period=20,
        slow_ema_period=50,
        slope_lookback=10,
        min_slope_pct=Decimal("0.01"),
        min_ema_gap_pct=Decimal("0.01"),
    )

    signal = strategy.on_market_state(_state_from_closes(closes))

    assert signal.side == SignalSide.HOLD
    assert "buy_blocked_by" in signal.reason


def test_regime_filtered_strategy_allows_buy_in_bullish_regime() -> None:
    closes = [100 + (index * 0.2) for index in range(260)]
    strategy = MarketRegimeFilteredStrategy(
        base_strategy=_AlwaysBuyStrategy(),
        fast_ema_period=20,
        slow_ema_period=50,
        slope_lookback=10,
        min_slope_pct=Decimal("0.01"),
        min_ema_gap_pct=Decimal("0.01"),
    )

    signal = strategy.on_market_state(_state_from_closes(closes))

    assert signal.side == SignalSide.BUY
    assert "regime_confirmed_bullish" in signal.reason


def test_regime_filtered_strategy_passes_sell_through() -> None:
    closes = [200 - (index * 0.2) for index in range(260)]
    strategy = MarketRegimeFilteredStrategy(
        base_strategy=_AlwaysSellStrategy(),
        fast_ema_period=20,
        slow_ema_period=50,
        slope_lookback=10,
        min_slope_pct=Decimal("0.01"),
        min_ema_gap_pct=Decimal("0.01"),
    )

    signal = strategy.on_market_state(_state_from_closes(closes))

    assert signal.side == SignalSide.SELL
    assert "base_exit_passthrough" in signal.reason
