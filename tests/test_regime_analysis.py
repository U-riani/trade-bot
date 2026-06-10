from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.metrics import BacktestTrade
from app.backtesting.regime_analysis import (
    attribute_trades_by_regime,
    build_regime_buckets,
    label_regimes,
    regime_candle_counts,
)
from app.market.models import Candle
from app.strategy.market_regime import MarketRegime

START = datetime(2026, 1, 1, tzinfo=UTC)


def make_candle(index: int, close: float) -> Candle:
    start = START + timedelta(minutes=index)
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


def make_trade(entry_index: int, pnl: str) -> BacktestTrade:
    entry = START + timedelta(minutes=entry_index) + timedelta(minutes=1)  # == candle.close_time
    return BacktestTrade(
        symbol="BTCUSDT",
        entry_time=entry,
        exit_time=entry + timedelta(minutes=5),
        entry_price=Decimal("100"),
        exit_price=Decimal("100"),
        quantity=Decimal("1"),
        quote_amount=Decimal("100"),
        entry_fee=Decimal("0"),
        exit_fee=Decimal("0"),
        pnl=Decimal(pnl),
        entry_reason="test",
        exit_reason="test",
    )


def test_label_regimes_uptrend_is_bullish() -> None:
    # A long steady climb should classify the latest candles as bullish.
    candles = [make_candle(i, 100 + i * 0.5) for i in range(300)]
    labels = label_regimes(candles, slow_ema_period=200, slope_lookback=20)

    counts = regime_candle_counts(labels)
    assert counts[MarketRegime.BULLISH] > 0
    assert counts[MarketRegime.BEARISH] == 0
    # The very last candle of a clean uptrend must be bullish.
    assert labels[candles[-1].close_time] == MarketRegime.BULLISH


def test_label_regimes_short_history_is_unknown() -> None:
    candles = [make_candle(i, 100) for i in range(50)]  # < slow_ema + slope needs
    labels = label_regimes(candles, slow_ema_period=200, slope_lookback=20)
    assert all(regime == MarketRegime.UNKNOWN for regime in labels.values())


def test_attribute_trades_by_regime_buckets_by_entry() -> None:
    labels = {
        START + timedelta(minutes=1): MarketRegime.BULLISH,
        START + timedelta(minutes=2): MarketRegime.SIDEWAYS,
    }
    trades = [make_trade(0, "5"), make_trade(1, "-3")]  # entries at minute 1 and 2

    grouped = attribute_trades_by_regime(trades, labels)
    assert len(grouped[MarketRegime.BULLISH]) == 1
    assert grouped[MarketRegime.BULLISH][0].pnl == Decimal("5")
    assert len(grouped[MarketRegime.SIDEWAYS]) == 1
    assert grouped[MarketRegime.SIDEWAYS][0].pnl == Decimal("-3")


def test_unlabeled_entry_falls_into_unknown() -> None:
    trades = [make_trade(99, "1")]  # entry_time not in labels
    grouped = attribute_trades_by_regime(trades, {})
    assert len(grouped[MarketRegime.UNKNOWN]) == 1


def test_build_regime_buckets_computes_stats_and_pnl() -> None:
    labels = {
        START + timedelta(minutes=1): MarketRegime.SIDEWAYS,
        START + timedelta(minutes=2): MarketRegime.SIDEWAYS,
        START + timedelta(minutes=3): MarketRegime.SIDEWAYS,
    }
    trades = [make_trade(0, "10"), make_trade(1, "10"), make_trade(2, "-5")]

    buckets = {bucket.regime: bucket for bucket in build_regime_buckets(trades, labels)}
    sideways = buckets[MarketRegime.SIDEWAYS]
    assert sideways.candle_count == 3
    assert sideways.stats.num_trades == 3
    assert sideways.total_pnl == Decimal("15")
    assert sideways.stats.profit_factor == Decimal("4")  # 20 / 5
