from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.market.models import Candle

from app.backtesting.multitimeframe_pullback_strategy import (
    MultiTimeframePullbackConfig,
    PullbackSetup,
    build_pullback_setup_cache,
    order_book_reversal_matches,
    run_multitimeframe_pullback_backtest,
)
from app.market.features import MarketFeatures


def _row(
    index: int,
    *,
    close: float,
    imbalance: float | None,
    timeframe: str = "1m",
    minute: int | None = None,
) -> MarketFeatures:
    size = {"1m": 1, "5m": 5, "15m": 15}[timeframe]
    offset = index * size if minute is None else minute
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset)
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe=timeframe,
        open_time=open_time,
        close_time=open_time + timedelta(minutes=size) - timedelta(milliseconds=1),
        close_price=close,
        volume=1.0,
        imbalance_top_20=imbalance,
    )


def _config(*, gated: bool) -> MultiTimeframePullbackConfig:
    return MultiTimeframePullbackConfig(
        feature_name="imbalance_top_20",
        reversal_threshold=0.5,
        horizon_bars=1,
        strategy_name="test_v29",
        require_order_book_reversal=gated,
        trend_ema_period=3,
        pullback_fast_ema_period=2,
        pullback_trend_ema_period=3,
        pullback_rsi_period=2,
    )


def test_order_book_reversal_requires_nonpositive_previous_value() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.1),
        _row(1, close=101, imbalance=0.7),
    ]
    assert order_book_reversal_matches(rows, index=1, config=_config(gated=True))

    rows[0] = _row(0, close=100, imbalance=0.2)
    assert not order_book_reversal_matches(rows, index=1, config=_config(gated=True))


def test_reversal_gate_enters_next_1m_candle_and_exits_after_horizon() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.8),  # signal
        _row(2, close=102, imbalance=0.1),  # entry
        _row(3, close=104, imbalance=0.1),  # exit
    ]
    cache = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "15m_trend_plus_5m_pullback"),
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    outcome = run_multitimeframe_pullback_backtest(
        entry_rows=rows,
        pullback_rows=[],
        trend_rows=[],
        config=_config(gated=True),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        setup_cache=cache,
    )
    assert outcome.result.metrics.round_trips == 1
    assert outcome.result.trades[0].entry_time == rows[2].close_time
    assert outcome.result.trades[0].exit_time == rows[3].close_time
    assert outcome.diagnostics.reversal_gate_passed == 1


def test_gap_safe_path_skips_reversal_trade() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.8),  # signal
        _row(2, close=102, imbalance=0.1, minute=5),  # gap before entry
        _row(3, close=104, imbalance=0.1, minute=6),
    ]
    cache = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "15m_trend_plus_5m_pullback"),
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    outcome = run_multitimeframe_pullback_backtest(
        entry_rows=rows,
        pullback_rows=[],
        trend_rows=[],
        config=_config(gated=True),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        setup_cache=cache,
    )
    assert outcome.result.metrics.round_trips == 0
    assert outcome.diagnostics.skipped_gap_signals == 1


def test_price_setup_cache_does_not_use_missing_future_higher_timeframe_data() -> None:
    base = [_row(index, close=100 + index, imbalance=0.1) for index in range(4)]
    cache = build_pullback_setup_cache(
        entry_rows=base,
        pullback_rows=[],
        trend_rows=[],
        config=_config(gated=False),
    )
    assert len(cache) == len(base)
    assert all(item.reason == "missing_higher_timeframe_candle" for item in cache)


def test_v29_1_merge_preserves_complete_price_timeline() -> None:
    from scripts.backtest_multitimeframe_pullback_strategy import _merge_candles_with_observations

    start = datetime(2026, 1, 1, tzinfo=UTC)
    candles = [
        Candle(
            exchange="binance_spot", symbol="BTCUSDT", timeframe="1m",
            open_time=start + timedelta(minutes=index),
            close_time=start + timedelta(minutes=index + 1) - timedelta(milliseconds=1),
            open=100 + index, high=101 + index, low=99 + index, close=100 + index, volume=1.0, is_closed=True,
        )
        for index in range(3)
    ]
    observation = _row(1, close=101, imbalance=0.42)
    rows = _merge_candles_with_observations(candles, [observation])

    assert len(rows) == 3
    assert [row.close_price for row in rows] == [100, 101, 102]
    assert rows[0].imbalance_top_20 is None
    assert rows[1].imbalance_top_20 == 0.42
    assert rows[2].imbalance_top_20 is None
