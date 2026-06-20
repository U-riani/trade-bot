from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.multitimeframe_pullback_delta_strategy import (
    MultiTimeframeDeltaConfig,
    observed_feature_delta,
    order_book_delta_matches,
    positive_feature_deltas,
    run_multitimeframe_pullback_delta_backtest,
)
from app.backtesting.multitimeframe_pullback_strategy import PullbackSetup
from app.market.features import MarketFeatures


def _row(
    index: int,
    *,
    close: float,
    imbalance: float | None,
    minute: int | None = None,
) -> MarketFeatures:
    offset = index if minute is None else minute
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset)
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        close_price=close,
        volume=1.0,
        imbalance_top_20=imbalance,
    )


def _config(*, gated: bool, threshold: float = 0.4) -> MultiTimeframeDeltaConfig:
    return MultiTimeframeDeltaConfig(
        feature_name="imbalance_top_20",
        delta_threshold=threshold,
        horizon_bars=1,
        strategy_name="test_v30",
        require_order_book_delta=gated,
        min_current_imbalance=0.0,
        trend_ema_period=3,
        pullback_fast_ema_period=2,
        pullback_trend_ema_period=3,
        pullback_rsi_period=2,
    )


def test_positive_feature_deltas_use_only_contiguous_observed_pairs() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.1),  # +0.3
        _row(2, close=102, imbalance=0.4),  # +0.3
        _row(3, close=103, imbalance=0.9, minute=8),  # gap, excluded
        _row(4, close=104, imbalance=0.8, minute=9),  # negative, excluded
    ]
    assert positive_feature_deltas(rows, feature_name="imbalance_top_20") == [0.30000000000000004, 0.30000000000000004]


def test_delta_gate_requires_positive_improvement_and_nonnegative_current_imbalance() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.3),
    ]
    matched, pair = order_book_delta_matches(rows, index=1, config=_config(gated=True))
    assert matched
    assert pair is not None
    assert pair[2] == 0.5

    rows[1] = _row(1, close=101, imbalance=-0.1)
    matched, pair = order_book_delta_matches(rows, index=1, config=_config(gated=True))
    assert not matched
    assert pair is not None


def test_missing_or_gapped_pair_is_unknown_not_a_zero_delta() -> None:
    rows = [
        _row(0, close=100, imbalance=0.1),
        _row(1, close=101, imbalance=0.5, minute=4),
    ]
    assert observed_feature_delta(rows, index=1, feature_name="imbalance_top_20") is None

    rows = [_row(0, close=100, imbalance=None), _row(1, close=101, imbalance=0.5)]
    assert observed_feature_delta(rows, index=1, feature_name="imbalance_top_20") is None


def test_delta_gate_enters_next_candle_and_exits_after_horizon() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.5),  # signal, delta +0.7
        _row(2, close=102, imbalance=0.1),  # entry
        _row(3, close=104, imbalance=0.1),  # exit
    ]
    cache = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "15m_trend_plus_5m_pullback"),
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    outcome = run_multitimeframe_pullback_delta_backtest(
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
    assert outcome.diagnostics.usable_order_book_delta_at_setup == 1
    assert outcome.diagnostics.delta_gate_passed == 1


def test_delta_gate_keeps_baseline_and_gated_signal_counts_separate() -> None:
    rows = [
        _row(0, close=100, imbalance=-0.2),
        _row(1, close=101, imbalance=0.0),  # delta +0.2, below gate
        _row(2, close=102, imbalance=0.1),
        _row(3, close=103, imbalance=0.1),
    ]
    cache = [
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(True, True, "15m_trend_plus_5m_pullback"),
        PullbackSetup(False, False, "same_5m_bar"),
        PullbackSetup(False, False, "same_5m_bar"),
    ]
    baseline = run_multitimeframe_pullback_delta_backtest(
        entry_rows=rows,
        pullback_rows=[],
        trend_rows=[],
        config=_config(gated=False),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        setup_cache=cache,
    )
    gated = run_multitimeframe_pullback_delta_backtest(
        entry_rows=rows,
        pullback_rows=[],
        trend_rows=[],
        config=_config(gated=True, threshold=0.4),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        setup_cache=cache,
    )
    assert baseline.diagnostics.price_only_setup_candidates == 1
    assert baseline.result.metrics.round_trips == 1
    assert gated.diagnostics.usable_order_book_delta_at_setup == 1
    assert gated.diagnostics.delta_gate_rejected == 1
    assert gated.result.metrics.round_trips == 0
