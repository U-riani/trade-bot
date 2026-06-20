from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.backtesting.order_book_strategy import (
    OrderBookThresholdConfig,
    feature_value,
    quantile_threshold,
    rows_with_feature,
    run_order_book_threshold_backtest_with_diagnostics,
    split_feature_rows,
    split_rows_by_feature_coverage,
)
from app.market.features import MarketFeatures


def _row(index: int, *, close: float = 100.0, imbalance: float | None = None, minute_step: int = 1) -> MarketFeatures:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minute_step * index)
    return MarketFeatures(
        exchange="binance_spot", symbol="BTCUSDT", timeframe="1m", open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        close_price=close, volume=1.0, imbalance_top_20=imbalance,
    )


def test_quantile_threshold_nearest_rank() -> None:
    rows = [_row(i, imbalance=value) for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    assert quantile_threshold(rows, "imbalance_top_20", 0.8) == pytest.approx(0.4)


def test_rows_with_feature_filters_missing_for_threshold_only() -> None:
    rows = [_row(2, imbalance=0.3), _row(1, imbalance=None), _row(0, imbalance=0.1)]
    assert [feature_value(row, "imbalance_top_20") for row in rows_with_feature(rows, "imbalance_top_20")] == [0.1, 0.3]


def test_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="Unknown market feature"):
        feature_value(_row(0, imbalance=0.1), "not_real")


def test_gap_safe_strategy_keeps_missing_feature_candles_in_clock() -> None:
    rows = [
        _row(0, close=100, imbalance=0.9),  # signal
        _row(1, close=101, imbalance=None), # entry despite missing next feature
        _row(2, close=102, imbalance=None),
        _row(3, close=104, imbalance=0.1),  # exit after two real 1m bars
    ]
    outcome = run_order_book_threshold_backtest_with_diagnostics(
        rows=rows,
        config=OrderBookThresholdConfig("imbalance_top_20", 0.8, 2, "test", timeframe="1m"),
        symbol="BTCUSDT", initial_quote_balance=Decimal("1000"), quote_amount=Decimal("100"),
    )
    assert outcome.result.metrics.round_trips == 1
    assert outcome.result.trades[0].entry_time == rows[1].close_time
    assert outcome.result.trades[0].exit_time == rows[3].close_time
    assert outcome.diagnostics.skipped_gap_signals == 0


def test_gap_safe_strategy_skips_signal_that_would_cross_gap() -> None:
    rows = [
        _row(0, close=100, imbalance=0.9),
        _row(1, close=101, imbalance=None),
        _row(5, close=103, imbalance=None),  # four-minute gap after potential entry
        _row(6, close=104, imbalance=None),
    ]
    outcome = run_order_book_threshold_backtest_with_diagnostics(
        rows=rows,
        config=OrderBookThresholdConfig("imbalance_top_20", 0.8, 2, "test", timeframe="1m"),
        symbol="BTCUSDT", initial_quote_balance=Decimal("1000"), quote_amount=Decimal("100"),
    )
    assert outcome.result.metrics.round_trips == 0
    assert outcome.diagnostics.gap_count == 1
    assert outcome.diagnostics.skipped_gap_signals == 1


def test_low_tail_can_trigger_contrarian_entry() -> None:
    rows = [_row(0, close=100, imbalance=0.1), _row(1, close=101, imbalance=None), _row(2, close=102, imbalance=None)]
    outcome = run_order_book_threshold_backtest_with_diagnostics(
        rows=rows,
        config=OrderBookThresholdConfig("imbalance_top_20", 0.2, 1, "test", timeframe="1m", entry_tail="low"),
        symbol="BTCUSDT", initial_quote_balance=Decimal("1000"), quote_amount=Decimal("100"),
    )
    assert outcome.result.metrics.round_trips == 1
    assert outcome.result.trades[0].pnl > 0


def test_split_feature_rows_preserves_all_rows_chronologically() -> None:
    rows = [_row(i, imbalance=0.1 if i % 2 == 0 else None) for i in range(10)]
    train, validation = split_feature_rows(rows, train_ratio=Decimal("0.7"))
    assert len(train) == 7
    assert len(validation) == 3
    assert train[-1].close_time < validation[0].close_time


def test_split_rows_by_feature_coverage_uses_observed_window_not_old_history() -> None:
    rows = [
        _row(0, imbalance=None),
        _row(1, imbalance=None),
        _row(2, imbalance=0.1),
        _row(3, imbalance=None),
        _row(4, imbalance=0.2),
        _row(5, imbalance=None),
        _row(6, imbalance=0.3),
        _row(7, imbalance=None),
        _row(8, imbalance=0.4),
    ]
    coverage = split_rows_by_feature_coverage(
        rows, feature_name="imbalance_top_20", train_ratio=Decimal("0.5")
    )
    assert coverage.full_rows[0].close_time == rows[2].close_time
    assert coverage.full_rows[-1].close_time == rows[8].close_time
    assert [feature_value(row, "imbalance_top_20") for row in rows_with_feature(coverage.train_rows, "imbalance_top_20")] == [0.1, 0.2]
    assert [feature_value(row, "imbalance_top_20") for row in rows_with_feature(coverage.validation_rows, "imbalance_top_20")] == [0.3, 0.4]
    assert coverage.train_rows[-1].close_time < coverage.validation_rows[0].close_time
