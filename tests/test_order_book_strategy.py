from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.backtesting.order_book_strategy import (
    OrderBookThresholdConfig,
    feature_value,
    quantile_threshold,
    rows_with_feature,
    run_order_book_threshold_backtest,
    split_feature_rows,
)
from app.market.features import MarketFeatures


def _row(index: int, *, close: float = 100.0, imbalance: float | None = None) -> MarketFeatures:
    open_time = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=5 * index)
    return MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="5m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=5) - timedelta(milliseconds=1),
        close_price=close,
        volume=1.0,
        imbalance_top_20=imbalance,
    )


def test_quantile_threshold_nearest_rank() -> None:
    rows = [_row(i, imbalance=value) for i, value in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    assert quantile_threshold(rows, "imbalance_top_20", 0.8) == pytest.approx(0.4)


def test_rows_with_feature_filters_missing_and_sorts() -> None:
    rows = [_row(2, imbalance=0.3), _row(1, imbalance=None), _row(0, imbalance=0.1)]
    filtered = rows_with_feature(rows, "imbalance_top_20")
    assert [row.close_price for row in filtered] == [100.0, 100.0]
    assert [feature_value(row, "imbalance_top_20") for row in filtered] == [0.1, 0.3]


def test_unknown_feature_raises() -> None:
    with pytest.raises(ValueError, match="Unknown market feature"):
        feature_value(_row(0, imbalance=0.1), "definitely_not_real")


def test_strategy_enters_next_row_not_signal_row_and_exits_after_horizon() -> None:
    rows = [
        _row(0, close=100, imbalance=0.9),  # signal only
        _row(1, close=101, imbalance=0.1),  # entry here
        _row(2, close=102, imbalance=0.1),
        _row(3, close=104, imbalance=0.1),  # exit here with horizon 2
    ]
    result = run_order_book_threshold_backtest(
        rows=rows,
        config=OrderBookThresholdConfig(
            feature_name="imbalance_top_20",
            entry_threshold=0.8,
            horizon_bars=2,
            strategy_name="test_ob_threshold",
        ),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )
    assert result.metrics.round_trips == 1
    trade = result.trades[0]
    assert trade.entry_time == rows[1].close_time
    assert trade.exit_time == rows[3].close_time
    assert trade.entry_price == Decimal("101")
    assert trade.exit_price == Decimal("104")
    assert trade.pnl > 0


def test_strategy_applies_fee_and_slippage() -> None:
    rows = [_row(0, close=100, imbalance=0.9), _row(1, close=100, imbalance=0.1), _row(2, close=100, imbalance=0.1)]
    result = run_order_book_threshold_backtest(
        rows=rows,
        config=OrderBookThresholdConfig(
            feature_name="imbalance_top_20",
            entry_threshold=0.8,
            horizon_bars=1,
            strategy_name="test_ob_threshold",
        ),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
        fee_rate_pct=Decimal("0.1"),
        slippage_pct=Decimal("0.1"),
    )
    assert result.metrics.round_trips == 1
    assert result.metrics.total_fees > 0
    assert result.trades[0].pnl < 0


def test_split_feature_rows_chronological() -> None:
    rows = [_row(i, imbalance=0.1) for i in range(10)]
    train, validation = split_feature_rows(rows, train_ratio=Decimal("0.7"))
    assert len(train) == 7
    assert len(validation) == 3
    assert train[-1].close_time < validation[0].close_time
