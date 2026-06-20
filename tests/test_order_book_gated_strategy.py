from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.order_book_gated_strategy import (
    GatedStrategyConfig,
    OrderBookGateConfig,
    PriceRuleConfig,
    gate_matches,
    price_rule_signal,
    run_order_book_gated_backtest,
)
from app.market.features import MarketFeatures


def _row(index: int, *, close: float, imbalance: float | None, minute: int | None = None) -> MarketFeatures:
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


def _breakout_rule() -> PriceRuleConfig:
    return PriceRuleConfig(
        strategy_kind="breakout_momentum",
        fast_ema_period=2,
        slow_ema_period=3,
        trend_ema_period=2,
        rsi_period=2,
        breakout_lookback=2,
        mean_reversion_lookback=3,
    )


def test_gate_matches_high_and_low_tails() -> None:
    row = _row(0, close=100, imbalance=0.2)
    assert gate_matches(row, OrderBookGateConfig("imbalance_top_20", "high", 0.1))
    assert not gate_matches(row, OrderBookGateConfig("imbalance_top_20", "high", 0.3))
    assert gate_matches(row, OrderBookGateConfig("imbalance_top_20", "low", 0.3))
    assert not gate_matches(row, OrderBookGateConfig("imbalance_top_20", "low", 0.1))


def test_breakout_rule_requires_contiguous_warm_history() -> None:
    rows = [_row(0, close=100, imbalance=0.9), _row(1, close=101, imbalance=0.9)]
    config = GatedStrategyConfig(
        price_rule=_breakout_rule(),
        horizon_bars=1,
        timeframe="1m",
        strategy_name="test",
    )
    assert price_rule_signal(rows, index=1, config=config) is None


def test_order_book_gate_filters_otherwise_identical_breakout() -> None:
    rows = [
        _row(0, close=100, imbalance=0.1),
        _row(1, close=101, imbalance=0.1),
        _row(2, close=103, imbalance=0.9),  # breakout signal
        _row(3, close=104, imbalance=0.1),  # entry
        _row(4, close=106, imbalance=0.1),  # exit
    ]
    baseline_config = GatedStrategyConfig(
        price_rule=_breakout_rule(),
        horizon_bars=1,
        timeframe="1m",
        strategy_name="baseline",
    )
    gated_config = GatedStrategyConfig(
        price_rule=_breakout_rule(),
        horizon_bars=1,
        timeframe="1m",
        strategy_name="gated",
        order_book_gate=OrderBookGateConfig("imbalance_top_20", "high", 0.8),
    )
    baseline = run_order_book_gated_backtest(
        rows=rows,
        config=baseline_config,
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
    )
    gated = run_order_book_gated_backtest(
        rows=rows,
        config=gated_config,
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
    )
    assert baseline.result.metrics.round_trips == 1
    assert gated.result.metrics.round_trips == 1
    assert gated.diagnostics.gate_passed_signals == 1
    assert baseline.result.trades[0].entry_time == rows[3].close_time
    assert gated.result.trades[0].pnl > 0


def test_gap_safe_path_skips_breakout_trade() -> None:
    rows = [
        _row(0, close=100, imbalance=0.1),
        _row(1, close=101, imbalance=0.1),
        _row(2, close=103, imbalance=0.9),  # signal
        _row(3, close=104, imbalance=0.1, minute=5),  # gap before planned entry
        _row(4, close=106, imbalance=0.1, minute=6),
    ]
    config = GatedStrategyConfig(
        price_rule=_breakout_rule(),
        horizon_bars=1,
        timeframe="1m",
        strategy_name="gated",
        order_book_gate=OrderBookGateConfig("imbalance_top_20", "high", 0.8),
    )
    outcome = run_order_book_gated_backtest(
        rows=rows,
        config=config,
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        quote_amount=Decimal("100"),
    )
    assert outcome.result.metrics.round_trips == 0
    assert outcome.diagnostics.skipped_gap_signals == 1
