from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.optimizer import generate_parameter_sets, optimize_parameter_grid
from app.market.models import Candle


def make_candle(index: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        volume=1.0,
    )


def test_generate_parameter_sets_filters_invalid_pairs() -> None:
    parameter_sets = generate_parameter_sets(
        ema_fast_values=[9, 21],
        ema_slow_values=[12, 21],
        rsi_period_values=[14],
        rsi_buy_min_values=[45, 70],
        rsi_buy_max_values=[60],
        rsi_sell_min_values=[75],
        stop_loss_pct_values=[Decimal("0.7")],
        take_profit_pct_values=[Decimal("1.2")],
    )

    assert {item.key for item in parameter_sets} == {
        "ema9_12__rsi14_45-60_sell75__sl0.7_tp1.2__trend0_gap0_atr0_0",
        "ema9_21__rsi14_45-60_sell75__sl0.7_tp1.2__trend0_gap0_atr0_0",
    }


def test_optimize_parameter_grid_returns_ranked_results() -> None:
    closes = [100 + ((index % 10) - 5) + (index * 0.05) for index in range(120)]
    candles = [make_candle(index, close) for index, close in enumerate(closes)]
    parameter_sets = generate_parameter_sets(
        ema_fast_values=[5, 9],
        ema_slow_values=[21],
        rsi_period_values=[14],
        rsi_buy_min_values=[40],
        rsi_buy_max_values=[75],
        rsi_sell_min_values=[70],
        stop_loss_pct_values=[Decimal("0.7")],
        take_profit_pct_values=[Decimal("1.2")],
    )

    results = optimize_parameter_grid(
        candles=candles,
        symbol="BTCUSDT",
        parameter_sets=parameter_sets,
        initial_quote_balance=Decimal("1000"),
        max_order_usdt=Decimal("10"),
        max_position_usdt=Decimal("50"),
        allow_only_one_open_position=True,
        fee_rate_pct=Decimal("0.1"),
        slippage_pct=Decimal("0.02"),
        min_round_trips=0,
    )

    assert [item.rank for item in results] == [1, 2]
    assert results[0].score >= results[1].score
    assert all(item.metrics.candles_processed == len(candles) for item in results)



def test_generate_parameter_sets_supports_v17_filters() -> None:
    parameter_sets = generate_parameter_sets(
        ema_fast_values=[12],
        ema_slow_values=[34],
        rsi_period_values=[14],
        rsi_buy_min_values=[45],
        rsi_buy_max_values=[65],
        rsi_sell_min_values=[70],
        stop_loss_pct_values=[Decimal("0.7")],
        take_profit_pct_values=[Decimal("1.2")],
        trend_ema_period_values=[None, 200],
        min_ema_gap_pct_values=[Decimal("0"), Decimal("0.05")],
        atr_period_values=[None, 14],
        min_atr_pct_values=[Decimal("0"), Decimal("0.08")],
    )

    assert len(parameter_sets) == 12
    assert any(item.trend_ema_period == 200 for item in parameter_sets)
    assert any(item.min_ema_gap_pct == Decimal("0.05") for item in parameter_sets)
    assert any(item.atr_period == 14 and item.min_atr_pct == Decimal("0.08") for item in parameter_sets)
    assert all(not (item.atr_period is None and item.min_atr_pct > 0) for item in parameter_sets)
