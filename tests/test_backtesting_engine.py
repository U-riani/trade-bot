from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.backtesting.engine import BacktestEngine
from app.market.models import Candle
from app.market.state import MarketState
from app.strategy.base import Strategy
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


class BuyEveryTimeStrategy(Strategy):
    name = "buy_every_time"

    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        return TradeSignal(
            strategy_name=self.name,
            symbol=market_state.symbol,
            side=SignalSide.BUY,
            confidence=1.0,
            reason="test_buy",
            created_at=utc_now(),
            suggested_quote_amount=Decimal("10"),
        )


def make_candle(index: int, close: float) -> Candle:
    start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
    return Candle(
        exchange="binance",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def test_backtest_executes_buy_and_take_profit_sell() -> None:
    engine = BacktestEngine(
        strategy=BuyEveryTimeStrategy(),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        max_order_usdt=Decimal("10"),
        max_position_usdt=Decimal("50"),
        stop_loss_pct=Decimal("5"),
        take_profit_pct=Decimal("1"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )

    result = engine.run([make_candle(0, 100), make_candle(1, 102)])

    assert result.metrics.executed_orders == 2
    assert result.metrics.round_trips == 1
    assert result.metrics.winning_trades == 1
    assert result.metrics.realized_pnl == Decimal("0.2")
    assert result.metrics.final_equity == Decimal("1000.2")
    assert not result.metrics.has_open_position


def test_backtest_keeps_single_open_position_when_no_exit() -> None:
    engine = BacktestEngine(
        strategy=BuyEveryTimeStrategy(),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        max_order_usdt=Decimal("10"),
        max_position_usdt=Decimal("50"),
        stop_loss_pct=Decimal("5"),
        take_profit_pct=Decimal("10"),
        fee_rate_pct=Decimal("0"),
        slippage_pct=Decimal("0"),
    )

    result = engine.run([make_candle(0, 100), make_candle(1, 100.5), make_candle(2, 100.7)])

    assert result.metrics.executed_orders == 1
    assert result.metrics.round_trips == 0
    assert result.metrics.has_open_position
    assert result.metrics.open_position_quantity > 0


def test_backtest_applies_fee_and_slippage() -> None:
    engine = BacktestEngine(
        strategy=BuyEveryTimeStrategy(),
        symbol="BTCUSDT",
        initial_quote_balance=Decimal("1000"),
        max_order_usdt=Decimal("10"),
        max_position_usdt=Decimal("50"),
        stop_loss_pct=Decimal("5"),
        take_profit_pct=Decimal("1"),
        fee_rate_pct=Decimal("0.1"),
        slippage_pct=Decimal("0.1"),
    )

    result = engine.run([make_candle(0, 100), make_candle(1, 102)])

    assert result.metrics.executed_orders == 2
    assert result.metrics.total_fees > 0
    assert result.metrics.final_equity < Decimal("1000.2")
    assert result.trades[0].entry_fee > 0
    assert result.trades[0].exit_fee > 0
    assert result.trades[0].entry_price > Decimal("100")
    assert result.trades[0].exit_price < Decimal("102")
