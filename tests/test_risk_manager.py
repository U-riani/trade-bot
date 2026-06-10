from decimal import Decimal

from app.execution.models import PortfolioSnapshot
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


def make_config() -> RiskConfig:
    return RiskConfig(
        max_order_usdt=Decimal("10"),
        max_position_usdt=Decimal("50"),
        max_daily_loss_usdt=Decimal("10"),
        max_trades_per_hour=5,
        cooldown_seconds=0,
    )


def test_risk_manager_approves_buy_when_valid():
    manager = RiskManager(make_config())
    signal = TradeSignal(
        strategy_name="test",
        symbol="BTCUSDT",
        side=SignalSide.BUY,
        confidence=0.8,
        reason="test_buy",
        created_at=utc_now(),
    )
    portfolio = PortfolioSnapshot(
        quote_balance=Decimal("100"),
        position_quantity=Decimal("0"),
        position_avg_entry_price=Decimal("0"),
        realized_pnl_today=Decimal("0"),
        latest_price=Decimal("50000"),
    )
    decision = manager.evaluate(signal, portfolio)
    assert decision.approved
    assert decision.order_request is not None


def test_risk_manager_rejects_sell_without_position():
    manager = RiskManager(make_config())
    signal = TradeSignal(
        strategy_name="test",
        symbol="BTCUSDT",
        side=SignalSide.SELL,
        confidence=0.8,
        reason="test_sell",
        created_at=utc_now(),
    )
    portfolio = PortfolioSnapshot(
        quote_balance=Decimal("100"),
        position_quantity=Decimal("0"),
        position_avg_entry_price=Decimal("0"),
        realized_pnl_today=Decimal("0"),
        latest_price=Decimal("50000"),
    )
    decision = manager.evaluate(signal, portfolio)
    assert not decision.approved
    assert decision.reason == "no_open_position"
