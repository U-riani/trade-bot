from decimal import Decimal

from app.execution.models import PortfolioSnapshot
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.risk.position_guard import build_position_exit_signal, calculate_exit_levels
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


def make_portfolio(*, latest_price: Decimal, avg_entry: Decimal = Decimal("100")) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        quote_balance=Decimal("90"),
        position_quantity=Decimal("0.1"),
        position_avg_entry_price=avg_entry,
        realized_pnl_today=Decimal("0"),
        latest_price=latest_price,
    )


def make_risk_manager(cooldown_seconds: int = 60) -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_order_usdt=Decimal("10"),
            max_position_usdt=Decimal("50"),
            max_daily_loss_usdt=Decimal("10"),
            max_trades_per_hour=1,
            cooldown_seconds=cooldown_seconds,
        )
    )


def test_position_guard_calculates_exit_levels():
    levels = calculate_exit_levels(
        avg_entry_price=Decimal("100"),
        stop_loss_pct=Decimal("0.7"),
        take_profit_pct=Decimal("1.2"),
    )
    assert levels.stop_loss_price == Decimal("99.300")
    assert levels.take_profit_price == Decimal("101.200")


def test_position_guard_emits_stop_loss_signal():
    signal = build_position_exit_signal(
        portfolio=make_portfolio(latest_price=Decimal("99.30")),
        symbol="BTCUSDT",
        stop_loss_pct=Decimal("0.7"),
        take_profit_pct=Decimal("1.2"),
    )

    assert signal is not None
    assert signal.side == SignalSide.SELL
    assert signal.strategy_name == "position_guard"
    assert signal.is_protective_exit
    assert signal.reason.startswith("stop_loss_triggered")


def test_position_guard_emits_take_profit_signal():
    signal = build_position_exit_signal(
        portfolio=make_portfolio(latest_price=Decimal("101.20")),
        symbol="BTCUSDT",
        stop_loss_pct=Decimal("0.7"),
        take_profit_pct=Decimal("1.2"),
    )

    assert signal is not None
    assert signal.side == SignalSide.SELL
    assert signal.is_protective_exit
    assert signal.reason.startswith("take_profit_triggered")


def test_position_guard_no_signal_without_open_position():
    signal = build_position_exit_signal(
        portfolio=PortfolioSnapshot(
            quote_balance=Decimal("100"),
            position_quantity=Decimal("0"),
            position_avg_entry_price=Decimal("0"),
            realized_pnl_today=Decimal("0"),
            latest_price=Decimal("90"),
        ),
        symbol="BTCUSDT",
        stop_loss_pct=Decimal("0.7"),
        take_profit_pct=Decimal("1.2"),
    )

    assert signal is None


def test_risk_manager_allows_sell_exit_even_when_daily_loss_reached():
    manager = make_risk_manager()
    signal = TradeSignal(
        strategy_name="position_guard",
        symbol="BTCUSDT",
        side=SignalSide.SELL,
        confidence=1.0,
        reason="stop_loss_triggered",
        created_at=utc_now(),
        is_protective_exit=True,
    )
    portfolio = PortfolioSnapshot(
        quote_balance=Decimal("90"),
        position_quantity=Decimal("0.1"),
        position_avg_entry_price=Decimal("100"),
        realized_pnl_today=Decimal("-100"),
        latest_price=Decimal("99"),
    )

    decision = manager.evaluate(signal, portfolio)
    assert decision.approved
    assert decision.order_request is not None
    assert decision.order_request.side.value == "sell"
