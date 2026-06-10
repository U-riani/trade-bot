from __future__ import annotations

import asyncio
from decimal import Decimal

from app.config.logging import configure_logging, get_logger
from app.execution.models import OrderStatus, PortfolioSnapshot
from app.execution.paper_executor import PaperExecutor
from app.risk.manager import RiskManager
from app.risk.models import RiskConfig
from app.risk.position_guard import build_position_exit_signal
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now

logger = get_logger(__name__)


def build_risk_manager() -> RiskManager:
    return RiskManager(
        RiskConfig(
            max_order_usdt=Decimal("10"),
            max_position_usdt=Decimal("50"),
            max_daily_loss_usdt=Decimal("10"),
            max_trades_per_hour=10,
            cooldown_seconds=0,
            allow_only_one_open_position=True,
        )
    )


def log_snapshot(label: str, snapshot: PortfolioSnapshot) -> None:
    logger.info(
        label,
        quote_balance=str(snapshot.quote_balance),
        position_quantity=str(snapshot.position_quantity),
        position_avg_entry_price=str(snapshot.position_avg_entry_price),
        realized_pnl_today=str(snapshot.realized_pnl_today),
        latest_price=str(snapshot.latest_price),
    )


async def main() -> None:
    configure_logging("INFO")

    symbol = "BTCUSDT"
    executor = PaperExecutor(initial_quote_balance=Decimal("1000"), symbol=symbol)
    risk_manager = build_risk_manager()

    executor.set_latest_price(Decimal("100"))
    log_snapshot("paper_cycle_initial_snapshot", executor.portfolio_snapshot())

    buy_signal = TradeSignal(
        strategy_name="paper_cycle_demo",
        symbol=symbol,
        side=SignalSide.BUY,
        confidence=1.0,
        reason="demo_buy_entry",
        created_at=utc_now(),
        suggested_quote_amount=Decimal("10"),
    )
    buy_decision = risk_manager.evaluate(buy_signal, executor.portfolio_snapshot())
    logger.info("paper_cycle_buy_risk_decision", approved=buy_decision.approved, reason=buy_decision.reason)
    if not buy_decision.approved or buy_decision.order_request is None:
        raise SystemExit("BUY was rejected, paper cycle demo cannot continue")

    buy_result = await executor.execute(buy_decision.order_request)
    logger.info(
        "paper_cycle_buy_result",
        status=buy_result.status.value,
        executed_quantity=str(buy_result.executed_quantity),
        executed_quote_quantity=str(buy_result.executed_quote_quantity),
    )
    if buy_result.status != OrderStatus.FILLED:
        raise SystemExit("BUY was not filled")
    risk_manager.register_executed_trade()
    log_snapshot("paper_cycle_after_buy_snapshot", executor.portfolio_snapshot())

    # Force a take-profit condition for demonstration. The live bot uses actual candle prices.
    executor.set_latest_price(Decimal("102"))
    exit_signal = build_position_exit_signal(
        portfolio=executor.portfolio_snapshot(),
        symbol=symbol,
        stop_loss_pct=Decimal("0.7"),
        take_profit_pct=Decimal("1.2"),
    )
    if exit_signal is None:
        raise SystemExit("Expected take-profit exit signal, but got none")

    logger.info(
        "paper_cycle_exit_signal",
        side=exit_signal.side.value,
        confidence=exit_signal.confidence,
        reason=exit_signal.reason,
    )
    sell_decision = risk_manager.evaluate(exit_signal, executor.portfolio_snapshot())
    logger.info("paper_cycle_sell_risk_decision", approved=sell_decision.approved, reason=sell_decision.reason)
    if not sell_decision.approved or sell_decision.order_request is None:
        raise SystemExit("SELL was rejected, paper cycle demo cannot close position")

    sell_result = await executor.execute(sell_decision.order_request)
    logger.info(
        "paper_cycle_sell_result",
        status=sell_result.status.value,
        executed_quantity=str(sell_result.executed_quantity),
        executed_quote_quantity=str(sell_result.executed_quote_quantity),
        raw_response=sell_result.raw_response,
    )
    if sell_result.status != OrderStatus.FILLED:
        raise SystemExit("SELL was not filled")
    risk_manager.register_executed_trade()
    log_snapshot("paper_cycle_final_snapshot", executor.portfolio_snapshot())

    if executor.portfolio_snapshot().has_open_position:
        raise SystemExit("Expected final position to be closed")
    logger.info("paper_cycle_demo_success")


if __name__ == "__main__":
    asyncio.run(main())
