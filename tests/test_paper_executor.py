from decimal import Decimal

import pytest

from app.execution.models import OrderRequest, OrderSide, OrderStatus, PortfolioSnapshot
from app.execution.paper_executor import PaperExecutor


@pytest.mark.asyncio
async def test_paper_buy_and_sell():
    executor = PaperExecutor(initial_quote_balance=Decimal("100"), symbol="BTCUSDT")
    executor.set_latest_price(Decimal("50000"))

    buy_result = await executor.execute(
        OrderRequest(symbol="BTCUSDT", side=OrderSide.BUY, quote_amount=Decimal("10"))
    )
    assert buy_result.status == OrderStatus.FILLED
    assert executor.portfolio_snapshot().has_open_position

    executor.set_latest_price(Decimal("51000"))
    sell_result = await executor.execute(OrderRequest(symbol="BTCUSDT", side=OrderSide.SELL))
    assert sell_result.status == OrderStatus.FILLED
    assert not executor.portfolio_snapshot().has_open_position
    assert executor.realized_pnl_today > 0


def test_paper_executor_restore_from_snapshot():
    executor = PaperExecutor(initial_quote_balance=Decimal("100"), symbol="BTCUSDT")
    executor.restore_from_snapshot(
        PortfolioSnapshot(
            quote_balance=Decimal("90"),
            position_quantity=Decimal("0.2"),
            position_avg_entry_price=Decimal("50"),
            realized_pnl_today=Decimal("1.5"),
            latest_price=Decimal("55"),
        )
    )

    snapshot = executor.portfolio_snapshot()
    assert snapshot.quote_balance == Decimal("90")
    assert snapshot.position_quantity == Decimal("0.2")
    assert snapshot.position_avg_entry_price == Decimal("50")
    assert snapshot.realized_pnl_today == Decimal("1.5")
    assert snapshot.latest_price == Decimal("55")
