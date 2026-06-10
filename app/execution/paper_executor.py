from __future__ import annotations

from decimal import Decimal

from app.config.logging import get_logger
from app.execution.executor import OrderExecutor
from app.execution.models import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    PortfolioSnapshot,
    Position,
)

logger = get_logger(__name__)


class PaperExecutor(OrderExecutor):
    def __init__(self, initial_quote_balance: Decimal, symbol: str) -> None:
        self.quote_balance = initial_quote_balance
        self.position = Position(symbol=symbol)
        self.latest_price: Decimal | None = None
        self.realized_pnl_today = Decimal("0")

    def set_latest_price(self, latest_price: Decimal) -> None:
        self.latest_price = latest_price

    def restore_from_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self.quote_balance = snapshot.quote_balance
        self.position.quantity = snapshot.position_quantity
        self.position.avg_entry_price = snapshot.position_avg_entry_price
        self.position.realized_pnl = snapshot.realized_pnl_today
        self.realized_pnl_today = snapshot.realized_pnl_today
        if snapshot.latest_price is not None:
            self.latest_price = snapshot.latest_price

        logger.info(
            "paper_portfolio_restored",
            symbol=self.position.symbol,
            quote_balance=str(self.quote_balance),
            position_quantity=str(self.position.quantity),
            position_avg_entry_price=str(self.position.avg_entry_price),
            realized_pnl_today=str(self.realized_pnl_today),
            latest_price=str(self.latest_price),
        )

    async def execute(self, request: OrderRequest) -> OrderResult:
        if self.latest_price is None:
            return OrderResult(
                client_order_id=request.client_order_id,
                exchange_order_id=None,
                symbol=request.symbol,
                side=request.side,
                status=OrderStatus.FAILED,
                executed_quantity=Decimal("0"),
                executed_quote_quantity=Decimal("0"),
                raw_response={"error": "latest_price_not_available"},
            )

        if request.side == OrderSide.BUY:
            return self._buy(request)
        return self._sell(request)

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        return PortfolioSnapshot(
            quote_balance=self.quote_balance,
            position_quantity=self.position.quantity,
            position_avg_entry_price=self.position.avg_entry_price,
            realized_pnl_today=self.realized_pnl_today,
            latest_price=self.latest_price,
        )

    def _buy(self, request: OrderRequest) -> OrderResult:
        quote_amount = request.quote_amount or Decimal("0")
        if quote_amount <= 0:
            return self._failed(request, "quote_amount_required")
        if quote_amount > self.quote_balance:
            return self._failed(request, "insufficient_quote_balance")

        price = self.latest_price or Decimal("0")
        quantity = quote_amount / price

        old_cost = self.position.quantity * self.position.avg_entry_price
        new_cost = old_cost + quote_amount
        new_quantity = self.position.quantity + quantity

        self.position.quantity = new_quantity
        self.position.avg_entry_price = new_cost / new_quantity if new_quantity > 0 else Decimal("0")
        self.quote_balance -= quote_amount

        logger.info(
            "paper_order_filled",
            side=request.side.value,
            symbol=request.symbol,
            quantity=str(quantity),
            quote_amount=str(quote_amount),
            price=str(price),
        )
        return OrderResult(
            client_order_id=request.client_order_id,
            exchange_order_id=f"paper_{request.client_order_id}",
            symbol=request.symbol,
            side=request.side,
            status=OrderStatus.FILLED,
            executed_quantity=quantity,
            executed_quote_quantity=quote_amount,
            raw_response={"mode": "paper", "price": str(price)},
        )

    def _sell(self, request: OrderRequest) -> OrderResult:
        if self.position.quantity <= 0:
            return self._failed(request, "no_open_position")

        quantity = request.quantity or self.position.quantity
        if quantity > self.position.quantity:
            return self._failed(request, "sell_quantity_exceeds_position")

        price = self.latest_price or Decimal("0")
        quote_received = quantity * price
        cost_basis = quantity * self.position.avg_entry_price
        pnl = quote_received - cost_basis

        self.position.quantity -= quantity
        if self.position.quantity == 0:
            self.position.avg_entry_price = Decimal("0")

        self.quote_balance += quote_received
        self.realized_pnl_today += pnl
        self.position.realized_pnl += pnl

        logger.info(
            "paper_order_filled",
            side=request.side.value,
            symbol=request.symbol,
            quantity=str(quantity),
            quote_amount=str(quote_received),
            price=str(price),
            pnl=str(pnl),
        )
        return OrderResult(
            client_order_id=request.client_order_id,
            exchange_order_id=f"paper_{request.client_order_id}",
            symbol=request.symbol,
            side=request.side,
            status=OrderStatus.FILLED,
            executed_quantity=quantity,
            executed_quote_quantity=quote_received,
            raw_response={"mode": "paper", "price": str(price), "pnl": str(pnl)},
        )

    def _failed(self, request: OrderRequest, reason: str) -> OrderResult:
        logger.warning("paper_order_failed", reason=reason, symbol=request.symbol, side=request.side.value)
        return OrderResult(
            client_order_id=request.client_order_id,
            exchange_order_id=None,
            symbol=request.symbol,
            side=request.side,
            status=OrderStatus.FAILED,
            executed_quantity=Decimal("0"),
            executed_quote_quantity=Decimal("0"),
            raw_response={"error": reason},
        )
