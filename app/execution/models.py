from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from app.utils.ids import new_id
from app.utils.time import utc_now


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"


class OrderStatus(StrEnum):
    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    REJECTED = "rejected"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType = OrderType.MARKET
    quantity: Decimal | None = None
    quote_amount: Decimal | None = None
    client_order_id: str = field(default_factory=lambda: new_id("order"))
    reason: str = ""


@dataclass(slots=True, frozen=True)
class OrderResult:
    client_order_id: str
    exchange_order_id: str | None
    symbol: str
    side: OrderSide
    status: OrderStatus
    executed_quantity: Decimal
    executed_quote_quantity: Decimal
    raw_response: dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @property
    def is_open(self) -> bool:
        return self.quantity > 0

    def market_value(self, latest_price: Decimal) -> Decimal:
        return self.quantity * latest_price


@dataclass(slots=True, frozen=True)
class PortfolioSnapshot:
    quote_balance: Decimal
    position_quantity: Decimal
    position_avg_entry_price: Decimal
    realized_pnl_today: Decimal
    latest_price: Decimal | None

    @property
    def has_open_position(self) -> bool:
        return self.position_quantity > 0

    @property
    def position_value(self) -> Decimal:
        if self.latest_price is None:
            return Decimal("0")
        return self.position_quantity * self.latest_price
