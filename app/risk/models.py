from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.execution.models import OrderRequest
from app.utils.time import utc_now


class RiskDecisionStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class RiskRejectReason(StrEnum):
    KILL_SWITCH_ACTIVE = "kill_switch_active"
    HOLD_SIGNAL = "hold_signal"
    INVALID_SIGNAL_SIDE = "invalid_signal_side"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    COOLDOWN = "cooldown"
    MAX_TRADES_PER_HOUR = "max_trades_per_hour"
    POSITION_ALREADY_OPEN = "position_already_open"
    NO_OPEN_POSITION = "no_open_position"
    MAX_ORDER_SIZE = "max_order_size"
    MAX_POSITION_SIZE = "max_position_size"
    LATEST_PRICE_MISSING = "latest_price_missing"
    INSUFFICIENT_BALANCE = "insufficient_balance"


@dataclass(slots=True, frozen=True)
class RiskDecision:
    status: RiskDecisionStatus
    reason: str
    created_at: datetime
    order_request: OrderRequest | None = None

    @property
    def approved(self) -> bool:
        return self.status == RiskDecisionStatus.APPROVED

    @classmethod
    def approve(cls, reason: str, order_request: OrderRequest) -> RiskDecision:
        return cls(
            status=RiskDecisionStatus.APPROVED,
            reason=reason,
            created_at=utc_now(),
            order_request=order_request,
        )

    @classmethod
    def reject(cls, reason: RiskRejectReason | str) -> RiskDecision:
        return cls(
            status=RiskDecisionStatus.REJECTED,
            reason=reason.value if isinstance(reason, RiskRejectReason) else reason,
            created_at=utc_now(),
            order_request=None,
        )


@dataclass(slots=True)
class RiskConfig:
    max_order_usdt: Decimal
    max_position_usdt: Decimal
    max_daily_loss_usdt: Decimal
    max_trades_per_hour: int
    cooldown_seconds: int
    allow_only_one_open_position: bool = True
