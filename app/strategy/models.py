from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from app.utils.time import utc_now


class SignalSide(StrEnum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(slots=True, frozen=True)
class TradeSignal:
    strategy_name: str
    symbol: str
    side: SignalSide
    confidence: float
    reason: str
    created_at: datetime
    suggested_quote_amount: Decimal | None = None
    is_protective_exit: bool = False

    @classmethod
    def hold(cls, strategy_name: str, symbol: str, reason: str) -> TradeSignal:
        return cls(
            strategy_name=strategy_name,
            symbol=symbol,
            side=SignalSide.HOLD,
            confidence=0.0,
            reason=reason,
            created_at=utc_now(),
        )
