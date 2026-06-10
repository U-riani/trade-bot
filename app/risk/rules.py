from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.utils.time import utc_now


@dataclass(slots=True)
class TradeLimiter:
    max_trades_per_hour: int
    trade_timestamps: deque[datetime] = field(default_factory=deque)

    def register_trade(self) -> None:
        self.trade_timestamps.append(utc_now())
        self._prune()

    def can_trade(self) -> bool:
        self._prune()
        return len(self.trade_timestamps) < self.max_trades_per_hour

    def _prune(self) -> None:
        threshold = utc_now() - timedelta(hours=1)
        while self.trade_timestamps and self.trade_timestamps[0] < threshold:
            self.trade_timestamps.popleft()
