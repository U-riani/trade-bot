from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from app.market.models import Candle, TradeTick
from app.utils.time import utc_now


@dataclass(slots=True)
class MarketState:
    symbol: str
    max_candles: int = 500
    candles: deque[Candle] = field(default_factory=deque)
    latest_price: float | None = None
    last_market_event_at: datetime | None = None

    def __post_init__(self) -> None:
        self.candles = deque(self.candles, maxlen=self.max_candles)

    def add_candle(self, candle: Candle) -> bool:
        """Add a new live candle.

        Returns False when the candle is duplicate/out-of-order. This protects
        the in-memory indicator window after startup warm-up from DB.
        """
        if self.candles and candle.open_time <= self.candles[-1].open_time:
            return False

        self.candles.append(candle)
        self.latest_price = candle.close
        self.last_market_event_at = utc_now()
        return True

    def load_historical_candles(self, candles: list[Candle]) -> int:
        """Warm up the in-memory state from trusted historical candles.

        This method assumes the caller already validated freshness and continuity.
        It sets last_market_event_at to the latest candle close time so health checks
        can still detect stale data naturally if live data stops arriving.
        """
        if not candles:
            return 0

        self.candles.clear()
        for candle in candles[-self.max_candles :]:
            self.candles.append(candle)

        latest = self.candles[-1]
        self.latest_price = latest.close
        self.last_market_event_at = latest.close_time
        return len(candles[-self.max_candles :])

    def add_trade(self, trade: TradeTick) -> None:
        self.latest_price = trade.price
        self.last_market_event_at = utc_now()

    def get_closes(self) -> list[float]:
        return [candle.close for candle in self.candles]

    def is_stale(self, stale_seconds: int) -> bool:
        if self.last_market_event_at is None:
            return True
        delta = utc_now() - self.last_market_event_at
        return delta.total_seconds() > stale_seconds
