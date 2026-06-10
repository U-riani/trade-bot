from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.market.models import Candle, TradeTick


@dataclass(slots=True)
class SimpleCandleBuilder:
    """Small candle builder for future trade-tick aggregation.

    The first MVP uses Binance kline streams, so this is mostly here to preserve the
    architecture for future upgrades where we may build candles from trade ticks.
    """

    exchange: str
    symbol: str
    timeframe: str = "1m"
    current_open_time: datetime | None = None
    current_close_time: datetime | None = None
    current_open: float | None = None
    current_high: float | None = None
    current_low: float | None = None
    current_close: float | None = None
    current_volume: float = 0.0

    def add_trade(self, trade: TradeTick) -> None:
        price = trade.price
        quantity = trade.quantity

        if self.current_open is None:
            self.current_open = price
            self.current_high = price
            self.current_low = price

        self.current_high = max(self.current_high or price, price)
        self.current_low = min(self.current_low or price, price)
        self.current_close = price
        self.current_volume += quantity

    def close_current(self, open_time: datetime, close_time: datetime) -> Candle | None:
        if self.current_open is None or self.current_close is None:
            return None

        candle = Candle(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            open_time=open_time,
            close_time=close_time,
            open=self.current_open,
            high=self.current_high or self.current_open,
            low=self.current_low or self.current_open,
            close=self.current_close,
            volume=self.current_volume,
            is_closed=True,
        )
        self.reset()
        return candle

    def reset(self) -> None:
        self.current_open_time = None
        self.current_close_time = None
        self.current_open = None
        self.current_high = None
        self.current_low = None
        self.current_close = None
        self.current_volume = 0.0
