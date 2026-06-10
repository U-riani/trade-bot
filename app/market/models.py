from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MarketEventType(StrEnum):
    CANDLE_CLOSED = "candle_closed"
    TRADE = "trade"


@dataclass(slots=True, frozen=True)
class Candle:
    exchange: str
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


@dataclass(slots=True, frozen=True)
class TradeTick:
    exchange: str
    symbol: str
    event_time: datetime
    price: float
    quantity: float


@dataclass(slots=True, frozen=True)
class MarketEvent:
    type: MarketEventType
    candle: Candle | None = None
    trade: TradeTick | None = None
