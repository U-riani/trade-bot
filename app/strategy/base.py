from __future__ import annotations

from abc import ABC, abstractmethod

from app.market.state import MarketState
from app.strategy.models import TradeSignal


class Strategy(ABC):
    name: str

    @abstractmethod
    def on_market_state(self, market_state: MarketState) -> TradeSignal:
        raise NotImplementedError
