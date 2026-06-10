from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.market.models import MarketEvent


class MarketDataStream(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[MarketEvent]:
        raise NotImplementedError
