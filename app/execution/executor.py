from __future__ import annotations

from abc import ABC, abstractmethod

from app.execution.models import OrderRequest, OrderResult, PortfolioSnapshot


class OrderExecutor(ABC):
    @abstractmethod
    async def execute(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    @abstractmethod
    def portfolio_snapshot(self) -> PortfolioSnapshot:
        raise NotImplementedError
