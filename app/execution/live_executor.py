from __future__ import annotations

from app.execution.executor import OrderExecutor
from app.execution.models import OrderRequest, OrderResult, PortfolioSnapshot


class LiveExecutor(OrderExecutor):
    """Live execution guard.

    Live trading is intentionally not implemented in the first MVP. This class
    exists so the project structure is ready, but real implementation must be
    added only after paper/testnet validation, order status checks, and monitoring.
    """

    async def execute(self, request: OrderRequest) -> OrderResult:
        raise RuntimeError("Live execution is intentionally disabled in this MVP")

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        raise RuntimeError("Live execution is intentionally disabled in this MVP")
