from __future__ import annotations

from app.exchange.binance_rest import BinanceRestClient
from app.execution.executor import OrderExecutor
from app.execution.models import OrderRequest, OrderResult, PortfolioSnapshot


class TestnetExecutor(OrderExecutor):
    """Binance Spot Testnet executor.

    This uses real API calls against testnet. Portfolio tracking is intentionally
    still local/minimal in this MVP. The next upgrade should add account balance
    synchronization and user data stream support.
    """

    def __init__(self, rest_client: BinanceRestClient, fallback_snapshot: PortfolioSnapshot) -> None:
        self.rest_client = rest_client
        self._snapshot = fallback_snapshot

    async def execute(self, request: OrderRequest) -> OrderResult:
        return await self.rest_client.place_market_order(request)

    def portfolio_snapshot(self) -> PortfolioSnapshot:
        return self._snapshot
