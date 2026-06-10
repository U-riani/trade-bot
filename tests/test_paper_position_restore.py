from decimal import Decimal

import pytest

from app.config.settings import Settings, TradeMode
from app.execution.models import PortfolioSnapshot
from app.execution.paper_executor import PaperExecutor
from app.main import restore_paper_position_from_db
from app.market.state import MarketState


class FakeRepository:
    def __init__(self, snapshot: PortfolioSnapshot | None) -> None:
        self.snapshot = snapshot
        self.requested_symbol: str | None = None
        self.fallback_quote_balance: Decimal | None = None
        self.latest_price: Decimal | None = None

    async def load_position_snapshot(
        self,
        *,
        symbol: str,
        fallback_quote_balance: Decimal,
        latest_price: Decimal | None,
    ) -> PortfolioSnapshot | None:
        self.requested_symbol = symbol
        self.fallback_quote_balance = fallback_quote_balance
        self.latest_price = latest_price
        return self.snapshot


@pytest.mark.asyncio
async def test_paper_position_restore_loads_saved_snapshot() -> None:
    settings = Settings(
        trade_mode=TradeMode.PAPER,
        load_paper_position_on_start=True,
        initial_quote_balance=Decimal("1000"),
    )
    state = MarketState(symbol="BTCUSDT")
    state.latest_price = 105
    executor = PaperExecutor(initial_quote_balance=Decimal("1000"), symbol="BTCUSDT")
    repository = FakeRepository(
        PortfolioSnapshot(
            quote_balance=Decimal("990"),
            position_quantity=Decimal("0.1"),
            position_avg_entry_price=Decimal("100"),
            realized_pnl_today=Decimal("0"),
            latest_price=Decimal("105"),
        )
    )

    await restore_paper_position_from_db(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        executor=executor,
        market_state=state,
    )

    snapshot = executor.portfolio_snapshot()
    assert snapshot.quote_balance == Decimal("990")
    assert snapshot.position_quantity == Decimal("0.1")
    assert snapshot.position_avg_entry_price == Decimal("100")
    assert snapshot.latest_price == Decimal("105")
    assert snapshot.has_open_position
    assert repository.requested_symbol == "BTCUSDT"
    assert repository.latest_price == Decimal("105")


@pytest.mark.asyncio
async def test_paper_position_restore_skips_when_disabled() -> None:
    settings = Settings(
        trade_mode=TradeMode.PAPER,
        load_paper_position_on_start=False,
        initial_quote_balance=Decimal("1000"),
    )
    state = MarketState(symbol="BTCUSDT")
    executor = PaperExecutor(initial_quote_balance=Decimal("1000"), symbol="BTCUSDT")
    repository = FakeRepository(
        PortfolioSnapshot(
            quote_balance=Decimal("990"),
            position_quantity=Decimal("0.1"),
            position_avg_entry_price=Decimal("100"),
            realized_pnl_today=Decimal("0"),
            latest_price=Decimal("105"),
        )
    )

    await restore_paper_position_from_db(
        settings=settings,
        repository=repository,  # type: ignore[arg-type]
        executor=executor,
        market_state=state,
    )

    snapshot = executor.portfolio_snapshot()
    assert snapshot.quote_balance == Decimal("1000")
    assert snapshot.position_quantity == Decimal("0")
    assert repository.requested_symbol is None
