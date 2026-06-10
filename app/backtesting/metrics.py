from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(slots=True, frozen=True)
class BacktestTrade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    quote_amount: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    pnl: Decimal
    entry_reason: str
    exit_reason: str

    @property
    def total_fees(self) -> Decimal:
        return self.entry_fee + self.exit_fee

    @property
    def return_pct(self) -> Decimal:
        capital_used = self.quote_amount + self.entry_fee
        if capital_used <= 0:
            return Decimal("0")
        return (self.pnl / capital_used) * Decimal("100")


@dataclass(slots=True, frozen=True)
class BacktestMetrics:
    candles_processed: int
    executed_orders: int
    round_trips: int
    winning_trades: int
    losing_trades: int
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_fees: Decimal
    max_drawdown: Decimal
    initial_equity: Decimal
    final_equity: Decimal
    open_position_quantity: Decimal
    open_position_avg_entry_price: Decimal
    last_price: Decimal | None

    @property
    def win_rate(self) -> float:
        if self.round_trips == 0:
            return 0.0
        return self.winning_trades / self.round_trips

    @property
    def return_pct(self) -> Decimal:
        if self.initial_equity <= 0:
            return Decimal("0")
        return ((self.final_equity - self.initial_equity) / self.initial_equity) * Decimal("100")

    @property
    def has_open_position(self) -> bool:
        return self.open_position_quantity > 0


@dataclass(slots=True, frozen=True)
class BacktestResult:
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
