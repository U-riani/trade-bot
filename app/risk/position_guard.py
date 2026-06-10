from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.execution.models import PortfolioSnapshot
from app.strategy.models import SignalSide, TradeSignal
from app.utils.time import utc_now


@dataclass(slots=True, frozen=True)
class PositionExitLevels:
    stop_loss_price: Decimal
    take_profit_price: Decimal


def calculate_exit_levels(
    *,
    avg_entry_price: Decimal,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
) -> PositionExitLevels:
    if avg_entry_price <= 0:
        raise ValueError("avg_entry_price must be positive")
    if stop_loss_pct <= 0:
        raise ValueError("stop_loss_pct must be positive")
    if take_profit_pct <= 0:
        raise ValueError("take_profit_pct must be positive")

    hundred = Decimal("100")
    stop_loss_price = avg_entry_price * (Decimal("1") - (stop_loss_pct / hundred))
    take_profit_price = avg_entry_price * (Decimal("1") + (take_profit_pct / hundred))
    return PositionExitLevels(
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
    )


def build_position_exit_signal(
    *,
    portfolio: PortfolioSnapshot,
    symbol: str,
    stop_loss_pct: Decimal,
    take_profit_pct: Decimal,
) -> TradeSignal | None:
    """Create a protective SELL signal when an open position hits exit levels.

    This guard is intentionally separate from the EMA/RSI strategy. Strategy entries
    can be slow and selective; protective exits must be checked on every fresh candle.
    """
    if not portfolio.has_open_position:
        return None
    if portfolio.latest_price is None:
        return None
    if portfolio.position_avg_entry_price <= 0:
        return None

    levels = calculate_exit_levels(
        avg_entry_price=portfolio.position_avg_entry_price,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
    )

    latest_price = portfolio.latest_price

    if latest_price <= levels.stop_loss_price:
        return TradeSignal(
            strategy_name="position_guard",
            symbol=symbol,
            side=SignalSide.SELL,
            confidence=1.0,
            reason=(
                "stop_loss_triggered: "
                f"latest_price={latest_price}, "
                f"avg_entry_price={portfolio.position_avg_entry_price}, "
                f"stop_loss_price={levels.stop_loss_price}, "
                f"stop_loss_pct={stop_loss_pct}"
            ),
            created_at=utc_now(),
            is_protective_exit=True,
        )

    if latest_price >= levels.take_profit_price:
        return TradeSignal(
            strategy_name="position_guard",
            symbol=symbol,
            side=SignalSide.SELL,
            confidence=1.0,
            reason=(
                "take_profit_triggered: "
                f"latest_price={latest_price}, "
                f"avg_entry_price={portfolio.position_avg_entry_price}, "
                f"take_profit_price={levels.take_profit_price}, "
                f"take_profit_pct={take_profit_pct}"
            ),
            created_at=utc_now(),
            is_protective_exit=True,
        )

    return None
