from __future__ import annotations

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Any

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database

logger = get_logger(__name__)

TABLES = ["candles", "signals", "risk_decisions", "orders", "positions", "bot_events"]


def _decimal_or_zero(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.database_enabled:
        logger.error("state_check_failed", reason="DATABASE_ENABLED=false")
        return

    db = Database(settings.database_url)
    await db.connect()
    try:
        pool = db.require_pool()
        ping = await db.ping()
        logger.info("db_connection_ok", database=ping["database_name"], user=ping["user_name"])

        for table in TABLES:
            count = await pool.fetchval(f"SELECT COUNT(*) FROM {table}")
            logger.info("db_table_count", table=table, rows=int(count or 0))

        latest_candle = await pool.fetchrow(
            """
            SELECT symbol, timeframe, close_time, close, volume
            FROM candles
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY close_time DESC
            LIMIT 1
            """,
            settings.normalized_symbol,
            settings.timeframe,
        )
        if latest_candle is not None:
            logger.info(
                "latest_candle",
                symbol=latest_candle["symbol"],
                timeframe=latest_candle["timeframe"],
                close_time=latest_candle["close_time"].isoformat(),
                close=str(latest_candle["close"]),
                volume=str(latest_candle["volume"]),
            )
        else:
            logger.warning("latest_candle_missing", symbol=settings.normalized_symbol, timeframe=settings.timeframe)

        latest_signal = await pool.fetchrow(
            """
            SELECT strategy_name, symbol, side, confidence, reason, created_at
            FROM signals
            WHERE symbol = $1
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            settings.normalized_symbol,
        )
        if latest_signal is not None:
            logger.info(
                "latest_signal",
                strategy=latest_signal["strategy_name"],
                symbol=latest_signal["symbol"],
                side=latest_signal["side"],
                confidence=str(latest_signal["confidence"]),
                reason=latest_signal["reason"],
                created_at=latest_signal["created_at"].isoformat(),
            )

        latest_order = await pool.fetchrow(
            """
            SELECT client_order_id, exchange_order_id, symbol, side, status,
                   executed_quantity, executed_quote_quantity, created_at
            FROM orders
            WHERE symbol = $1
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            settings.normalized_symbol,
        )
        if latest_order is not None:
            logger.info(
                "latest_order",
                client_order_id=latest_order["client_order_id"],
                exchange_order_id=latest_order["exchange_order_id"],
                symbol=latest_order["symbol"],
                side=latest_order["side"],
                status=latest_order["status"],
                executed_quantity=str(latest_order["executed_quantity"]),
                executed_quote_quantity=str(latest_order["executed_quote_quantity"]),
                created_at=latest_order["created_at"].isoformat(),
            )
        else:
            logger.info("latest_order_missing", symbol=settings.normalized_symbol)

        latest_position = await pool.fetchrow(
            """
            SELECT symbol, quantity, avg_entry_price, realized_pnl, quote_balance, updated_at
            FROM positions
            WHERE symbol = $1
            ORDER BY updated_at DESC, id DESC
            LIMIT 1
            """,
            settings.normalized_symbol,
        )
        if latest_position is not None:
            quantity = _decimal_or_zero(latest_position["quantity"])
            avg_entry_price = _decimal_or_zero(latest_position["avg_entry_price"])
            quote_balance = _decimal_or_zero(latest_position["quote_balance"])
            realized_pnl = _decimal_or_zero(latest_position["realized_pnl"])
            latest_price = _decimal_or_zero(latest_candle["close"]) if latest_candle is not None else Decimal("0")
            market_value = quantity * latest_price
            cost_basis = quantity * avg_entry_price
            unrealized_pnl = market_value - cost_basis
            total_equity = quote_balance + market_value

            logger.info(
                "paper_position_state",
                symbol=latest_position["symbol"],
                has_open_position=quantity > 0,
                quantity=str(quantity),
                avg_entry_price=str(avg_entry_price),
                latest_price=str(latest_price),
                quote_balance=str(quote_balance),
                market_value=str(market_value),
                realized_pnl=str(realized_pnl),
                unrealized_pnl=str(unrealized_pnl),
                total_equity=str(total_equity),
                updated_at=latest_position["updated_at"].isoformat(),
            )
        else:
            logger.info("paper_position_missing", symbol=settings.normalized_symbol)
    finally:
        await db.close()
        logger.info("db_connection_closed")


if __name__ == "__main__":
    asyncio.run(main())
