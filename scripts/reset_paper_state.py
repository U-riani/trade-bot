from __future__ import annotations

import argparse
import asyncio
from decimal import Decimal

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset local paper trading portfolio state for one symbol.")
    parser.add_argument("--symbol", default=None, help="Trading symbol to reset. Defaults to SYMBOL from .env.")
    parser.add_argument(
        "--quote-balance",
        default=None,
        help="Paper quote balance after reset. Defaults to INITIAL_QUOTE_BALANCE from .env.",
    )
    parser.add_argument(
        "--reset-realized-pnl",
        action="store_true",
        help="Set realized_pnl to 0. If omitted, existing realized_pnl is preserved when a row exists.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)

    if not settings.database_enabled:
        logger.error("paper_state_reset_failed", reason="DATABASE_ENABLED=false")
        return

    symbol = (args.symbol or settings.normalized_symbol).upper()
    quote_balance = Decimal(str(args.quote_balance)) if args.quote_balance is not None else settings.initial_quote_balance

    db = Database(settings.database_url)
    await db.connect()
    try:
        pool = db.require_pool()
        existing_realized_pnl = await pool.fetchval(
            "SELECT realized_pnl FROM positions WHERE symbol = $1 ORDER BY updated_at DESC LIMIT 1",
            symbol,
        )
        realized_pnl = Decimal("0") if args.reset_realized_pnl or existing_realized_pnl is None else Decimal(str(existing_realized_pnl))

        await pool.execute(
            """
            INSERT INTO positions(symbol, quantity, avg_entry_price, realized_pnl, quote_balance, updated_at)
            VALUES($1, 0, 0, $2, $3, NOW())
            ON CONFLICT (symbol) DO UPDATE SET
                quantity = 0,
                avg_entry_price = 0,
                realized_pnl = EXCLUDED.realized_pnl,
                quote_balance = EXCLUDED.quote_balance,
                updated_at = NOW()
            """,
            symbol,
            realized_pnl,
            quote_balance,
        )
        logger.info(
            "paper_state_reset_done",
            symbol=symbol,
            quantity="0",
            avg_entry_price="0",
            realized_pnl=str(realized_pnl),
            quote_balance=str(quote_balance),
        )
    finally:
        await db.close()
        logger.info("db_connection_closed")


if __name__ == "__main__":
    asyncio.run(main())
