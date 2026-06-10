from __future__ import annotations

import asyncio

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database
from app.storage.repositories import TradingRepository

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    db = Database(settings.database_url)
    try:
        await db.connect()
        ping = await db.ping()
        logger.info(
            "db_connection_ok",
            database=ping["database_name"],
            user=ping["user_name"],
        )

        repository = TradingRepository(db)
        for table_name in ["candles", "signals", "risk_decisions", "orders", "positions", "bot_events"]:
            try:
                count = await repository.count_table(table_name)
                logger.info("db_table_ok", table=table_name, rows=count)
            except Exception as exc:
                logger.error("db_table_check_failed", table=table_name, error=str(exc))
    finally:
        await db.close()
        logger.info("db_connection_closed")


if __name__ == "__main__":
    asyncio.run(main())
