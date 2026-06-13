from __future__ import annotations

import asyncio
from pathlib import Path

from app.config.logging import configure_logging, get_logger
from app.config.settings import get_settings
from app.storage.db import Database

logger = get_logger(__name__)


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    migrations_dir = Path(__file__).resolve().parents[1] / "app" / "storage" / "migrations"
    base_migration_path = migrations_dir / "001_init.sql"
    market_features_migration_path = migrations_dir / "003_market_features.sql"
    order_book_migration_path = migrations_dir / "004_order_book_snapshots.sql"
    collector_runs_migration_path = migrations_dir / "005_order_book_collector_runs.sql"
    trade_pressure_migration_path = migrations_dir / "006_trade_pressure_features.sql"
    timescale_migration_path = migrations_dir / "002_timescale_optional.sql"

    db = Database(settings.database_url)
    try:
        await db.connect()
        await db.apply_migration_file(base_migration_path)
        logger.info("db_migration_applied", migration=str(base_migration_path))

        await db.apply_migration_file(market_features_migration_path)
        logger.info("db_migration_applied", migration=str(market_features_migration_path))

        await db.apply_migration_file(order_book_migration_path)
        logger.info("db_migration_applied", migration=str(order_book_migration_path))

        await db.apply_migration_file(collector_runs_migration_path)
        logger.info("db_migration_applied", migration=str(collector_runs_migration_path))

        await db.apply_migration_file(trade_pressure_migration_path)
        logger.info("db_migration_applied", migration=str(trade_pressure_migration_path))

        if settings.database_use_timescaledb:
            await db.apply_migration_file(timescale_migration_path)
            logger.info("db_migration_applied", migration=str(timescale_migration_path))
        else:
            logger.info("db_timescaledb_skipped", reason="DATABASE_USE_TIMESCALEDB=false")
    finally:
        await db.close()
        logger.info("db_connection_closed")


if __name__ == "__main__":
    asyncio.run(main())
