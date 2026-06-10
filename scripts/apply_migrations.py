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
    timescale_migration_path = migrations_dir / "002_timescale_optional.sql"

    db = Database(settings.database_url)
    try:
        await db.connect()
        await db.apply_migration_file(base_migration_path)
        logger.info("db_migration_applied", migration=str(base_migration_path))

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
