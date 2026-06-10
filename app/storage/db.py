from __future__ import annotations

from pathlib import Path

import asyncpg


class Database:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    def require_pool(self) -> asyncpg.Pool:
        if self.pool is None:
            raise RuntimeError("Database pool is not connected")
        return self.pool

    async def ping(self) -> dict[str, str]:
        pool = self.require_pool()
        row = await pool.fetchrow(
            """
            SELECT
                current_database() AS database_name,
                current_user AS user_name,
                version() AS version
            """
        )
        if row is None:
            raise RuntimeError("Database ping returned no result")
        return {
            "database_name": row["database_name"],
            "user_name": row["user_name"],
            "version": row["version"],
        }

    async def apply_migration_file(self, migration_path: str | Path) -> None:
        sql = Path(migration_path).read_text(encoding="utf-8")
        pool = self.require_pool()
        async with pool.acquire() as connection:
            await connection.execute(sql)
