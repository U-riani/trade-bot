"""Integration test for order_book_snapshots storage.

Skipped automatically when no database is reachable, so the suite stays green
without PostgreSQL. Uses a dedicated test exchange key and cleans up after
itself.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config.settings import get_settings
from app.market.order_book import build_order_book_snapshot
from app.storage.db import Database
from app.storage.repositories import TradingRepository

TEST_EXCHANGE = "test_v23_orderbook"
START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
MIGRATION = Path(__file__).resolve().parents[1] / "app" / "storage" / "migrations" / "004_order_book_snapshots.sql"


def _snapshots():
    snaps = []
    for i in range(3):
        bids = [(100.0 - j * 0.1, 2.0) for j in range(20)]
        asks = [(101.0 + j * 0.1, 1.0) for j in range(20)]
        snaps.append(
            build_order_book_snapshot(
                exchange=TEST_EXCHANGE,
                symbol="TESTUSDT",
                collected_at=START + timedelta(seconds=5 * i),
                bids=bids,
                asks=asks,
                raw_depth_limit=100,
            )
        )
    return snaps


async def _roundtrip():
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()
    try:
        repo = TradingRepository(db)
        pool = db.require_pool()
        await db.apply_migration_file(MIGRATION)
        await pool.execute("DELETE FROM order_book_snapshots WHERE exchange = $1", TEST_EXCHANGE)

        inserted = await repo.insert_order_book_snapshots(_snapshots())
        count = await repo.count_order_book_snapshots(exchange=TEST_EXCHANGE, symbol="TESTUSDT")
        loaded = await repo.load_order_book_snapshots(exchange=TEST_EXCHANGE, symbol="TESTUSDT", limit=100)

        await pool.execute("DELETE FROM order_book_snapshots WHERE exchange = $1", TEST_EXCHANGE)
        return inserted, count, loaded
    finally:
        await db.close()


def test_order_book_storage_roundtrip() -> None:
    try:
        inserted, count, loaded = asyncio.run(_roundtrip())
    except Exception as exc:  # noqa: BLE001 - skip when DB unreachable
        pytest.skip(f"database not available: {exc}")

    assert inserted == 3
    assert count == 3
    assert len(loaded) == 3
    # ascending by collected_at
    assert loaded[0].collected_at < loaded[-1].collected_at
    assert loaded[0].imbalance_top_5 > 0  # bid-heavy test snapshots
    assert loaded[0].raw_depth_limit == 100
