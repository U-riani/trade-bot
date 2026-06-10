"""Integration test for market_features storage.

Requires a reachable PostgreSQL (the project's research DB). It is skipped
automatically when the database cannot be connected, so the suite stays green
on machines without a database. It uses a dedicated test exchange key and
cleans up after itself, so it never touches real research data.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config.settings import get_settings
from app.market.features import MarketFeatures
from app.storage.db import Database
from app.storage.repositories import TradingRepository

TEST_EXCHANGE = "test_v22_features"
START = datetime(2026, 1, 1, tzinfo=UTC)
MIGRATION = Path(__file__).resolve().parents[1] / "app" / "storage" / "migrations" / "003_market_features.sql"


def _rows() -> list[MarketFeatures]:
    rows = []
    for i in range(3):
        close_time = START + timedelta(minutes=5 * (i + 1))
        rows.append(
            MarketFeatures(
                exchange=TEST_EXCHANGE,
                symbol="TESTUSDT",
                timeframe="5m",
                open_time=close_time - timedelta(minutes=5),
                close_time=close_time,
                close_price=100.0 + i,
                volume=10.0 + i,
                quote_volume=1000.0 + i,
                taker_buy_base_volume=6.0,
                taker_buy_quote_volume=600.0,
                taker_buy_ratio=0.5,
            )
        )
    return rows


async def _roundtrip():
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()
    try:
        repo = TradingRepository(db)
        pool = db.require_pool()
        await db.apply_migration_file(MIGRATION)
        await pool.execute("DELETE FROM market_features WHERE exchange = $1", TEST_EXCHANGE)

        rows = _rows()
        inserted = await repo.insert_market_features(rows)
        inserted_again = await repo.insert_market_features(rows)  # duplicates
        loaded = await repo.load_market_features(
            exchange=TEST_EXCHANGE, symbol="TESTUSDT", timeframe="5m", limit=100
        )
        count = await repo.count_market_features(exchange=TEST_EXCHANGE, symbol="TESTUSDT", timeframe="5m")

        await pool.execute("DELETE FROM market_features WHERE exchange = $1", TEST_EXCHANGE)
        return inserted, inserted_again, loaded, count
    finally:
        await db.close()


def test_market_features_storage_roundtrip() -> None:
    try:
        inserted, inserted_again, loaded, count = asyncio.run(_roundtrip())
    except Exception as exc:  # noqa: BLE001 - skip when DB is unreachable
        pytest.skip(f"database not available: {exc}")

    assert inserted == 3
    assert inserted_again == 0  # ON CONFLICT DO NOTHING
    assert count == 3
    assert len(loaded) == 3
    # load_market_features returns ascending by close_time.
    assert loaded[0].close_time < loaded[-1].close_time
    assert loaded[0].taker_buy_ratio == 0.5
    assert loaded[0].order_book_imbalance is None
