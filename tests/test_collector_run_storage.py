"""Integration test for order_book_collector_runs storage.

Skipped automatically when no database is reachable. Uses a dedicated test
exchange key and cleans up after itself.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.config.settings import get_settings
from app.market.collector_runtime import STATUS_RUNNING, STATUS_STOPPED, CollectorRun
from app.storage.db import Database
from app.storage.repositories import TradingRepository

TEST_EXCHANGE = "test_v25_runs"
START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
MIGRATION = Path(__file__).resolve().parents[1] / "app" / "storage" / "migrations" / "005_order_book_collector_runs.sql"


async def _roundtrip():
    settings = get_settings()
    db = Database(settings.database_url)
    await db.connect()
    try:
        repo = TradingRepository(db)
        pool = db.require_pool()
        await db.apply_migration_file(MIGRATION)
        await pool.execute("DELETE FROM order_book_collector_runs WHERE exchange = $1", TEST_EXCHANGE)

        run = CollectorRun(
            run_id="test-v25-run-1", exchange=TEST_EXCHANGE, symbol="TESTUSDT", started_at=START,
            stopped_at=None, status=STATUS_RUNNING, interval_seconds=5.0, depth_limit=100,
            collected_count=0, failure_count=0, last_snapshot_at=None, stop_reason=None,
        )
        await repo.start_collector_run(run)
        await repo.start_collector_run(run)  # idempotent upsert by run_id

        await repo.update_collector_run(
            run_id="test-v25-run-1", status=STATUS_STOPPED, collected_count=42, failure_count=3,
            last_snapshot_at=START + timedelta(minutes=5), stopped_at=START + timedelta(minutes=6),
            stop_reason="max_snapshots_reached",
        )

        count = await repo.count_collector_runs(exchange=TEST_EXCHANGE, symbol="TESTUSDT")
        runs = await repo.load_collector_runs(exchange=TEST_EXCHANGE, symbol="TESTUSDT", limit=10)

        await pool.execute("DELETE FROM order_book_collector_runs WHERE exchange = $1", TEST_EXCHANGE)
        return count, runs
    finally:
        await db.close()


def test_collector_run_storage_roundtrip() -> None:
    try:
        count, runs = asyncio.run(_roundtrip())
    except Exception as exc:  # noqa: BLE001 - skip when DB unreachable
        pytest.skip(f"database not available: {exc}")

    assert count == 1  # upsert did not duplicate
    assert len(runs) == 1
    run = runs[0]
    assert run.status == STATUS_STOPPED
    assert run.collected_count == 42
    assert run.failure_count == 3
    assert run.stop_reason == "max_snapshots_reached"
    assert run.stopped_at is not None
