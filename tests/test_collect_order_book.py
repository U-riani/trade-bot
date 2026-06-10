from __future__ import annotations

import asyncio

from scripts.collect_order_book_features import _collect_once


class FakeClient:
    """Stand-in for BinanceRestClient so the collector logic needs no network."""

    def __init__(self, depth: dict) -> None:
        self._depth = depth
        self.calls = 0

    async def get_order_book(self, *, symbol: str, limit: int) -> dict:
        self.calls += 1
        return self._depth


def _full_depth() -> dict:
    return {
        "bids": [[str(100.0 - i * 0.1), "2.0"] for i in range(20)],
        "asks": [[str(101.0 + i * 0.1), "1.0"] for i in range(20)],
    }


def test_collect_once_dry_run_no_db() -> None:
    # repository=None models dry-run / DB-unavailable: it must still succeed
    # and must not attempt any storage.
    client = FakeClient(_full_depth())
    ok = asyncio.run(
        _collect_once(client=client, repository=None, symbol="BTCUSDT", exchange="binance_spot", limit=100)
    )
    assert ok is True
    assert client.calls == 1


def test_collect_once_empty_book_returns_false() -> None:
    client = FakeClient({"bids": [], "asks": []})
    ok = asyncio.run(
        _collect_once(client=client, repository=None, symbol="BTCUSDT", exchange="binance_spot", limit=100)
    )
    assert ok is False
