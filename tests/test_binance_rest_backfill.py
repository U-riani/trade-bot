from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.exchange.binance_rest import BinanceRestClient
from app.utils.time import utc_now


def kline(open_time: datetime, close: str) -> list[object]:
    open_ms = int(open_time.timestamp() * 1000)
    close_ms = int((open_time + timedelta(minutes=1) - timedelta(milliseconds=1)).timestamp() * 1000)
    return [open_ms, close, close, close, close, "1.5", close_ms]


@pytest.mark.asyncio
async def test_get_closed_candles_filters_open_candle(monkeypatch):
    now = utc_now()
    closed_open = int((now - timedelta(minutes=2)).timestamp() * 1000)
    closed_close = int((now - timedelta(minutes=1)).timestamp() * 1000)
    open_open = int(now.timestamp() * 1000)
    open_close = int((now + timedelta(minutes=1)).timestamp() * 1000)

    async def fake_get_klines(
        *,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ):
        return [
            [closed_open, "100", "110", "90", "105", "1.5", closed_close],
            [open_open, "105", "106", "104", "105.5", "0.3", open_close],
        ]

    client = BinanceRestClient(testnet=True)
    monkeypatch.setattr(client, "get_klines", fake_get_klines)

    try:
        candles = await client.get_closed_candles(symbol="BTCUSDT", timeframe="1m", limit=10)
    finally:
        await client.close()

    assert len(candles) == 1
    assert candles[0].symbol == "BTCUSDT"
    assert candles[0].close == 105.0
    assert candles[0].volume == 1.5


@pytest.mark.asyncio
async def test_get_historical_closed_candles_pages_backwards(monkeypatch):
    base = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    calls: list[int | None] = []

    pages = [
        [kline(base + timedelta(minutes=2), "102"), kline(base + timedelta(minutes=3), "103")],
        [kline(base, "100"), kline(base + timedelta(minutes=1), "101")],
    ]

    async def fake_get_klines(
        *,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ):
        calls.append(end_time_ms)
        return pages[len(calls) - 1]

    client = BinanceRestClient(testnet=True)
    monkeypatch.setattr(client, "get_klines", fake_get_klines)

    try:
        candles = await client.get_historical_closed_candles(
            symbol="BTCUSDT",
            timeframe="1m",
            limit=4,
        )
    finally:
        await client.close()

    assert [c.close for c in candles] == [100.0, 101.0, 102.0, 103.0]
    assert len(calls) == 2
    assert calls[0] is None
    assert calls[1] < int((base + timedelta(minutes=2)).timestamp() * 1000)


@pytest.mark.asyncio
async def test_get_historical_closed_candles_deduplicates_and_trims(monkeypatch):
    base = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    pages = [
        [kline(base + timedelta(minutes=1), "101"), kline(base + timedelta(minutes=2), "102")],
        [kline(base, "100"), kline(base + timedelta(minutes=1), "101")],
    ]
    calls = 0

    async def fake_get_klines(
        *,
        symbol: str,
        interval: str,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ):
        nonlocal calls
        page = pages[calls]
        calls += 1
        return page

    client = BinanceRestClient(testnet=True)
    monkeypatch.setattr(client, "get_klines", fake_get_klines)

    try:
        candles = await client.get_historical_closed_candles(
            symbol="BTCUSDT",
            timeframe="1m",
            limit=2,
        )
    finally:
        await client.close()

    assert [c.close for c in candles] == [101.0, 102.0]
