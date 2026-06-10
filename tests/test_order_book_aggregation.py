from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.market.models import Candle
from app.market.order_book import build_order_book_snapshot
from scripts.aggregate_order_book_features import build_order_book_feature_rows

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)  # minute-aligned


def make_candle(minute: int) -> Candle:
    open_time = START + timedelta(minutes=minute)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=open_time,
        close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0 + minute,
        volume=10.0 + minute,
    )


def make_snapshot(minute: int, second: int, bid_qty: float = 2.0):
    bids = [(100.0 - i * 0.1, bid_qty) for i in range(20)]
    asks = [(101.0 + i * 0.1, 1.0) for i in range(20)]
    return build_order_book_snapshot(
        exchange="binance_spot",
        symbol="BTCUSDT",
        collected_at=START + timedelta(minutes=minute, seconds=second),
        bids=bids,
        asks=asks,
        raw_depth_limit=100,
    )


def test_snapshots_join_to_correct_candle_bucket() -> None:
    candles = [make_candle(0), make_candle(1), make_candle(2)]
    # Two snapshots in minute 0, one in minute 2, none in minute 1.
    snapshots = [make_snapshot(0, 10), make_snapshot(0, 40), make_snapshot(2, 5)]

    rows = build_order_book_feature_rows(
        candles, snapshots, exchange="binance_spot", symbol="BTCUSDT", timeframe="1m"
    )

    by_close = {r.close_time: r for r in rows}
    assert len(rows) == 2  # minute 1 had no snapshots
    minute0 = by_close[candles[0].close_time]
    assert minute0.order_book_snapshot_count == 2
    assert minute0.close_price == 100.0  # carried from candle
    assert minute0.volume == 10.0
    assert minute0.imbalance_top_5 > 0  # bid-heavy snapshots
    minute2 = by_close[candles[2].close_time]
    assert minute2.order_book_snapshot_count == 1


def test_min_snapshots_filter() -> None:
    candles = [make_candle(0)]
    snapshots = [make_snapshot(0, 10)]  # only 1 snapshot in the bucket
    rows = build_order_book_feature_rows(
        candles, snapshots, exchange="binance_spot", symbol="BTCUSDT", timeframe="1m", min_snapshots=2
    )
    assert rows == []


def test_no_snapshots_yields_no_rows() -> None:
    candles = [make_candle(0), make_candle(1)]
    rows = build_order_book_feature_rows(
        candles, [], exchange="binance_spot", symbol="BTCUSDT", timeframe="1m"
    )
    assert rows == []


def test_5m_bucket_groups_multiple_minutes() -> None:
    # The script passes already-resampled candles, so a 5m run gets one 5m
    # candle. Snapshots spread across its five minutes all land in its bucket.
    five_min_candle = Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="5m",
        open_time=START,  # 12:00, aligned to a 5m boundary
        close_time=START + timedelta(minutes=5) - timedelta(milliseconds=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=50.0,
    )
    snapshots = [make_snapshot(0, 5), make_snapshot(2, 5), make_snapshot(4, 5)]
    rows = build_order_book_feature_rows(
        [five_min_candle], snapshots, exchange="binance_spot", symbol="BTCUSDT", timeframe="5m"
    )
    assert len(rows) == 1
    assert rows[0].order_book_snapshot_count == 3
