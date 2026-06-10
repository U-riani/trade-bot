from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.backtesting.order_book_pipeline import (
    REASON_NO_FEATURE_ROWS,
    REASON_NO_FORWARD_RETURNS,
    REASON_NO_SNAPSHOTS,
    REASON_NOT_ENOUGH_SAMPLES,
    REASON_READY,
    compute_match_status,
    compute_readiness,
    compute_snapshot_status,
    order_book_sample_size,
    snapshots_after_latest_candle,
)
from app.market.features import MarketFeatures
from app.market.models import Candle
from app.market.order_book import build_order_book_snapshot

START = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_snapshot(minute: int, second: int = 0):
    bids = [(100.0 - i * 0.1, 2.0) for i in range(20)]
    asks = [(101.0 + i * 0.1, 1.0) for i in range(20)]
    return build_order_book_snapshot(
        exchange="binance_spot", symbol="BTCUSDT",
        collected_at=START + timedelta(minutes=minute, seconds=second),
        bids=bids, asks=asks, raw_depth_limit=100,
    )


def make_candle(minute: int) -> Candle:
    open_time = START + timedelta(minutes=minute)
    return Candle(
        exchange="binance_spot", symbol="BTCUSDT", timeframe="1m",
        open_time=open_time, close_time=open_time + timedelta(minutes=1) - timedelta(milliseconds=1),
        open=100.0, high=101.0, low=99.0, close=100.0 + minute, volume=10.0,
    )


def make_feature_row(close_time, *, snapshot_count, imbalance=0.5) -> MarketFeatures:
    return MarketFeatures(
        exchange="binance_spot", symbol="BTCUSDT", timeframe="1m",
        open_time=close_time - timedelta(minutes=1), close_time=close_time,
        close_price=100.0, volume=10.0,
        order_book_imbalance=imbalance, order_book_snapshot_count=snapshot_count,
    )


# --- snapshot status ---

def test_snapshot_status_basic() -> None:
    snaps = [make_snapshot(0, 0), make_snapshot(0, 20), make_snapshot(0, 40)]
    status = compute_snapshot_status(snaps, reference_time=START + timedelta(seconds=50))
    assert status.total == 3
    assert status.first_at == START
    assert status.latest_at == START + timedelta(seconds=40)
    assert status.last_1h == 3
    assert status.last_24h == 3
    assert status.avg_interval_seconds == 20.0  # 40s span / 2 gaps


def test_snapshot_status_rolling_windows() -> None:
    # one snapshot 2h ago, one now -> last_1h counts only the recent one
    snaps = [make_snapshot(0), make_snapshot(120)]
    status = compute_snapshot_status(snaps, reference_time=START + timedelta(minutes=120))
    assert status.last_1h == 1
    assert status.last_24h == 2


def test_snapshot_status_empty() -> None:
    status = compute_snapshot_status([], reference_time=START)
    assert status.total == 0
    assert status.first_at is None
    assert status.avg_interval_seconds is None


# --- match status / unmatched detection ---

def test_match_status_detects_unmatched_too_new() -> None:
    candles = [make_candle(0), make_candle(1), make_candle(2)]
    latest_close = candles[-1].close_time
    # two snapshots in minute 0 (has candle), one in minute 5 (no candle, too new)
    snaps = [make_snapshot(0, 5), make_snapshot(0, 30), make_snapshot(5, 0)]
    feature_rows = [make_feature_row(candles[0].close_time, snapshot_count=2)]

    status = compute_match_status(
        timeframe="1m", snapshots=snaps, candles=candles,
        feature_rows=feature_rows, latest_candle_close=latest_close,
    )
    assert status.matched_snapshots == 2
    assert status.unmatched_snapshots == 1
    assert status.too_new_snapshots == 1
    assert status.buckets_with_order_book == 1
    assert status.latest_order_book_bucket_close == candles[0].close_time


# --- sample size / readiness ---

def test_order_book_sample_size_by_horizon() -> None:
    candles = [make_candle(m) for m in range(10)]
    rows = [
        make_feature_row(candles[5].close_time, snapshot_count=1),
        make_feature_row(candles[9].close_time, snapshot_count=1),
    ]
    assert order_book_sample_size(candles, rows, 1) == 1  # idx5 ok, idx9 has no next
    assert order_book_sample_size(candles, rows, 3) == 1  # idx5 (5+3<10), idx9 no
    assert order_book_sample_size(candles, rows, 6) == 0  # idx5 (5+6>=10) no


def test_readiness_no_snapshots() -> None:
    candles = [make_candle(m) for m in range(10)]
    r = compute_readiness(timeframe="1m", candles=candles, feature_rows=[], snapshot_total=0, min_samples=100)
    assert r.reason == REASON_NO_SNAPSHOTS and not r.ready


def test_readiness_no_feature_rows() -> None:
    candles = [make_candle(m) for m in range(10)]
    r = compute_readiness(timeframe="1m", candles=candles, feature_rows=[], snapshot_total=5, min_samples=100)
    assert r.reason == REASON_NO_FEATURE_ROWS and not r.ready


def test_readiness_no_forward_returns() -> None:
    candles = [make_candle(m) for m in range(10)]
    rows = [make_feature_row(candles[9].close_time, snapshot_count=1)]  # last candle, no forward
    r = compute_readiness(timeframe="1m", candles=candles, feature_rows=rows, snapshot_total=5, min_samples=100)
    assert r.reason == REASON_NO_FORWARD_RETURNS and not r.ready


def test_readiness_not_enough_samples() -> None:
    candles = [make_candle(m) for m in range(10)]
    rows = [make_feature_row(candles[i].close_time, snapshot_count=1) for i in range(3)]
    r = compute_readiness(timeframe="1m", candles=candles, feature_rows=rows, snapshot_total=5, min_samples=100)
    assert r.reason == REASON_NOT_ENOUGH_SAMPLES and not r.ready


def test_readiness_ready() -> None:
    candles = [make_candle(m) for m in range(200)]
    rows = [make_feature_row(candles[i].close_time, snapshot_count=1) for i in range(150)]
    r = compute_readiness(timeframe="1m", candles=candles, feature_rows=rows, snapshot_total=200, min_samples=100)
    assert r.reason == REASON_READY and r.ready
    assert r.sample_size_h6 >= 100


# --- snapshots after latest candle ---

def test_snapshots_after_latest_candle() -> None:
    snaps = [make_snapshot(0), make_snapshot(5)]
    latest_close = START + timedelta(minutes=3)
    assert snapshots_after_latest_candle(snaps, latest_close) == 1
    assert snapshots_after_latest_candle(snaps, None) == 2  # no candles -> all are "after"
