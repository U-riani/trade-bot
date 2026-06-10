from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.market.order_book import (
    aggregate_snapshots,
    build_order_book_snapshot,
    imbalance,
    parse_depth_levels,
    snapshot_from_depth_response,
    top_volume,
)

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _bids(qty: float) -> list[tuple[float, float]]:
    # 20 descending bid prices (best first), constant qty.
    return [(100.0 - i * 0.1, qty) for i in range(20)]


def _asks(qty: float) -> list[tuple[float, float]]:
    # 20 ascending ask prices (best first), constant qty.
    return [(101.0 + i * 0.1, qty) for i in range(20)]


def test_parse_depth_levels_drops_zero_qty() -> None:
    levels = parse_depth_levels([["100.5", "1.0"], ["100.4", "0"], ["100.3", "2.5"]])
    assert levels == [(100.5, 1.0), (100.3, 2.5)]


def test_top_volume() -> None:
    levels = [(100.0, 1.0), (99.0, 2.0), (98.0, 3.0)]
    assert top_volume(levels, 2) == 3.0
    assert top_volume(levels, 10) == 6.0  # depth beyond available sums all


def test_imbalance_basic() -> None:
    assert imbalance(10.0, 5.0) == (10.0 - 5.0) / 15.0
    assert imbalance(0.0, 0.0) == 0.0  # empty side -> defined as 0


def test_build_snapshot_spread_and_balanced_imbalance() -> None:
    snap = build_order_book_snapshot(
        exchange="binance_spot", symbol="BTCUSDT", collected_at=NOW,
        bids=_bids(1.0), asks=_asks(1.0), raw_depth_limit=100,
    )
    assert snap is not None
    assert snap.best_bid_price == 100.0
    assert snap.best_ask_price == 101.0
    assert snap.spread == 1.0
    assert snap.spread_pct == 1.0  # 1 / 100 * 100
    assert snap.bid_volume_top_5 == 5.0
    assert snap.ask_volume_top_10 == 10.0
    assert snap.imbalance_top_20 == 0.0  # equal sizes


def test_build_snapshot_imbalance_sign() -> None:
    snap = build_order_book_snapshot(
        exchange="binance_spot", symbol="BTCUSDT", collected_at=NOW,
        bids=_bids(2.0), asks=_asks(1.0), raw_depth_limit=100,
    )
    assert snap is not None
    # bid_vol_5 = 10, ask_vol_5 = 5 -> (10-5)/15
    assert snap.imbalance_top_5 == (10.0 - 5.0) / 15.0
    assert snap.imbalance_top_5 > 0  # bid-heavy


def test_build_snapshot_empty_book_is_none() -> None:
    assert build_order_book_snapshot(
        exchange="x", symbol="y", collected_at=NOW, bids=[], asks=_asks(1.0), raw_depth_limit=100
    ) is None


def test_snapshot_from_depth_response() -> None:
    depth = {
        "bids": [[str(100.0 - i * 0.1), "1.0"] for i in range(20)],
        "asks": [[str(101.0 + i * 0.1), "1.0"] for i in range(20)],
    }
    snap = snapshot_from_depth_response(depth, exchange="binance_spot", symbol="BTCUSDT",
                                        collected_at=NOW, raw_depth_limit=100)
    assert snap is not None
    assert snap.spread_pct == 1.0
    assert snap.raw_depth_limit == 100


def test_aggregate_snapshots_averages() -> None:
    snap_a = build_order_book_snapshot(exchange="e", symbol="s", collected_at=NOW,
                                       bids=_bids(2.0), asks=_asks(1.0), raw_depth_limit=100)
    snap_b = build_order_book_snapshot(exchange="e", symbol="s", collected_at=NOW + timedelta(seconds=5),
                                       bids=_bids(1.0), asks=_asks(1.0), raw_depth_limit=100)
    agg = aggregate_snapshots([snap_a, snap_b])
    assert agg is not None
    assert agg.snapshot_count == 2
    # avg of imbalance(0.333) and imbalance(0.0)
    assert agg.avg_imbalance_top_5 == (snap_a.imbalance_top_5 + snap_b.imbalance_top_5) / 2
    assert agg.first_snapshot_at == NOW
    assert agg.last_snapshot_at == NOW + timedelta(seconds=5)


def test_aggregate_empty_is_none() -> None:
    assert aggregate_snapshots([]) is None
