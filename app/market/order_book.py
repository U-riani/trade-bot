"""V23 order-book snapshot model and pure depth math.

A snapshot is one observation of the live Binance order book. From the raw
(price, qty) levels we derive:
  * spread / spread_pct from the best bid and ask
  * resting volume on each side within the top 5 / 10 / 20 levels
  * imbalance at each of those depths: (bid_vol - ask_vol) / (bid_vol + ask_vol)

imbalance is in [-1, 1]; positive means more resting bid size than ask size,
which is a (weak, untested) sign of buy pressure. Whether it predicts anything
is exactly what V23 is collecting data to find out later.

Everything here is pure and forward-looking: Binance only serves the current
book, so these values describe the instant they were fetched and are never
attached to historical candles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

DEPTHS = (5, 10, 20)

DepthLevel = tuple[float, float]  # (price, quantity)


@dataclass(slots=True, frozen=True)
class OrderBookSnapshot:
    exchange: str
    symbol: str
    collected_at: datetime
    best_bid_price: float
    best_ask_price: float
    spread: float
    spread_pct: float
    bid_volume_top_5: float
    ask_volume_top_5: float
    bid_volume_top_10: float
    ask_volume_top_10: float
    bid_volume_top_20: float
    ask_volume_top_20: float
    imbalance_top_5: float
    imbalance_top_10: float
    imbalance_top_20: float
    raw_depth_limit: int


def parse_depth_levels(levels: list) -> list[DepthLevel]:
    """Parse Binance depth levels (``[["price","qty"], ...]``) into floats.

    Zero-quantity levels are dropped (Binance uses them as deletions).
    """
    parsed: list[DepthLevel] = []
    for level in levels:
        if len(level) < 2:
            continue
        price = float(level[0])
        quantity = float(level[1])
        if quantity <= 0:
            continue
        parsed.append((price, quantity))
    return parsed


def top_volume(levels: list[DepthLevel], depth: int) -> float:
    """Sum quantity over the first ``depth`` levels (levels must be pre-sorted)."""
    return sum(quantity for _price, quantity in levels[:depth])


def imbalance(bid_volume: float, ask_volume: float) -> float:
    """(bid - ask) / (bid + ask), in [-1, 1]; 0 when the book side is empty."""
    total = bid_volume + ask_volume
    if total <= 0:
        return 0.0
    return (bid_volume - ask_volume) / total


def build_order_book_snapshot(
    *,
    exchange: str,
    symbol: str,
    collected_at: datetime,
    bids: list[DepthLevel],
    asks: list[DepthLevel],
    raw_depth_limit: int,
) -> OrderBookSnapshot | None:
    """Build a snapshot from parsed depth levels.

    Bids must be sorted best (highest price) first and asks best (lowest price)
    first, which is how Binance returns them. Returns None when either side is
    empty, since spread/imbalance are undefined without a two-sided book.
    """
    if not bids or not asks:
        return None

    best_bid_price = bids[0][0]
    best_ask_price = asks[0][0]
    spread = best_ask_price - best_bid_price
    spread_pct = (spread / best_bid_price * 100.0) if best_bid_price > 0 else 0.0

    bid_vol = {depth: top_volume(bids, depth) for depth in DEPTHS}
    ask_vol = {depth: top_volume(asks, depth) for depth in DEPTHS}

    return OrderBookSnapshot(
        exchange=exchange,
        symbol=symbol,
        collected_at=collected_at,
        best_bid_price=best_bid_price,
        best_ask_price=best_ask_price,
        spread=spread,
        spread_pct=spread_pct,
        bid_volume_top_5=bid_vol[5],
        ask_volume_top_5=ask_vol[5],
        bid_volume_top_10=bid_vol[10],
        ask_volume_top_10=ask_vol[10],
        bid_volume_top_20=bid_vol[20],
        ask_volume_top_20=ask_vol[20],
        imbalance_top_5=imbalance(bid_vol[5], ask_vol[5]),
        imbalance_top_10=imbalance(bid_vol[10], ask_vol[10]),
        imbalance_top_20=imbalance(bid_vol[20], ask_vol[20]),
        raw_depth_limit=raw_depth_limit,
    )


@dataclass(slots=True, frozen=True)
class OrderBookBucketAggregate:
    """Order-book features aggregated over all snapshots in one candle bucket."""

    snapshot_count: int
    avg_spread_pct: float
    avg_imbalance_top_5: float
    avg_imbalance_top_10: float
    avg_imbalance_top_20: float
    max_imbalance: float
    min_imbalance: float
    avg_bid_volume_top_20: float
    avg_ask_volume_top_20: float
    first_snapshot_at: datetime
    last_snapshot_at: datetime


def aggregate_snapshots(snapshots: list[OrderBookSnapshot]) -> OrderBookBucketAggregate | None:
    """Average a bucket's snapshots into one feature record.

    Returns None for an empty bucket so callers can skip it rather than write a
    fabricated zero row. max/min imbalance use the top-10 depth as the
    representative imbalance.
    """
    count = len(snapshots)
    if count == 0:
        return None

    def avg(getter) -> float:
        return sum(getter(s) for s in snapshots) / count

    imbalance_top_10 = [s.imbalance_top_10 for s in snapshots]
    return OrderBookBucketAggregate(
        snapshot_count=count,
        avg_spread_pct=avg(lambda s: s.spread_pct),
        avg_imbalance_top_5=avg(lambda s: s.imbalance_top_5),
        avg_imbalance_top_10=avg(lambda s: s.imbalance_top_10),
        avg_imbalance_top_20=avg(lambda s: s.imbalance_top_20),
        max_imbalance=max(imbalance_top_10),
        min_imbalance=min(imbalance_top_10),
        avg_bid_volume_top_20=avg(lambda s: s.bid_volume_top_20),
        avg_ask_volume_top_20=avg(lambda s: s.ask_volume_top_20),
        first_snapshot_at=min(s.collected_at for s in snapshots),
        last_snapshot_at=max(s.collected_at for s in snapshots),
    )


def snapshot_from_depth_response(
    depth: dict,
    *,
    exchange: str,
    symbol: str,
    collected_at: datetime,
    raw_depth_limit: int,
) -> OrderBookSnapshot | None:
    """Build a snapshot directly from a Binance /depth JSON response."""
    bids = parse_depth_levels(depth.get("bids", []))
    asks = parse_depth_levels(depth.get("asks", []))
    return build_order_book_snapshot(
        exchange=exchange,
        symbol=symbol,
        collected_at=collected_at,
        bids=bids,
        asks=asks,
        raw_depth_limit=raw_depth_limit,
    )
