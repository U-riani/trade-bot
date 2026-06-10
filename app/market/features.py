"""V22 non-price market features.

This module holds the data model for the ``market_features`` table plus the
pure functions that derive features. It is deliberately split between two kinds
of feature, because conflating them is how research quietly cheats on time:

1. Historical, candle/kline-derived features (volume, quote_volume, taker buy
   volumes, taker_buy_ratio). These exist for any past candle.
2. Order-book features (bid/ask volume, imbalance, spread). Binance REST only
   serves the *current* depth snapshot, so these can be collected going forward
   but DO NOT exist for historical candles. We never synthesize them from price.

Derived ratios used in analysis (volume spike, body %, wick %) are computed at
analysis time from OHLC, not stored here, so the table stays a record of raw
observed quantities rather than a pile of opinions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.market.models import Candle
from app.utils.time import ms_to_datetime
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class MarketFeatures:
    """One row of the market_features table (id/created_at are DB-managed)."""

    exchange: str
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    close_price: float
    volume: float
    quote_volume: float | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None
    taker_buy_ratio: float | None = None
    order_book_bid_volume: float | None = None
    order_book_ask_volume: float | None = None
    order_book_imbalance: float | None = None
    spread_pct: float | None = None


@dataclass(slots=True, frozen=True)
class TakerAggregate:
    """Taker/quote volumes aggregated to a target-timeframe bucket."""

    quote_volume: float | None = None
    taker_buy_base_volume: float | None = None
    taker_buy_quote_volume: float | None = None


@dataclass(slots=True, frozen=True)
class KlineTakerRow:
    """Taker/quote volumes parsed from one raw Binance kline row."""

    open_time: datetime
    volume: float
    quote_volume: float
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


@dataclass(slots=True, frozen=True)
class OrderBookFeatures:
    """Forward-only features derived from a single live depth snapshot."""

    bid_volume: float
    ask_volume: float
    imbalance: float
    spread_pct: float | None


def compute_taker_buy_ratio(taker_buy_base_volume: float | None, volume: float | None) -> float | None:
    """Fraction of base volume that was taker-buy (aggressive buyers).

    Returns None when inputs are missing or volume is zero, rather than guessing.
    A ratio above 0.5 means more market-buy pressure than market-sell pressure
    over the candle.
    """
    if taker_buy_base_volume is None or volume is None or volume <= 0:
        return None
    return taker_buy_base_volume / volume


def build_market_features(
    candles: list[Candle],
    *,
    exchange: str,
    taker_by_close_time: dict[datetime, TakerAggregate] | None = None,
    order_book_by_close_time: dict[datetime, OrderBookFeatures] | None = None,
) -> list[MarketFeatures]:
    """Build feature rows from (already target-timeframe) candles.

    ``taker_by_close_time`` supplies the kline-derived taker/quote volumes keyed
    by the candle's close_time. ``order_book_by_close_time`` is normally empty
    for historical builds and only populated for live forward collection; when a
    candle has no order-book entry its order-book columns stay None instead of
    being invented.
    """
    taker_map = taker_by_close_time or {}
    book_map = order_book_by_close_time or {}
    rows: list[MarketFeatures] = []

    for candle in candles:
        taker = taker_map.get(candle.close_time)
        quote_volume = taker.quote_volume if taker is not None else None
        taker_base = taker.taker_buy_base_volume if taker is not None else None
        taker_quote = taker.taker_buy_quote_volume if taker is not None else None
        ratio = compute_taker_buy_ratio(taker_base, candle.volume)

        book = book_map.get(candle.close_time)
        rows.append(
            MarketFeatures(
                exchange=exchange,
                symbol=candle.symbol,
                timeframe=candle.timeframe,
                open_time=candle.open_time,
                close_time=candle.close_time,
                close_price=candle.close,
                volume=candle.volume,
                quote_volume=quote_volume,
                taker_buy_base_volume=taker_base,
                taker_buy_quote_volume=taker_quote,
                taker_buy_ratio=ratio,
                order_book_bid_volume=book.bid_volume if book is not None else None,
                order_book_ask_volume=book.ask_volume if book is not None else None,
                order_book_imbalance=book.imbalance if book is not None else None,
                spread_pct=book.spread_pct if book is not None else None,
            )
        )
    return rows


def parse_taker_rows(raw_klines: list[list[Any]]) -> list[KlineTakerRow]:
    """Extract taker/quote volumes from raw Binance kline rows.

    Binance kline layout: [0]=open_time, [5]=volume, [7]=quote_volume,
    [9]=taker_buy_base_volume, [10]=taker_buy_quote_volume. Rows missing those
    fields are skipped rather than zero-filled.
    """
    rows: list[KlineTakerRow] = []
    for item in raw_klines:
        if len(item) < 11:
            continue
        rows.append(
            KlineTakerRow(
                open_time=ms_to_datetime(item[0]),
                volume=float(item[5]),
                quote_volume=float(item[7]),
                taker_buy_base_volume=float(item[9]),
                taker_buy_quote_volume=float(item[10]),
            )
        )
    return rows


def bucket_start_timestamp(value: datetime, bucket_seconds: int) -> int:
    """UTC-aligned bucket start (seconds) matching resample_candles' alignment."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    timestamp = int(value.timestamp())
    return timestamp - (timestamp % bucket_seconds)


def aggregate_taker_by_bucket(
    rows: list[KlineTakerRow],
    *,
    target_timeframe: str,
) -> dict[int, TakerAggregate]:
    """Sum 1m taker/quote volumes into target-timeframe buckets, keyed by bucket start ts.

    Taker and quote volumes are additive, so a 5m bucket's taker_buy_base is just
    the sum of its five 1m taker_buy_base values. The key is the UTC-aligned
    bucket start so it can be matched to a resampled candle via
    ``bucket_start_timestamp(candle.open_time, target_seconds)``.
    """
    target_seconds = timeframe_to_seconds(target_timeframe)
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        key = bucket_start_timestamp(row.open_time, target_seconds)
        sums[key][0] += row.quote_volume
        sums[key][1] += row.taker_buy_base_volume
        sums[key][2] += row.taker_buy_quote_volume

    return {
        key: TakerAggregate(
            quote_volume=values[0],
            taker_buy_base_volume=values[1],
            taker_buy_quote_volume=values[2],
        )
        for key, values in sums.items()
    }


def candle_body_pct(candle: Candle) -> float:
    """Absolute candle body size as a percent of the open price."""
    if candle.open <= 0:
        return 0.0
    return abs(candle.close - candle.open) / candle.open * 100.0


def candle_upper_wick_pct(candle: Candle) -> float:
    """Upper wick (high above the body) as a percent of the open price."""
    if candle.open <= 0:
        return 0.0
    body_top = max(candle.open, candle.close)
    return max(candle.high - body_top, 0.0) / candle.open * 100.0


def candle_lower_wick_pct(candle: Candle) -> float:
    """Lower wick (low below the body) as a percent of the open price."""
    if candle.open <= 0:
        return 0.0
    body_bottom = min(candle.open, candle.close)
    return max(body_bottom - candle.low, 0.0) / candle.open * 100.0


def volume_spike_ratios(candles: list[Candle], lookback: int) -> list[float | None]:
    """Per-candle volume divided by the mean volume of the prior `lookback` candles.

    Index i is None until there are `lookback` prior candles, or when the prior
    mean volume is zero. A ratio of 2.0 means twice the recent average volume.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")

    ratios: list[float | None] = []
    for index, candle in enumerate(candles):
        if index < lookback:
            ratios.append(None)
            continue
        window = candles[index - lookback : index]
        mean_volume = sum(item.volume for item in window) / lookback
        ratios.append(candle.volume / mean_volume if mean_volume > 0 else None)
    return ratios


def order_book_features_from_depth(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
) -> OrderBookFeatures:
    """Compute order-book features from a depth snapshot (price, qty) levels.

    FORWARD-ONLY: this describes the book at the instant it was fetched. It must
    never be attached to a historical candle, because that book no longer exists
    and Binance does not serve past snapshots.

    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume), in
    [-1, 1]; positive means more resting bid size than ask size.
    """
    bid_volume = sum(qty for _price, qty in bids)
    ask_volume = sum(qty for _price, qty in asks)

    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total > 0 else 0.0

    spread_pct: float | None = None
    if bids and asks:
        best_bid = max(price for price, _qty in bids)
        best_ask = min(price for price, _qty in asks)
        if best_bid > 0 and best_ask >= best_bid:
            spread_pct = (best_ask - best_bid) / best_bid * 100.0

    return OrderBookFeatures(
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        imbalance=imbalance,
        spread_pct=spread_pct,
    )
