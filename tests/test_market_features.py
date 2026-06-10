from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.market.features import (
    KlineTakerRow,
    MarketFeatures,
    TakerAggregate,
    aggregate_taker_by_bucket,
    bucket_start_timestamp,
    build_market_features,
    candle_body_pct,
    candle_lower_wick_pct,
    candle_upper_wick_pct,
    compute_taker_buy_ratio,
    order_book_features_from_depth,
    parse_taker_rows,
    volume_spike_ratios,
)
from app.market.models import Candle

START = datetime(2026, 1, 1, tzinfo=UTC)


def make_candle(index: int, *, close: float = 100.0, open_: float = 100.0, high: float = 101.0,
                low: float = 99.0, volume: float = 10.0) -> Candle:
    start = START + timedelta(minutes=index)
    return Candle(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="1m",
        open_time=start,
        close_time=start + timedelta(minutes=1),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def test_market_features_optional_fields_default_none() -> None:
    feat = MarketFeatures(
        exchange="binance_spot",
        symbol="BTCUSDT",
        timeframe="5m",
        open_time=START,
        close_time=START + timedelta(minutes=5),
        close_price=100.0,
        volume=50.0,
    )
    assert feat.taker_buy_ratio is None
    assert feat.order_book_imbalance is None
    assert feat.spread_pct is None


def test_compute_taker_buy_ratio() -> None:
    assert compute_taker_buy_ratio(50.0, 100.0) == 0.5
    assert compute_taker_buy_ratio(None, 100.0) is None
    assert compute_taker_buy_ratio(50.0, 0.0) is None


def test_build_market_features_attaches_taker_and_leaves_order_book_null() -> None:
    candles = [make_candle(0, volume=100.0)]
    taker = {candles[0].close_time: TakerAggregate(quote_volume=12000.0, taker_buy_base_volume=60.0,
                                                   taker_buy_quote_volume=7200.0)}
    rows = build_market_features(candles, exchange="binance_spot", taker_by_close_time=taker)

    assert len(rows) == 1
    row = rows[0]
    assert row.quote_volume == 12000.0
    assert row.taker_buy_base_volume == 60.0
    assert row.taker_buy_ratio == 0.6  # 60 / 100
    # Order-book fields must stay None for historical rows.
    assert row.order_book_bid_volume is None
    assert row.order_book_imbalance is None
    assert row.spread_pct is None


def test_build_market_features_without_taker_is_null() -> None:
    candles = [make_candle(0)]
    rows = build_market_features(candles, exchange="binance_spot")
    assert rows[0].taker_buy_ratio is None
    assert rows[0].quote_volume is None


def test_parse_taker_rows_reads_kline_fields() -> None:
    open_ms = int(START.timestamp() * 1000)
    close_ms = int((START + timedelta(minutes=1)).timestamp() * 1000)
    raw = [[open_ms, "100", "101", "99", "100.5", "12.5", close_ms, "1250.0", 42, "7.5", "750.0", "0"]]
    rows = parse_taker_rows(raw)
    assert len(rows) == 1
    assert rows[0].volume == 12.5
    assert rows[0].quote_volume == 1250.0
    assert rows[0].taker_buy_base_volume == 7.5


def test_parse_taker_rows_skips_short_rows() -> None:
    assert parse_taker_rows([[1, 2, 3]]) == []


def test_aggregate_taker_by_bucket_sums_into_5m() -> None:
    rows = [
        KlineTakerRow(open_time=START + timedelta(minutes=m), volume=10.0, quote_volume=1000.0,
                      taker_buy_base_volume=6.0, taker_buy_quote_volume=600.0)
        for m in range(5)
    ]
    buckets = aggregate_taker_by_bucket(rows, target_timeframe="5m")
    key = bucket_start_timestamp(START, 300)
    assert key in buckets
    assert buckets[key].taker_buy_base_volume == 30.0
    assert buckets[key].quote_volume == 5000.0


def test_bucket_start_timestamp_alignment() -> None:
    # 12:03 in a 5m bucket aligns to 12:00.
    t = datetime(2026, 1, 1, 12, 3, tzinfo=UTC)
    assert bucket_start_timestamp(t, 300) == int(datetime(2026, 1, 1, 12, 0, tzinfo=UTC).timestamp())


def test_order_book_features_from_depth() -> None:
    bids = [(100.0, 2.0), (99.0, 3.0)]
    asks = [(101.0, 1.0), (102.0, 4.0)]
    feats = order_book_features_from_depth(bids, asks)
    assert feats.bid_volume == 5.0
    assert feats.ask_volume == 5.0
    assert feats.imbalance == 0.0
    assert feats.spread_pct == 1.0  # (101-100)/100 * 100


def test_order_book_imbalance_sign() -> None:
    feats = order_book_features_from_depth([(100.0, 9.0)], [(101.0, 1.0)])
    assert feats.imbalance == 0.8  # (9-1)/10


def test_candle_shape_features() -> None:
    candle = make_candle(0, open_=100.0, close=101.0, high=102.0, low=99.5)
    assert candle_body_pct(candle) == 1.0  # |101-100|/100*100
    assert candle_upper_wick_pct(candle) == 1.0  # (102-101)/100*100
    assert candle_lower_wick_pct(candle) == 0.5  # (100-99.5)/100*100


def test_volume_spike_ratios() -> None:
    candles = [make_candle(i, volume=10.0) for i in range(5)] + [make_candle(5, volume=30.0)]
    ratios = volume_spike_ratios(candles, lookback=5)
    assert ratios[:5] == [None, None, None, None, None]
    assert ratios[5] == 3.0  # 30 / mean(10,10,10,10,10)
