from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.market.trade_pressure import aggregate_trade_pressure_by_bucket, parse_agg_trade_row, parse_agg_trades


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def test_parse_agg_trade_direction_mapping() -> None:
    buy = parse_agg_trade_row({"a": 1, "p": "100", "q": "2", "T": _ms(datetime(2026, 1, 1, tzinfo=UTC)), "m": False})
    sell = parse_agg_trade_row({"a": 2, "p": "100", "q": "3", "T": _ms(datetime(2026, 1, 1, tzinfo=UTC)), "m": True})
    assert buy.taker_is_buy is True
    assert sell.taker_is_buy is False
    assert buy.quote_quantity == 200


def test_aggregate_trade_pressure_ratios_and_intensity() -> None:
    raw = [
        {"a": 1, "p": "100", "q": "2", "T": _ms(datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)), "m": False},
        {"a": 2, "p": "100", "q": "1", "T": _ms(datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)), "m": True},
    ]
    rows = parse_agg_trades(raw)
    buckets = aggregate_trade_pressure_by_bucket(rows, target_timeframe="1m")
    assert len(buckets) == 1
    agg = next(iter(buckets.values()))
    assert agg.trade_count == 2
    assert agg.taker_buy_trade_count == 1
    assert agg.taker_sell_trade_count == 1
    assert agg.taker_buy_base_volume == 2
    assert agg.taker_sell_base_volume == 1
    assert agg.taker_net_base_volume == 1
    assert agg.taker_buy_trade_ratio == pytest.approx(0.5)
    assert agg.taker_buy_base_ratio == pytest.approx(2 / 3)
    assert agg.avg_trade_quote_size == pytest.approx(150)
    assert agg.trade_count_intensity == pytest.approx(2)
