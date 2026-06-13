"""V26 historical aggregate-trade pressure features.

Binance Spot does not provide historical order-book depth through normal REST, but
it does provide historical aggregate trades. This module converts aggTrades into
bucketed taker pressure features that can be researched while live order-book data
continues to accumulate.

No trading logic lives here. Everything is pure and unit-testable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.market.features import bucket_start_timestamp
from app.utils.time import ms_to_datetime
from app.utils.timeframe import timeframe_to_seconds


@dataclass(slots=True, frozen=True)
class AggTradeRow:
    trade_id: int
    price: float
    quantity: float
    quote_quantity: float
    trade_time: datetime
    buyer_is_maker: bool

    @property
    def taker_is_buy(self) -> bool:
        """False buyer_is_maker means buyer took liquidity, so aggressive buy."""
        return not self.buyer_is_maker


@dataclass(slots=True, frozen=True)
class TradePressureAggregate:
    bucket_start_ts: int
    trade_count: int
    taker_buy_trade_count: int
    taker_sell_trade_count: int
    taker_buy_base_volume: float
    taker_sell_base_volume: float
    taker_buy_quote_volume: float
    taker_sell_quote_volume: float
    taker_net_base_volume: float
    taker_net_quote_volume: float
    taker_buy_trade_ratio: float | None
    taker_buy_base_ratio: float | None
    taker_buy_quote_ratio: float | None
    avg_trade_quote_size: float | None
    trade_count_intensity: float
    quote_volume_intensity: float


def parse_agg_trade_row(item: dict[str, Any]) -> AggTradeRow:
    """Parse one Binance aggTrade row.

    Binance fields:
    - a: aggregate trade id
    - p: price
    - q: quantity
    - T: trade time in ms
    - m: buyer is maker

    Direction mapping:
    - m == True  -> buyer was maker, seller was taker, aggressive sell
    - m == False -> buyer was taker, aggressive buy
    """
    price = float(item["p"])
    quantity = float(item["q"])
    return AggTradeRow(
        trade_id=int(item["a"]),
        price=price,
        quantity=quantity,
        quote_quantity=price * quantity,
        trade_time=ms_to_datetime(int(item["T"])),
        buyer_is_maker=bool(item["m"]),
    )


def parse_agg_trades(items: list[dict[str, Any]]) -> list[AggTradeRow]:
    rows: list[AggTradeRow] = []
    for item in items:
        try:
            rows.append(parse_agg_trade_row(item))
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def aggregate_trade_pressure_by_bucket(
    rows: list[AggTradeRow],
    *,
    target_timeframe: str,
) -> dict[int, TradePressureAggregate]:
    """Aggregate parsed trades into UTC-aligned timeframe buckets."""
    target_seconds = timeframe_to_seconds(target_timeframe)
    bucket_minutes = target_seconds / 60.0

    sums: dict[int, dict[str, float]] = defaultdict(
        lambda: {
            "trade_count": 0.0,
            "buy_count": 0.0,
            "sell_count": 0.0,
            "buy_base": 0.0,
            "sell_base": 0.0,
            "buy_quote": 0.0,
            "sell_quote": 0.0,
        }
    )

    for row in rows:
        key = bucket_start_timestamp(row.trade_time, target_seconds)
        bucket = sums[key]
        bucket["trade_count"] += 1.0
        if row.taker_is_buy:
            bucket["buy_count"] += 1.0
            bucket["buy_base"] += row.quantity
            bucket["buy_quote"] += row.quote_quantity
        else:
            bucket["sell_count"] += 1.0
            bucket["sell_base"] += row.quantity
            bucket["sell_quote"] += row.quote_quantity

    result: dict[int, TradePressureAggregate] = {}
    for key, values in sums.items():
        trade_count = int(values["trade_count"])
        buy_count = int(values["buy_count"])
        sell_count = int(values["sell_count"])
        buy_base = values["buy_base"]
        sell_base = values["sell_base"]
        buy_quote = values["buy_quote"]
        sell_quote = values["sell_quote"]
        total_base = buy_base + sell_base
        total_quote = buy_quote + sell_quote
        result[key] = TradePressureAggregate(
            bucket_start_ts=key,
            trade_count=trade_count,
            taker_buy_trade_count=buy_count,
            taker_sell_trade_count=sell_count,
            taker_buy_base_volume=buy_base,
            taker_sell_base_volume=sell_base,
            taker_buy_quote_volume=buy_quote,
            taker_sell_quote_volume=sell_quote,
            taker_net_base_volume=buy_base - sell_base,
            taker_net_quote_volume=buy_quote - sell_quote,
            taker_buy_trade_ratio=_ratio(float(buy_count), float(trade_count)),
            taker_buy_base_ratio=_ratio(buy_base, total_base),
            taker_buy_quote_ratio=_ratio(buy_quote, total_quote),
            avg_trade_quote_size=_ratio(total_quote, float(trade_count)),
            trade_count_intensity=float(trade_count) / bucket_minutes if bucket_minutes > 0 else 0.0,
            quote_volume_intensity=total_quote / bucket_minutes if bucket_minutes > 0 else 0.0,
        )
    return result
