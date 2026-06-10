from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from app.market.models import Candle
from app.utils.timeframe import timeframe_to_seconds


def resample_candles(
    candles: list[Candle],
    *,
    target_timeframe: str,
    source_timeframe: str = "1m",
    require_complete_buckets: bool = True,
) -> list[Candle]:
    """Aggregate lower-timeframe candles into a higher timeframe.

    V19 uses this to research whether 1m noise is what is turning every strategy
    into a fee donation machine. The function aligns buckets to natural UTC
    boundaries, so 5m candles start at :00/:05/:10, and 15m candles at
    :00/:15/:30/:45.
    """
    sorted_candles = sorted(candles, key=lambda item: item.open_time)
    if not sorted_candles:
        return []

    source_seconds = timeframe_to_seconds(source_timeframe)
    target_seconds = timeframe_to_seconds(target_timeframe)

    if target_seconds < source_seconds:
        raise ValueError("target_timeframe must be greater than or equal to source_timeframe")
    if target_seconds % source_seconds != 0:
        raise ValueError("target_timeframe must be a clean multiple of source_timeframe")
    if target_seconds == source_seconds:
        return [
            Candle(
                exchange=candle.exchange,
                symbol=candle.symbol,
                timeframe=target_timeframe,
                open_time=candle.open_time,
                close_time=candle.close_time,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
                is_closed=candle.is_closed,
            )
            for candle in sorted_candles
        ]

    expected_per_bucket = target_seconds // source_seconds
    buckets: dict[int, list[Candle]] = defaultdict(list)
    for candle in sorted_candles:
        bucket_start_ts = _bucket_start_timestamp(candle.open_time, target_seconds)
        buckets[bucket_start_ts].append(candle)

    resampled: list[Candle] = []
    for _bucket_start_ts, bucket_candles in sorted(buckets.items(), key=lambda item: item[0]):
        bucket_candles = sorted(bucket_candles, key=lambda item: item.open_time)
        if require_complete_buckets and not _is_complete_bucket(bucket_candles, expected_per_bucket, source_seconds):
            continue

        first = bucket_candles[0]
        last = bucket_candles[-1]
        resampled.append(
            Candle(
                exchange=first.exchange,
                symbol=first.symbol,
                timeframe=target_timeframe,
                open_time=first.open_time,
                close_time=last.close_time,
                open=first.open,
                high=max(candle.high for candle in bucket_candles),
                low=min(candle.low for candle in bucket_candles),
                close=last.close,
                volume=sum(candle.volume for candle in bucket_candles),
                is_closed=all(candle.is_closed for candle in bucket_candles),
            )
        )
    return resampled


def _bucket_start_timestamp(value: datetime, bucket_seconds: int) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    timestamp = int(value.timestamp())
    return timestamp - (timestamp % bucket_seconds)


def _is_complete_bucket(candles: list[Candle], expected_count: int, source_seconds: int) -> bool:
    if len(candles) != expected_count:
        return False

    previous_open = candles[0].open_time
    for candle in candles[1:]:
        delta = candle.open_time - previous_open
        if int(delta.total_seconds()) != source_seconds:
            return False
        previous_open = candle.open_time
    return True
