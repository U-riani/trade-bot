from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.market.models import Candle
from app.utils.time import utc_now
from app.utils.timeframe import timeframe_to_seconds


@dataclass(frozen=True, slots=True)
class CandleWarmupValidation:
    can_use: bool
    reason: str
    loaded_count: int
    latest_close_time: datetime | None = None


def validate_startup_candles(
    candles: list[Candle],
    *,
    timeframe: str,
    max_age_seconds: int,
    gap_tolerance_seconds: int,
) -> CandleWarmupValidation:
    if not candles:
        return CandleWarmupValidation(
            can_use=False,
            reason="no_candles_found",
            loaded_count=0,
        )

    expected_gap_seconds = timeframe_to_seconds(timeframe)
    latest = candles[-1]
    age_seconds = (utc_now() - latest.close_time).total_seconds()

    if age_seconds > max_age_seconds:
        return CandleWarmupValidation(
            can_use=False,
            reason=(
                "stale_candles: "
                f"latest_age_seconds={age_seconds:.2f}, "
                f"max_age_seconds={max_age_seconds}"
            ),
            loaded_count=len(candles),
            latest_close_time=latest.close_time,
        )

    max_allowed_gap = expected_gap_seconds + gap_tolerance_seconds
    for previous, current in zip(candles[:-1], candles[1:], strict=True):
        actual_gap = (current.close_time - previous.close_time).total_seconds()
        if actual_gap <= 0:
            return CandleWarmupValidation(
                can_use=False,
                reason=(
                    "invalid_candle_order: "
                    f"previous_close={previous.close_time.isoformat()}, "
                    f"current_close={current.close_time.isoformat()}"
                ),
                loaded_count=len(candles),
                latest_close_time=latest.close_time,
            )
        if actual_gap > max_allowed_gap:
            return CandleWarmupValidation(
                can_use=False,
                reason=(
                    "candle_gap_detected: "
                    f"previous_close={previous.close_time.isoformat()}, "
                    f"current_close={current.close_time.isoformat()}, "
                    f"gap_seconds={actual_gap:.2f}, "
                    f"max_allowed_gap_seconds={max_allowed_gap}"
                ),
                loaded_count=len(candles),
                latest_close_time=latest.close_time,
            )

    return CandleWarmupValidation(
        can_use=True,
        reason="candles_fresh_and_continuous",
        loaded_count=len(candles),
        latest_close_time=latest.close_time,
    )
