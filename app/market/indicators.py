from __future__ import annotations

from app.market.models import Candle


def ema(values: list[float], period: int) -> list[float]:
    if period <= 0:
        raise ValueError("period must be positive")
    if not values:
        return []

    multiplier = 2 / (period + 1)
    result = [float(values[0])]

    for value in values[1:]:
        result.append((float(value) - result[-1]) * multiplier + result[-1])

    return result


def rsi(values: list[float], period: int = 14) -> float | None:
    if period <= 0:
        raise ValueError("period must be positive")
    if len(values) < period + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []

    recent_values = [float(value) for value in values[-(period + 1):]]

    # We compare every value with the next value.
    # recent_values has period + 1 items, therefore both sliced lists below
    # have exactly period items. Do not zip the unsliced list with recent_values[1:],
    # because strict=True correctly raises when lengths differ. Tiny detail, huge tantrum.
    for previous, current in zip(recent_values[:-1], recent_values[1:], strict=True):
        change = current - previous
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        elif change < 0:
            gains.append(0.0)
            losses.append(abs(change))
        else:
            gains.append(0.0)
            losses.append(0.0)

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0

    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def atr(candles: list[Candle], period: int = 14) -> float | None:
    """Average True Range using the most recent closed candles.

    This is deliberately simple and deterministic for local backtesting. It is
    not Wilder-smoothed ATR, because the first use here is filtering tiny noisy
    1m moves, not winning a sacred indicator purity contest.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    if len(candles) < period + 1:
        return None

    recent = candles[-(period + 1) :]
    true_ranges: list[float] = []
    for previous, current in zip(recent[:-1], recent[1:], strict=True):
        high = float(current.high)
        low = float(current.low)
        previous_close = float(previous.close)
        true_ranges.append(
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close),
            )
        )

    return sum(true_ranges) / period
