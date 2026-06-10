from __future__ import annotations


def timeframe_to_seconds(timeframe: str) -> int:
    """Convert common exchange timeframe strings to seconds.

    Supported examples: 1m, 3m, 5m, 15m, 1h, 4h, 1d.
    """
    value = timeframe.strip().lower()
    if len(value) < 2:
        raise ValueError(f"Invalid timeframe: {timeframe!r}")

    unit = value[-1]
    amount_text = value[:-1]
    if not amount_text.isdigit():
        raise ValueError(f"Invalid timeframe amount: {timeframe!r}")

    amount = int(amount_text)
    if amount <= 0:
        raise ValueError(f"Timeframe amount must be positive: {timeframe!r}")

    if unit == "s":
        return amount
    if unit == "m":
        return amount * 60
    if unit == "h":
        return amount * 60 * 60
    if unit == "d":
        return amount * 24 * 60 * 60

    raise ValueError(f"Unsupported timeframe unit: {timeframe!r}")
