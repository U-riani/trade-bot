from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ms_to_datetime(value: int | float | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)
