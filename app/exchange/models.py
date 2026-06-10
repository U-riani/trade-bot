from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ExchangeMessage:
    exchange: str
    raw: dict[str, Any]
