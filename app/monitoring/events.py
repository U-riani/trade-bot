from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.utils.time import utc_now


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True, frozen=True)
class BotEvent:
    event_type: str
    severity: Severity
    message: str
    raw_data: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
