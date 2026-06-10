from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    clean_prefix = prefix.lower().strip().replace(" ", "_")
    return f"{clean_prefix}_{uuid4().hex}"
