"""UTC time helpers for database timestamps.

MySQL ``DATETIME`` columns in this project store UTC values without timezone
metadata.  Build those values from a timezone-aware clock, then deliberately
drop the timezone before persistence.  This keeps the existing schema contract
while avoiding the deprecated ``datetime.utcnow()`` API.
"""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC time as a naive value for MySQL ``DATETIME``."""
    return datetime.now(timezone.utc).replace(tzinfo=None)
