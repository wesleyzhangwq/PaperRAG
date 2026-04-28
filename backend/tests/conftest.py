"""Pytest bootstrap: app modules read MySQL URL at import time."""
from __future__ import annotations

import os

# Avoid requiring a real MySQL server when running unit tests that use their own SQLite engine.
os.environ.setdefault("MYSQL_URL", "sqlite+pysqlite:///:memory:")
