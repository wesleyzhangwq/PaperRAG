"""Pytest defaults so app modules can import without a real MySQL .env."""
from __future__ import annotations

import os

# app.db.mysql reads Settings at import time; tests use sqlite fixtures instead.
os.environ.setdefault("MYSQL_HOST", "127.0.0.1")
