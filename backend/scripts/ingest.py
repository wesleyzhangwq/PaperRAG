"""Phase 3 entrypoint — parse PDFs, chunk, embed, persist.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --force           # re-embed everything
    python scripts/ingest.py --metadata path/to.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make "app" importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.ingest import run_ingest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=str, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    stats = run_ingest(metadata_json=args.metadata, force=args.force)
    return 0 if stats.get("failed", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
