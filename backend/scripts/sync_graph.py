"""Synchronize successfully ingested local papers into the Neo4j projection."""
from __future__ import annotations

import argparse
import json

from app.services.graph_sync import run_graph_sync


def main() -> int:
    parser = argparse.ArgumentParser(description="Synchronize Cite Scope's citation graph.")
    parser.add_argument("--all", action="store_true", help="Sync every successfully ingested paper.")
    parser.add_argument("--paper-id", action="append", default=[], help="Sync one arXiv ID; repeatable.")
    args = parser.parse_args()
    if args.all == bool(args.paper_id):
        parser.error("choose exactly one of --all or --paper-id")
    stats = run_graph_sync(None if args.all else args.paper_id)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
