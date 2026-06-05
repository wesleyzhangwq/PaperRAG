"""CLI for PaperRAG weekly paper radar."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, time, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.weekly_radar import RadarConfig, run_weekly_radar  # noqa: E402


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = datetime.combine(parsed.date(), time.min, tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch, rank, report, and optionally ingest weekly papers.")
    parser.add_argument("--dry-run", action="store_true", help="Rank and write reports without PDF download or ingest.")
    parser.add_argument("--no-ingest", action="store_true", help="Rank and write reports only.")
    parser.add_argument("--top-k", type=int, default=None, help="Number of papers to select.")
    parser.add_argument("--window-days", type=int, default=None, help="Recent window size in days.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Maximum arXiv candidates to fetch before ranking.")
    parser.add_argument("--categories", type=str, default=None, help="Comma-separated arXiv categories.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Report output directory.")
    parser.add_argument("--since", type=str, default=None, help="Window start date/datetime, e.g. 2026-06-01.")
    parser.add_argument("--until", type=str, default=None, help="Window end date/datetime, e.g. 2026-06-08.")
    pdf_group = parser.add_mutually_exclusive_group()
    pdf_group.add_argument("--delete-pdfs", action="store_true", default=None, help="Delete PDFs after ingest.")
    pdf_group.add_argument("--keep-pdfs", action="store_false", dest="delete_pdfs", help="Keep PDFs after ingest.")
    args = parser.parse_args()

    config = RadarConfig.from_settings()
    overrides = {}
    if args.top_k is not None:
        overrides["top_k"] = args.top_k
    if args.window_days is not None:
        overrides["window_days"] = args.window_days
    if args.max_candidates is not None:
        overrides["max_candidates"] = args.max_candidates
    if args.categories:
        overrides["categories"] = tuple(part.strip() for part in args.categories.split(",") if part.strip())
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.delete_pdfs is not None:
        overrides["delete_pdfs_after_ingest"] = args.delete_pdfs
    if overrides:
        config = replace(config, **overrides)

    result = run_weekly_radar(
        config=config,
        dry_run=args.dry_run,
        no_ingest=args.no_ingest,
        window_start=_parse_date(args.since),
        window_end=_parse_date(args.until),
    )

    print(f"[weekly-radar] topic={result.topic_name}")
    print(f"[weekly-radar] window={result.window_start.isoformat()}..{result.window_end.isoformat()}")
    print(f"[weekly-radar] candidates={result.candidate_count} selected={len(result.selected)}")
    print(f"[weekly-radar] ingest_stats={result.ingest_stats}")
    if result.report_json:
        print(f"[weekly-radar] report_json={result.report_json}")
    if result.report_markdown:
        print(f"[weekly-radar] report_markdown={result.report_markdown}")

    if not result.selected:
        return 3
    return 2 if result.ingest_stats.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
