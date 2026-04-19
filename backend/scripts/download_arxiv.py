"""
Phase 1: 从 arXiv 官方 API 拉取最新 cs.AI / cs.CL / cs.LG 领域论文 metadata，
下载 PDF 并产出 data/metadata_filtered.json。

用法：
    python scripts/download_arxiv.py --limit 50
    python scripts/download_arxiv.py --limit 500   # 扩量
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import arxiv
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
METADATA_JSON = DATA_DIR / "metadata_filtered.json"
FAILED_TXT = DATA_DIR / "failed.txt"

CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]
MIN_YEAR = 2023

UA = "PaperRAG/0.1 (https://github.com/; mailto:dev@paperrag.local)"
SLEEP_BETWEEN_DOWNLOADS = 3.0
HTTP_TIMEOUT = 60
MAX_RETRIES = 2


def build_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{c}" for c in categories)


def fetch_metadata(limit: int) -> list[dict]:
    """Query arXiv API, return list of paper metadata dicts."""
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=build_query(CATEGORIES),
        max_results=limit * 2,  # over-fetch to allow year filtering
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    records: list[dict] = []
    seen_ids: set[str] = set()
    print(f"[metadata] Querying arXiv: categories={CATEGORIES}, limit={limit}")

    for result in client.results(search):
        if len(records) >= limit:
            break
        year = result.published.year
        if year < MIN_YEAR:
            continue
        paper_id = result.get_short_id().split("v")[0]
        if paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)

        primary_cat = result.primary_category
        if primary_cat not in CATEGORIES and not any(c in CATEGORIES for c in result.categories):
            continue

        records.append({
            "paper_id": paper_id,
            "title": (result.title or "").strip().replace("\n", " "),
            "authors": [a.name for a in result.authors],
            "year": year,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat() if result.updated else None,
            "primary_category": primary_cat,
            "categories": list(result.categories),
            "doi": result.doi,
            "abstract": (result.summary or "").strip().replace("\n", " "),
            "pdf_url": result.pdf_url,
            "entry_id": result.entry_id,
        })

    print(f"[metadata] Collected {len(records)} papers (target={limit})")
    return records


def download_pdf(paper_id: str, pdf_url: str, out_path: Path) -> bool:
    if out_path.exists() and out_path.stat().st_size > 10_000:
        return True
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(pdf_url, headers={"User-Agent": UA}, timeout=HTTP_TIMEOUT, stream=True)
            r.raise_for_status()
            tmp = out_path.with_suffix(".pdf.part")
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            tmp.rename(out_path)
            return True
        except Exception as e:
            print(f"[pdf] {paper_id} attempt {attempt}/{MAX_RETRIES} failed: {e}")
            time.sleep(2 * attempt)
    return False


def download_pdfs(records: Iterable[dict]) -> tuple[list[dict], list[str]]:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    ok_records: list[dict] = []
    failed_ids: list[str] = []

    records = list(records)
    for rec in tqdm(records, desc="Downloading PDFs", unit="pdf"):
        paper_id = rec["paper_id"]
        out_path = PDF_DIR / f"{paper_id}.pdf"
        if download_pdf(paper_id, rec["pdf_url"], out_path):
            rec["pdf_path"] = str(out_path.relative_to(PROJECT_ROOT))
            ok_records.append(rec)
        else:
            failed_ids.append(paper_id)
        time.sleep(SLEEP_BETWEEN_DOWNLOADS)

    return ok_records, failed_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Download arXiv papers for PaperRAG.")
    parser.add_argument("--limit", type=int, default=50, help="Number of papers (default 50).")
    parser.add_argument("--skip-download", action="store_true", help="Only fetch metadata.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = fetch_metadata(limit=args.limit)
    if not records:
        print("[error] No metadata fetched.", file=sys.stderr)
        return 1

    if args.skip_download:
        METADATA_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2))
        print(f"[done] Metadata only → {METADATA_JSON}")
        return 0

    ok_records, failed = download_pdfs(records)

    METADATA_JSON.write_text(json.dumps(ok_records, ensure_ascii=False, indent=2))
    print(f"[done] {len(ok_records)} PDFs downloaded, metadata → {METADATA_JSON}")

    if failed:
        FAILED_TXT.write_text("\n".join(failed))
        print(f"[warn] {len(failed)} PDFs failed, see {FAILED_TXT}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
