"""Export a bounded, evidence-backed paper corpus from MySQL for RAG evals."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
SECTION_SIGNALS = (
    "abstract",
    "introduction",
    "method",
    "methodology",
    "approach",
    "architecture",
    "experiment",
    "evaluation",
    "result",
    "conclusion",
    "摘要",
    "引言",
    "方法",
    "实验",
    "结果",
    "结论",
)
LOW_VALUE_SIGNALS = ("references", "bibliography", "acknowledgement", "appendix")


def normalize_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    return SECRET_RE.sub("[REDACTED_SECRET]", normalized)


def _value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _chunk_record(chunk: object, *, text_chars: int) -> dict:
    return {
        "chunk_id": str(_value(chunk, "chunk_id") or ""),
        "paper_id": str(_value(chunk, "paper_id") or ""),
        "chunk_index": int(_value(chunk, "chunk_index", 0) or 0),
        "page_num": _value(chunk, "page_num"),
        "text": normalize_text(str(_value(chunk, "text", _value(chunk, "chunk_text", ""))))[
            : max(1, int(text_chars))
        ],
    }


def _representative_score(chunk: dict) -> tuple[int, int, str]:
    text = str(chunk.get("text") or "").lower()
    index = int(chunk.get("chunk_index") or 0)
    score = max(0, 45 - index * 4)
    if index == 0:
        score += 80
    score += 60 * sum(1 for signal in SECTION_SIGNALS if signal in text)
    score -= 100 * sum(1 for signal in LOW_VALUE_SIGNALS if signal in text)
    score += min(len(text) // 200, 10)
    return score, -index, str(chunk.get("chunk_id") or "")


def select_representative_chunks(
    chunks: Iterable[object],
    *,
    limit: int = 8,
    text_chars: int = 1400,
) -> list[dict]:
    records = [
        _chunk_record(chunk, text_chars=text_chars)
        for chunk in chunks
    ]
    records = [record for record in records if record["chunk_id"] and record["text"]]
    ranked = sorted(records, key=_representative_score, reverse=True)
    selected = ranked[: max(1, int(limit))]
    return sorted(selected, key=lambda item: (item["chunk_index"], item["chunk_id"]))


def build_paper_record(
    paper: object,
    chunks: Iterable[object],
    *,
    evidence_limit: int = 8,
    text_chars: int = 1400,
    max_evidence_chars: int = 8000,
) -> dict:
    paper_id = str(_value(paper, "paper_id") or "")
    chunk_list = list(chunks)
    for chunk in chunk_list:
        chunk_paper_id = str(_value(chunk, "paper_id") or "")
        if chunk_paper_id != paper_id:
            raise ValueError(f"chunk {str(_value(chunk, 'chunk_id') or '')} does not belong to {paper_id}")

    evidence_chunks = select_representative_chunks(
        chunk_list,
        limit=evidence_limit,
        text_chars=text_chars,
    )
    blocks = [
        f"[{item['chunk_id']}] {item['text']}"
        for item in evidence_chunks
    ]
    evidence_text = "\n\n".join(blocks)[: max(1, int(max_evidence_chars))]
    return {
        "paper_id": paper_id,
        "title": normalize_text(str(_value(paper, "title") or "")),
        "authors": _value(paper, "authors") or [],
        "year": int(_value(paper, "year", 0) or 0),
        "primary_category": str(_value(paper, "primary_category") or ""),
        "categories": [str(value) for value in (_value(paper, "categories") or [])],
        "doi": _value(paper, "doi"),
        "abstract": normalize_text(str(_value(paper, "abstract") or "")),
        "chunk_count": len(chunk_list),
        "evidence_chunks": evidence_chunks,
        "evidence_text": evidence_text,
    }


def export_mysql_corpus(
    *,
    evidence_limit: int = 8,
    text_chars: int = 1400,
    max_evidence_chars: int = 8000,
) -> list[dict]:
    from app.db.mysql import SessionLocal, init_db
    from app.models.paper import Chunk, Paper

    init_db()
    db = SessionLocal()
    try:
        papers = (
            db.query(Paper)
            .filter(Paper.ingest_status == "ok", Paper.num_chunks > 0)
            .order_by(Paper.paper_id.asc())
            .all()
        )
        chunks_by_paper: dict[str, list[object]] = defaultdict(list)
        for chunk in db.query(Chunk).order_by(Chunk.paper_id, Chunk.chunk_index).all():
            chunks_by_paper[str(chunk.paper_id)].append(chunk)
        return [
            build_paper_record(
                paper,
                chunks_by_paper.get(str(paper.paper_id), []),
                evidence_limit=evidence_limit,
                text_chars=text_chars,
                max_evidence_chars=max_evidence_chars,
            )
            for paper in papers
        ]
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the ready MySQL paper corpus for evals.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-limit", type=int, default=8)
    parser.add_argument("--text-chars", type=int, default=1400)
    parser.add_argument("--max-evidence-chars", type=int, default=8000)
    args = parser.parse_args()

    records = export_mysql_corpus(
        evidence_limit=args.evidence_limit,
        text_chars=args.text_chars,
        max_evidence_chars=args.max_evidence_chars,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "papers": len(records),
        "chunks": sum(int(record["chunk_count"]) for record in records),
        "evidence_chunks": sum(len(record["evidence_chunks"]) for record in records),
        "output": str(args.output),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
