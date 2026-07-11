"""Repair low-quality generated questions without changing the eval plan.

This keeps qid, expected paper IDs, difficulty, type, and tags stable while
regenerating questions whose text clearly came from deterministic fallbacks.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from eval.scripts.gen_questions import (  # noqa: E402
    COMPARISON_PROMPT,
    CONCEPT_LOCATE_PROMPT,
    FACT_EXTRACT_PROMPT,
    METHOD_DETAIL_PROMPT,
    TREND_PROMPT,
    _call_llm,
    _fallback_comparison_question,
    _fallback_reference,
    _fallback_single_paper_question,
    _fallback_trend_question,
    _get_llm,
    evidence_chunk_ids_for_papers,
    paper_evidence_excerpt,
)

FALLBACK_MARKERS = (
    "该问题对应论文《",
    "需要结合论文内容回答",
    "回答应分别说明两者关注的问题",
    "该问题需要综合分析",
    "在研究目标、技术路线和适用场景上有什么差异",
    "在实验设置、数据集、基线或结果方面给出了哪些具体事实",
    "采用了什么核心方法或模型结构来解决其研究问题",
    "这组论文体现出哪些共同趋势、方法差异和研究重点",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def is_low_quality_generated_question(question: dict) -> bool:
    if question.get("type") == "negative":
        return False
    text = f"{question.get('query') or ''} {question.get('reference_answer') or ''}"
    return any(marker in text for marker in FALLBACK_MARKERS)


def _paper_by_id(metadata_path: Path) -> dict[str, dict]:
    papers = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {str(p["paper_id"]): p for p in papers}


def _trend_topic(question: dict) -> str:
    query = str(question.get("query") or "")
    if query.startswith("围绕") and "，" in query:
        return query.split("，", 1)[0].removeprefix("围绕").strip()
    return "相关论文"


def _apply_result(question: dict, result: dict, *, fallback: tuple) -> dict:
    out = dict(question)
    generated_question = str(result.get("question") or "").strip()
    generated_answer = str(result.get("reference_answer") or "").strip()
    out["query"] = generated_question or fallback[0]
    out["reference_answer"] = generated_answer or fallback[1]
    out["generation_status"] = "llm" if generated_question and generated_answer else "fallback"
    return out


def repair_question(llm: ChatOpenAI, question: dict, papers: dict[str, dict]) -> dict:
    qtype = question.get("type")
    expected_ids = [str(pid) for pid in question.get("expected_paper_ids") or []]
    expected_papers = [papers[pid] for pid in expected_ids if pid in papers]
    if not expected_papers:
        return question

    if qtype == "concept_locate":
        paper = expected_papers[0]
        prompt = CONCEPT_LOCATE_PROMPT.format(
            paper_id=paper["paper_id"],
            title=paper["title"],
            category=paper["primary_category"],
            evidence=paper_evidence_excerpt(paper),
        )
        fallback = _fallback_single_paper_question(paper, "concept_locate")
    elif qtype == "method_detail":
        paper = expected_papers[0]
        prompt = METHOD_DETAIL_PROMPT.format(
            paper_id=paper["paper_id"],
            title=paper["title"],
            category=paper["primary_category"],
            evidence=paper_evidence_excerpt(paper),
        )
        fallback = _fallback_single_paper_question(paper, "method_detail")
    elif qtype == "fact_extract":
        paper = expected_papers[0]
        prompt = FACT_EXTRACT_PROMPT.format(
            paper_id=paper["paper_id"],
            title=paper["title"],
            category=paper["primary_category"],
            evidence=paper_evidence_excerpt(paper),
        )
        fallback = _fallback_single_paper_question(paper, "fact_extract")
    elif qtype == "comparison":
        if len(expected_papers) < 2:
            return question
        a, b = expected_papers[:2]
        prompt = COMPARISON_PROMPT.format(
            paper_id_a=a["paper_id"],
            title_a=a["title"],
            evidence_a=paper_evidence_excerpt(a, max_chars=3000),
            paper_id_b=b["paper_id"],
            title_b=b["title"],
            evidence_b=paper_evidence_excerpt(b, max_chars=3000),
        )
        fallback = _fallback_comparison_question(a, b)
    elif qtype == "trend_synthesis":
        topic = _trend_topic(question)
        papers_list = "\n".join(
            f"- [{p['paper_id']}] {p['title']}: {paper_evidence_excerpt(p, max_chars=700)}"
            for p in expected_papers[:8]
        )
        prompt = TREND_PROMPT.format(topic=topic, papers_list=papers_list)
        fallback = _fallback_trend_question(topic, expected_papers)
    else:
        return question

    try:
        result = _call_llm(llm, prompt, retries=3)
    except Exception as exc:
        print(f"  keep {question['qid']}: {exc}", file=sys.stderr)
        result = {}
    repaired = _apply_result(question, result, fallback=fallback)
    repaired["evidence_chunk_ids"] = evidence_chunk_ids_for_papers(expected_papers)
    return repaired


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair fallback-like eval questions")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sleep-sec", type=float, default=0.8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = load_jsonl(Path(args.input))
    papers = _paper_by_id(Path(args.metadata))
    targets = [
        idx for idx, question in enumerate(questions)
        if is_low_quality_generated_question(question)
    ]
    if args.limit is not None:
        targets = targets[: args.limit]

    print(f"Repair targets: {len(targets)}", file=sys.stderr)
    llm = _get_llm()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output, questions)

    repaired = 0
    remaining_low_quality = 0
    for target_no, idx in enumerate(targets, start=1):
        old = questions[idx]
        print(f"  Repair {target_no}/{len(targets)}: {old['qid']} {old['type']}", file=sys.stderr)
        new = repair_question(llm, old, papers)
        if new != old:
            repaired += 1
        if is_low_quality_generated_question(new):
            remaining_low_quality += 1
        questions[idx] = new
        write_jsonl_atomic(output, questions)
        time.sleep(args.sleep_sec)

    print(
        f"Repaired {repaired}/{len(targets)} questions; remaining low-quality markers: {remaining_low_quality}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
