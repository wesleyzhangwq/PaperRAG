"""Run a pure RAG evaluation without the LangGraph agent loop.

The runner measures:
- retrieval ranking quality,
- fixed top-k context quality,
- optional fixed-context answer generation quality.

It deliberately does not evaluate planning, routing, reflection, or other agent
framework stages.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from eval.rag_metrics import (  # noqa: E402
    compare_metric,
    evaluate_retrieval_case,
    render_markdown_report,
    summarize_retrieval_cases,
)

PLACEHOLDER_QUERIES = {"<question text>", "<question>", "..."}
CITATION_RE = re.compile(r"(?:arxiv:)?([0-9]{4}\.[0-9]{4,6})", re.IGNORECASE)
SECRET_RE = re.compile(r"sk-[A-Za-z0-9_-]{10,}")
TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
ABSTENTION_TERMS = (
    "信息不足",
    "无法回答",
    "无法确定",
    "没有足够",
    "未提供",
    "不在语料",
    "not enough information",
    "insufficient information",
    "cannot answer",
    "no evidence",
)


@dataclass
class EvalDocument:
    page_content: str
    metadata: dict


def _is_placeholder_question(item: dict) -> bool:
    query = str(item.get("query") or "").strip()
    reference = str(item.get("reference_answer") or "").strip()
    return (
        not query
        or query in PLACEHOLDER_QUERIES
        or query.startswith("<question")
        or reference in {"<reference answer>", "..."}
    )


def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if _is_placeholder_question(item):
            continue
        rows.append(item)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No valid questions in {path}")
    return rows


def redact_sensitive_text(text: str) -> str:
    return SECRET_RE.sub("[REDACTED_SECRET]", text or "")


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def _paper_text(paper: dict) -> str:
    evidence_text = " ".join(
        str(chunk.get("text") or "")
        for chunk in (paper.get("evidence_chunks") or [])
        if isinstance(chunk, dict)
    )
    return " ".join(
        str(bit or "")
        for bit in (
            paper.get("title"),
            paper.get("abstract"),
            evidence_text,
            paper.get("primary_category"),
            " ".join(paper.get("categories") or []),
        )
    )


def load_lexical_corpus(path: Path) -> list[dict]:
    papers = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(papers, list):
        raise ValueError(f"Lexical corpus must be a JSON list: {path}")
    return [paper for paper in papers if paper.get("paper_id") and _paper_text(paper).strip()]


def lexical_paper_retrieve(
    query: str,
    papers: list[dict],
    *,
    top_k: int,
) -> list[tuple[EvalDocument, float]]:
    from rank_bm25 import BM25Okapi

    if top_k < 1:
        return []
    corpus_tokens = [_tokenize(_paper_text(paper)) for paper in papers]
    query_tokens = _tokenize(query)
    if not papers or not query_tokens or all(not tokens for tokens in corpus_tokens):
        return []

    bm25 = BM25Okapi(corpus_tokens)
    scores = list(bm25.get_scores(query_tokens))
    ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)
    results: list[tuple[EvalDocument, float]] = []
    for idx, score in ranked[:top_k]:
        paper = papers[idx]
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        evidence = "\n".join(
            str(chunk.get("text") or "")
            for chunk in (paper.get("evidence_chunks") or [])
            if isinstance(chunk, dict) and chunk.get("text")
        )
        content = redact_sensitive_text(f"{title}\n{abstract}\n{evidence}".strip())
        metadata = {
            "paper_id": str(paper.get("paper_id") or ""),
            "title": title,
            "chunk_id": f"{paper.get('paper_id')}:paper",
            "page": None,
        }
        results.append((EvalDocument(page_content=content, metadata=metadata), float(score)))
    return results


def build_retrieved_chunks(
    results: Iterable[tuple[object, float]],
    *,
    snippet_chars: int = 420,
) -> list[dict]:
    chunks: list[dict] = []
    for idx, (doc, score) in enumerate(results, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        content = redact_sensitive_text(str(getattr(doc, "page_content", "") or ""))
        chunks.append(
            {
                "rank": idx,
                "paper_id": str(metadata.get("paper_id") or ""),
                "title": str(metadata.get("title") or metadata.get("paper_title") or ""),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id") or "",
                "page": metadata.get("page"),
                "score": round(float(score), 6),
                "snippet": content[:snippet_chars],
            }
        )
    return chunks


def _reset_ranks(chunks: list[dict]) -> list[dict]:
    out = []
    for idx, chunk in enumerate(chunks, start=1):
        item = dict(chunk)
        item["rank"] = idx
        out.append(item)
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalized_scores(chunks: list[dict]) -> dict[int, float]:
    scores = [float(chunk.get("score") or 0.0) for chunk in chunks]
    if not scores:
        return {}
    mn = min(scores)
    mx = max(scores)
    if abs(mx - mn) < 1e-12:
        return {idx: 0.5 for idx in range(len(chunks))}
    return {idx: (score - mn) / (mx - mn) for idx, score in enumerate(scores)}


def postprocess_chunks(
    chunks: list[dict],
    *,
    strategy: str,
    context_k: int,
    mmr_lambda: float = 0.65,
) -> list[dict]:
    if strategy == "raw":
        return _reset_ranks(chunks)
    if context_k < 1:
        return []
    if strategy == "paper_dedup":
        seen: set[str] = set()
        selected: list[dict] = []
        for chunk in chunks:
            pid = str(chunk.get("paper_id") or "")
            if not pid or pid in seen:
                continue
            selected.append(chunk)
            seen.add(pid)
            if len(selected) >= context_k:
                break
        return _reset_ranks(selected)
    if strategy != "mmr_dedup":
        raise ValueError(f"Unknown context strategy: {strategy}")

    relevance = _normalized_scores(chunks)
    token_sets = [
        set(_tokenize(f"{chunk.get('title') or ''} {chunk.get('snippet') or ''}"))
        for chunk in chunks
    ]
    selected_indices: list[int] = []
    selected_pids: set[str] = set()
    candidates = list(range(len(chunks)))
    lam = max(0.0, min(1.0, float(mmr_lambda)))
    while candidates and len(selected_indices) < context_k:
        best_idx = None
        best_score = None
        for idx in candidates:
            pid = str(chunks[idx].get("paper_id") or "")
            if pid in selected_pids:
                continue
            redundancy = max(
                (_jaccard(token_sets[idx], token_sets[j]) for j in selected_indices),
                default=0.0,
            )
            score = lam * relevance.get(idx, 0.0) - (1.0 - lam) * redundancy
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        selected_pids.add(str(chunks[best_idx].get("paper_id") or ""))
        candidates.remove(best_idx)
    return _reset_ranks([chunks[idx] for idx in selected_indices])


def extract_citation_pids(answer: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in CITATION_RE.findall(answer or ""):
        if match not in seen:
            found.append(match)
            seen.add(match)
    return found


def detect_abstention(answer: str) -> bool:
    text = (answer or "").lower()
    return any(term.lower() in text for term in ABSTENTION_TERMS)


def evaluate_generation_case(
    *,
    answer: str,
    expected_paper_ids: list[str],
    expected_mode: str,
    context_pids: list[str],
) -> dict:
    expected = {str(pid) for pid in expected_paper_ids if str(pid).strip()}
    context = {str(pid) for pid in context_pids if str(pid).strip()}
    citations = extract_citation_pids(answer)
    cited = set(citations)
    abstained = detect_abstention(answer)

    mode_correct = abstained if expected_mode in {"insufficient", "refuse"} else not abstained
    row: dict = {
        "answer_abstained": abstained,
        "mode_correct": bool(mode_correct),
        "citation_pids": citations,
        "citation_count": len(citations),
    }
    if not citations:
        row["citation_support_rate"] = None
        row["citation_precision"] = None
        row["citation_expected_hit"] = 0.0 if expected else None
        return row

    supported = cited & context
    row["citation_support_rate"] = len(supported) / len(cited)
    if expected:
        row["citation_precision"] = len(cited & expected) / len(cited)
        row["citation_expected_hit"] = 1.0 if cited & expected else 0.0
    else:
        row["citation_precision"] = None
        row["citation_expected_hit"] = None
    return row


def rows_from_retrieval_detail(
    detail: dict,
    *,
    k_values: list[int],
    context_k: int,
) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(detail.get("per_question") or [], start=1):
        predicted = item.get("predicted_pids") or item.get("ranked_pids") or []
        chunks = [
            {
                "rank": rank,
                "paper_id": str(pid),
                "title": "",
                "chunk_id": "",
                "page": None,
                "score": None,
                "snippet": "",
            }
            for rank, pid in enumerate(predicted, start=1)
        ]
        row = evaluate_retrieval_case(
            qid=item.get("qid") or f"q{idx}",
            query=item.get("query") or "",
            expected_paper_ids=item.get("expected_pids")
            or item.get("expected_paper_ids")
            or [],
            expected_mode=item.get("expected_mode") or ("insufficient" if item.get("is_negative") else "answer"),
            difficulty=item.get("difficulty", "unknown"),
            qtype=item.get("type", "unknown"),
            retrieved_chunks=chunks,
            k_values=k_values,
            context_k=context_k,
            latency_s=item.get("latency") or item.get("latency_s"),
        )
        legacy_metric_map = {
            "ndcg": "ndcg_at_5",
            "precision": "precision_at_5",
            "recall": "recall_at_5",
            "rr": "mrr",
        }
        for source_key, target_key in legacy_metric_map.items():
            if item.get(source_key) is not None:
                row[target_key] = item[source_key]
        row["retrieved_chunks"] = chunks
        rows.append(row)
    return rows


def _context_pids(chunks: list[dict], context_k: int) -> list[str]:
    return [
        str(chunk.get("paper_id") or "")
        for chunk in chunks[:context_k]
        if str(chunk.get("paper_id") or "")
    ]


def _build_context(chunks: list[dict], context_k: int) -> str:
    blocks = []
    for chunk in chunks[:context_k]:
        pid = chunk.get("paper_id") or "unknown"
        title = chunk.get("title") or "untitled"
        snippet = redact_sensitive_text(chunk.get("snippet") or "")
        blocks.append(f"[arxiv:{pid}] {title}\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def generate_fixed_context_answer(query: str, context: str) -> str:
    from langchain_openai import ChatOpenAI

    from app.core.config import get_settings

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_api_base,
        api_key=settings.llm_api_key,
        temperature=0.0,
        max_retries=2,
    )
    prompt = f"""You are evaluating a pure RAG system.

Answer the user question using only the retrieved context. Cite supporting
papers with [arxiv:paper_id]. If the context is insufficient, say in Chinese:
"信息不足，无法回答该问题。"

Question:
{query}

Retrieved context:
{context}
"""
    response = llm.invoke(prompt)
    return str(response.content or "").strip()


def _summarize_generation(rows: list[dict]) -> dict:
    generated = [row for row in rows if "answer_abstained" in row]
    if not generated:
        return {}

    def avg(key: str) -> float | None:
        values = [row[key] for row in generated if row.get(key) is not None]
        if not values:
            return None
        return round(sum(float(v) for v in values) / len(values), 4)

    return {
        "generation_count": len(generated),
        "mode_accuracy": avg("mode_correct"),
        "citation_support_rate": avg("citation_support_rate"),
        "citation_precision": avg("citation_precision"),
        "citation_expected_hit": avg("citation_expected_hit"),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and not isinstance(row.get(key), (list, dict)):
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_baseline_summary(path: str | None) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "summary" in payload:
        return payload["summary"]
    return payload


def _comparison_rows(baseline: dict | None, candidate: dict) -> list[dict]:
    if not baseline:
        return []
    keys = [
        "ndcg_at_5",
        "recall_at_5",
        "mrr",
        "context_chunk_precision",
        "context_recall",
        "negative_max_score_mean",
        "latency_p90",
    ]
    return [
        compare_metric(key, baseline=baseline.get(key), candidate=candidate.get(key))
        for key in keys
    ]


def _render_graph_gate_report(gates: dict | None) -> str:
    if not gates:
        return ""
    lines = ["", "## Graph RAG Merge Gates", "", f"- Overall: {'PASS' if gates['passed'] else 'FAIL'}", ""]
    lines.extend(["| Gate | Baseline | Candidate | Threshold | Pass |", "|---|---:|---:|---:|:---:|"])
    for check in gates["checks"]:
        baseline = "n/a" if check.get("baseline") is None else f"{check['baseline']:.4f}"
        candidate = "n/a" if check.get("candidate") is None else f"{check['candidate']:.4f}"
        lines.append(
            f"| {check['name']} | {baseline} | {candidate} | {check['threshold']:.4f} | {'PASS' if check['passed'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def _settings_manifest() -> dict:
    from app.core.config import get_settings

    settings = get_settings()
    return {
        "retrieval_k": settings.retrieval_k,
        "final_context_k": settings.final_context_k,
        "hybrid_retrieval_enabled": settings.hybrid_retrieval_enabled,
        "hybrid_alpha": settings.hybrid_alpha,
        "hybrid_oversample": settings.hybrid_oversample,
        "hybrid_max_fetch": settings.hybrid_max_fetch,
        "cache_retrieval_enabled": settings.cache_retrieval_enabled,
        "graph_rag_enabled": settings.graph_rag_enabled,
        "graph_seed_papers": settings.graph_seed_papers,
        "graph_max_hops": settings.graph_max_hops,
        "graph_candidate_limit": settings.graph_candidate_limit,
        "graph_query_timeout_ms": settings.graph_query_timeout_ms,
        "env_overrides": {
            key: os.environ.get(key)
            for key in (
                "RETRIEVAL_K",
                "FINAL_CONTEXT_K",
                "HYBRID_RETRIEVAL_ENABLED",
                "HYBRID_ALPHA",
                "HYBRID_OVERSAMPLE",
                "HYBRID_MAX_FETCH",
            )
            if os.environ.get(key) is not None
        },
    }


def run_pure_rag_eval(
    *,
    questions: list[dict],
    k_values: list[int],
    context_k: int,
    retrieval_top_k: int | None,
    generate: bool,
    graph_expansion_top_k: int | None = None,
    retriever_name: str = "service",
    lexical_papers: list[dict] | None = None,
    context_strategy: str = "raw",
    mmr_lambda: float = 0.65,
) -> list[dict]:
    if retriever_name == "service":
        from app.services.retriever import retrieve
    elif retriever_name == "service_graph":
        from app.core.config import get_settings
        from eval.graph_rag_eval import retrieve_service_graph

        if not get_settings().graph_rag_enabled:
            raise ValueError("service_graph requires GRAPH_RAG_ENABLED=true")
    elif retriever_name != "lexical_paper":
        raise ValueError(f"Unknown retriever: {retriever_name}")
    elif lexical_papers is None:
        raise ValueError("lexical_paper retriever requires lexical_papers")

    rows: list[dict] = []
    for idx, item in enumerate(questions, start=1):
        query = item["query"]
        qid = item.get("qid", f"q{idx}")
        print(f"[{idx}/{len(questions)}] {qid}: retrieving ({retriever_name})", file=sys.stderr)

        t0 = time.perf_counter()
        graph_outcome = None
        if retriever_name == "service":
            results = retrieve(query, top_k=retrieval_top_k)
        elif retriever_name == "service_graph":
            settings = get_settings()
            seed_top_k = retrieval_top_k or settings.retrieval_k
            expansion_top_k = graph_expansion_top_k or seed_top_k
            graph_outcome = retrieve_service_graph(
                query,
                seed_top_k=seed_top_k,
                expansion_top_k=expansion_top_k,
            )
            results = graph_outcome.results
        else:
            results = lexical_paper_retrieve(
                query,
                lexical_papers or [],
                top_k=retrieval_top_k or context_k,
            )
        latency_s = time.perf_counter() - t0
        chunks = postprocess_chunks(
            build_retrieved_chunks(results),
            strategy=context_strategy,
            context_k=context_k,
            mmr_lambda=mmr_lambda,
        )

        row = evaluate_retrieval_case(
            qid=qid,
            query=query,
            expected_paper_ids=item.get("expected_paper_ids") or [],
            expected_mode=item.get("expected_mode", "answer"),
            difficulty=item.get("difficulty", "unknown"),
            qtype=item.get("type", "unknown"),
            retrieved_chunks=chunks,
            k_values=k_values,
            context_k=context_k,
            latency_s=latency_s,
        )
        row["retrieved_chunks"] = chunks
        row["reference_answer"] = item.get("reference_answer", "")
        if graph_outcome is not None:
            row.update(
                {
                    "graph_expansion_ms": graph_outcome.graph_expansion_ms,
                    "graph_fallback_reason": graph_outcome.graph_fallback_reason,
                    "graph_candidate_count": graph_outcome.graph_candidate_count,
                }
            )

        if generate:
            context = _build_context(chunks, context_k)
            answer_t0 = time.perf_counter()
            answer = generate_fixed_context_answer(query, context)
            row["generation_latency_s"] = round(time.perf_counter() - answer_t0, 4)
            answer = redact_sensitive_text(answer)
            row["answer"] = answer
            row.update(
                evaluate_generation_case(
                    answer=answer,
                    expected_paper_ids=item.get("expected_paper_ids") or [],
                    expected_mode=item.get("expected_mode", "answer"),
                    context_pids=_context_pids(chunks, context_k),
                )
            )

        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pure RAG evaluation.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "eval/datasets/questions_v2.jsonl"),
    )
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "eval/results/rag"),
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument(
        "--graph-expansion-top-k",
        type=int,
        default=None,
        help="Qdrant chunk count for the production graph second pass; defaults to --retrieval-top-k.",
    )
    parser.add_argument(
        "--retriever",
        choices=["service", "service_graph", "lexical_paper"],
        default="service",
        help="Retrieval backend: local service, production Graph RAG path, or rough paper-level BM25.",
    )
    parser.add_argument(
        "--lexical-corpus",
        type=str,
        default=str(PROJECT_ROOT / "eval/datasets/qdrant_papers_100_20260708.json"),
        help="Paper metadata JSON used by --retriever lexical_paper.",
    )
    parser.add_argument(
        "--context-strategy",
        choices=["raw", "paper_dedup", "mmr_dedup"],
        default="raw",
        help="Post-process retrieved chunks before metric/context evaluation.",
    )
    parser.add_argument("--mmr-lambda", type=float, default=0.65)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--compare-summary", type=str, default=None)
    parser.add_argument(
        "--from-detail-json",
        type=str,
        default=None,
        help="Replay an existing retrieval detail JSON instead of calling Qdrant.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    run_id = args.run_id or f"rag-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    questions: list[dict] = []
    detail_source: str | None = None
    if args.from_detail_json:
        detail_source = str(Path(args.from_detail_json))
        detail = json.loads(Path(args.from_detail_json).read_text(encoding="utf-8"))
        rows = rows_from_retrieval_detail(
            detail,
            k_values=args.k_values,
            context_k=args.context_k,
        )
        if args.limit is not None:
            rows = rows[: args.limit]
        print(f"Loaded {len(rows)} rows from {detail_source}", file=sys.stderr)
    else:
        questions = load_questions(dataset_path, limit=args.limit)
        print(f"Loaded {len(questions)} valid questions from {dataset_path}", file=sys.stderr)
        if args.retriever == "service_graph":
            from eval.graph_rag_eval import require_graph_corpus_coverage

            coverage = require_graph_corpus_coverage(questions)
            print(f"Graph corpus preflight: {coverage}", file=sys.stderr)

    lexical_papers = None
    if args.retriever == "lexical_paper" and not args.from_detail_json:
        lexical_papers = load_lexical_corpus(Path(args.lexical_corpus))
        print(f"Loaded {len(lexical_papers)} lexical papers from {args.lexical_corpus}", file=sys.stderr)

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    if not args.from_detail_json:
        rows = run_pure_rag_eval(
            questions=questions,
            k_values=args.k_values,
            context_k=args.context_k,
            retrieval_top_k=args.retrieval_top_k,
            generate=args.generate,
            graph_expansion_top_k=args.graph_expansion_top_k,
            retriever_name=args.retriever,
            lexical_papers=lexical_papers,
            context_strategy=args.context_strategy,
            mmr_lambda=args.mmr_lambda,
        )
    elapsed_s = time.perf_counter() - t0

    summary = summarize_retrieval_cases(rows, args.k_values)
    summary.update(_summarize_generation(rows))
    summary["elapsed_total_s"] = round(elapsed_s, 4)

    baseline_summary = _load_baseline_summary(args.compare_summary)
    comparisons = _comparison_rows(baseline_summary, summary)
    graph_gates = None
    if args.retriever == "service_graph":
        from eval.graph_rag_eval import evaluate_graph_merge_gates, summarize_graph_expansion

        summary.update(summarize_graph_expansion(rows))
        if baseline_summary:
            graph_gates = evaluate_graph_merge_gates(baseline_summary, summary)
    manifest = {
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "detail_source": detail_source,
        "question_count": len(rows) if args.from_detail_json else len(questions),
        "k_values": args.k_values,
        "context_k": args.context_k,
        "retrieval_top_k": args.retrieval_top_k,
        "graph_expansion_top_k": args.graph_expansion_top_k,
        "retriever": args.retriever,
        "lexical_corpus": str(Path(args.lexical_corpus)) if args.retriever == "lexical_paper" else None,
        "context_strategy": args.context_strategy,
        "mmr_lambda": args.mmr_lambda,
        "generate": args.generate,
        "settings": _settings_manifest(),
    }

    _write_jsonl(run_dir / "per_question.jsonl", rows)
    _write_csv(run_dir / "per_question.csv", rows)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {"summary": summary, "comparisons": comparisons, "graph_gates": graph_gates},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = render_markdown_report(
            run_id=run_id,
            dataset_name=dataset_path.name,
            summary=summary,
            comparisons=comparisons,
        ) + _render_graph_gate_report(graph_gates)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    print(f"Saved pure RAG eval to {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
