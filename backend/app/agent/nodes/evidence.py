"""Evidence processing node: filter → rerank → compress retrieved context.

Maps to the enterprise pipeline's 证据过滤/重排/压缩 stage. Runs after all
retrieval steps complete, before the sufficiency check:

1. rerank   — scored chunks (local hybrid retrieval) sorted desc; unscored
              docs (arXiv/web) keep insertion order after the scored block
2. cap      — at most ``MAX_CHUNKS_PER_PAPER`` chunks per paper, preventing
              a single paper from dominating the context window
3. compress — total character budget; lowest-value chunks dropped first

Deterministic by design (no LLM): cheap, reproducible, unit-testable.
"""
from __future__ import annotations

import time
from collections import defaultdict

from langchain_core.documents import Document

from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace

MAX_CHUNKS_PER_PAPER = 4
CONTEXT_CHAR_BUDGET = 14000


def _score(doc: Document) -> float | None:
    raw = (doc.metadata or {}).get("score")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def evidence_node(state: AgentState) -> dict:
    """Rerank, cap and compress ``retrieval_context`` in place."""
    t0 = time.perf_counter()
    with stage("evidence") as s:
        docs = list(state.get("retrieval_context") or [])
        before = len(docs)

        # 1) rerank: scored first (desc), unscored keep insertion order after
        scored = [(d, _score(d)) for d in docs]
        ranked = sorted(
            (pair for pair in scored if pair[1] is not None),
            key=lambda pair: pair[1],
            reverse=True,
        ) + [pair for pair in scored if pair[1] is None]

        # 2) per-paper cap
        per_paper: dict[str, int] = defaultdict(int)
        capped: list[Document] = []
        dropped_cap = 0
        for doc, _ in ranked:
            pid = (doc.metadata or {}).get("paper_id") or ""
            if pid:
                if per_paper[pid] >= MAX_CHUNKS_PER_PAPER:
                    dropped_cap += 1
                    continue
                per_paper[pid] += 1
            capped.append(doc)

        # 3) char budget compression (keep highest-ranked within budget)
        kept: list[Document] = []
        dropped_budget = 0
        used_chars = 0
        for doc in capped:
            chars = len(doc.page_content or "")
            if used_chars + chars > CONTEXT_CHAR_BUDGET and kept:
                dropped_budget += 1
                continue
            kept.append(doc)
            used_chars += chars

        papers = len({(d.metadata or {}).get("paper_id") for d in kept if (d.metadata or {}).get("paper_id")})
        evidence_stats = {
            "before": before,
            "after": len(kept),
            "dropped_cap": dropped_cap,
            "dropped_budget": dropped_budget,
            "papers": papers,
            "context_chars": used_chars,
        }

        if before == 0:
            s.warning("没有检索到任何证据", detail=evidence_stats)
        else:
            s.done(
                f"保留 {len(kept)}/{before} 个片段（{papers} 篇论文，{used_chars} 字符）",
                detail=evidence_stats,
            )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="evidence_node",
        action="evidence_process",
        input_summary=f"{before} chunks in",
        output_summary=f"kept {len(kept)}, dropped cap={dropped_cap} budget={dropped_budget}",
        duration_ms=duration,
        detail=evidence_stats,
    )
    return {
        "retrieval_context": kept,
        "evidence_stats": evidence_stats,
        "step_traces": state["step_traces"] + [trace],
    }


__all__ = ["evidence_node", "MAX_CHUNKS_PER_PAPER", "CONTEXT_CHAR_BUDGET"]
