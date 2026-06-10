"""Citation gate node: resolve citations and strip unverifiable ones.

Maps to the enterprise pipeline's 引用与权限过滤 stage:

1. resolve   — every ``[arxiv:ID]`` in the answer → Source object from MySQL
2. filter    — citations that resolve to nothing (not in DB, not in context)
               are hallucinations that slipped past groundedness; the marker
               is stripped from the answer text and recorded
3. permission hook — single-user deployment grants all papers; the
               ``_is_visible`` predicate is the extension point for ACLs

The answer that leaves this node contains only citations the UI can prove.
"""
from __future__ import annotations

import re
import time

from sqlalchemy.orm import Session

from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace
from app.models.paper import Paper
from app.schemas.chat import Source

_CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def _is_visible(paper: Paper) -> bool:
    """Permission predicate. Single-user: everything visible. Extension point
    for multi-tenant ACL filtering (e.g. paper.owner_id checks)."""
    return True


def citation_gate_node(state: AgentState, *, db: Session) -> dict:
    """Extract citations, resolve to Sources, strip unresolvable markers."""
    t0 = time.perf_counter()
    with stage("citation") as s:
        answer = state.get("final_answer") or ""
        context_ids = {
            (d.metadata or {}).get("paper_id")
            for d in state.get("retrieval_context") or []
            if (d.metadata or {}).get("paper_id")
        }

        cited_ids: list[str] = []
        for m in _CITATION_RE.finditer(answer):
            pid = m.group(1)
            if pid not in cited_ids:
                cited_ids.append(pid)

        sources: list[Source] = []
        removed: list[str] = []
        for pid in cited_ids:
            paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
            if paper is not None and _is_visible(paper):
                sources.append(Source(
                    paper_id=pid,
                    title=paper.title or "",
                    authors=paper.authors or [],
                    year=paper.year,
                    primary_category=paper.primary_category,
                    doi=paper.doi,
                    arxiv_url=f"https://arxiv.org/abs/{pid}",
                ))
            elif pid in context_ids:
                # In context (e.g. fresh arXiv API result) but not ingested into
                # MySQL yet — keep the citation, build a minimal source.
                sources.append(Source(
                    paper_id=pid,
                    title=_title_from_context(state, pid),
                    authors=[],
                    year=None,
                    primary_category=None,
                    doi=None,
                    arxiv_url=f"https://arxiv.org/abs/{pid}",
                ))
            else:
                # Unresolvable citation: strip the marker from the answer.
                removed.append(pid)
                answer = answer.replace(f"[arxiv:{pid}]", "")

        if removed:
            answer = re.sub(r" {2,}", " ", answer)
            s.warning(
                f"解析 {len(sources)} 个引用，剔除 {len(removed)} 个无法核实的引用",
                detail={"resolved": len(sources), "removed": removed},
            )
        else:
            s.done(
                f"解析 {len(sources)} 个引用" if sources else "本次回答没有引用",
                detail={"resolved": len(sources), "removed": []},
            )

    duration = round((time.perf_counter() - t0) * 1000, 2)
    trace = StepTrace(
        node="citation_gate_node",
        action="citation_gate",
        input_summary=f"{len(cited_ids)} citation ids in answer",
        output_summary=f"resolved {len(sources)}, removed {len(removed)}",
        duration_ms=duration,
        detail={"resolved": [s_.paper_id for s_ in sources], "removed": removed},
    )

    out: dict = {
        "sources": sources,
        "removed_citations": removed,
        "step_traces": state["step_traces"] + [trace],
    }
    if removed:
        out["final_answer"] = answer
    return out


def _title_from_context(state: AgentState, paper_id: str) -> str:
    for d in state.get("retrieval_context") or []:
        md = d.metadata or {}
        if md.get("paper_id") == paper_id and md.get("title"):
            return str(md["title"])
    return ""


__all__ = ["citation_gate_node"]
