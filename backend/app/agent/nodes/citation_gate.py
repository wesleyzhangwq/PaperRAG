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

import time

from sqlalchemy.orm import Session

from app.agent.stages import stage
from app.agent.state import AgentState, StepTrace
from app.models.paper import Paper
from app.schemas.chat import Source
from app.utils.citations import extract_arxiv_ids, strip_disallowed_citations


def _is_visible(paper: Paper) -> bool:
    """Permission predicate. Single-user: everything visible. Extension point
    for multi-tenant ACL filtering (e.g. paper.owner_id checks)."""
    return True


def _paper_source_kind(paper: Paper) -> str:
    value = getattr(paper, "source_kind", None)
    return value if isinstance(value, str) and value else "arxiv"


def _paper_media_type(paper: Paper) -> str | None:
    value = getattr(paper, "media_type", None)
    return value if isinstance(value, str) and value else None


def citation_gate_node(state: AgentState, *, db: Session) -> dict:
    """Extract citations, resolve to Sources, strip unresolvable markers."""
    t0 = time.perf_counter()
    with stage("citation") as s:
        answer = state.get("final_answer") or ""
        synthesis_ids = state.get("synthesis_context_paper_ids")
        if synthesis_ids is not None:
            context_ids = {str(pid).strip() for pid in synthesis_ids if str(pid).strip()}
        else:
            # Compatibility fallback for checkpoints written before synthesis
            # context IDs were persisted.
            context_ids = {
                (d.metadata or {}).get("paper_id")
                for d in state.get("retrieval_context") or []
                if (d.metadata or {}).get("paper_id")
            }

        cited_ids = extract_arxiv_ids(answer)

        sources: list[Source] = []
        allowed_ids: set[str] = set()
        for pid in cited_ids:
            paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
            if pid in context_ids and paper is not None and _is_visible(paper):
                source_kind = _paper_source_kind(paper)
                allowed_ids.add(pid)
                sources.append(Source(
                    paper_id=pid,
                    title=paper.title or "",
                    authors=paper.authors or [],
                    year=paper.year,
                    primary_category=paper.primary_category,
                    source_kind=source_kind,
                    media_type=_paper_media_type(paper),
                    doi=paper.doi,
                    arxiv_url=(
                        f"https://arxiv.org/abs/{pid}"
                        if source_kind == "arxiv"
                        else None
                    ),
                ))
            elif pid in context_ids and paper is None:
                allowed_ids.add(pid)
                # In context (e.g. fresh arXiv API result) but not ingested into
                # MySQL yet — keep the citation, build a minimal source.
                sources.append(Source(
                    paper_id=pid,
                    title=_title_from_context(state, pid),
                    authors=[],
                    year=None,
                    primary_category=None,
                    source_kind="arxiv",
                    doi=None,
                    arxiv_url=f"https://arxiv.org/abs/{pid}",
                ))
            else:
                continue

        # Strip every unsupported syntax variant (bracketed, whitespace, URL)
        # through the same canonical parser used by groundedness and eval.
        answer, removed = strip_disallowed_citations(answer, allowed_ids)

        if removed:
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
