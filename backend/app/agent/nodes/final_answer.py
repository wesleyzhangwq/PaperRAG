"""Final answer node: format output and build citation sources."""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.agent.state import AgentState
from app.models.paper import Paper
from app.schemas.chat import Source


_CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def final_answer_node(state: AgentState, *, db: Session) -> dict:
    """Extract citations from answer and build Source objects."""
    answer = state["final_answer"] or ""
    cited_ids = []
    for m in _CITATION_RE.finditer(answer):
        pid = m.group(1)
        if pid not in cited_ids:
            cited_ids.append(pid)

    sources = []
    for pid in cited_ids:
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        if paper:
            sources.append(Source(
                paper_id=pid,
                title=paper.title or "",
                authors=paper.authors or [],
                year=paper.year,
                primary_category=paper.primary_category,
                doi=paper.doi,
                arxiv_url=f"https://arxiv.org/abs/{pid}",
            ))

    return {"sources": sources}
