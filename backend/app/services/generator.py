"""RAG chain: inspired by P4_RAG项目案例/rag.py — compose prompt → Ollama → parse → citations.

We keep the chain deliberately simple (no history in MVP), but the LangChain
pipe structure mirrors the reference implementation.
"""
from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.paper import Paper
from app.schemas.chat import ChatFilter, ChatRequest, ChatResponse, Source
from app.services.retriever import retrieve

settings = get_settings()


SYSTEM_PROMPT = """你是一个严谨的学术论文问答助手。请严格基于“已知参考资料”回答用户问题。

硬性规则：
1. 回答必须基于下方参考资料；若资料不足以回答，请如实说“参考资料不足以回答该问题”。
2. 每一条论据/结论末尾必须追加对应 paper_id 作为引用，格式为 `[arxiv:PAPER_ID]`；多个来源用 `[arxiv:ID1][arxiv:ID2]`。
3. 禁止编造 paper_id；只能使用下方参考资料中出现过的 paper_id。
4. 输出使用简洁的 Markdown，中文作答（除非用户用英文提问）。
"""


USER_TEMPLATE = """参考资料（每段前有 [arxiv:PAPER_ID | title | page=N]）：

{context}

用户问题：{query}

请作答，并严格按上述规则在每条论据后附加 [arxiv:PAPER_ID]。"""


_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_TEMPLATE),
])


def _format_context(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        md = d.metadata or {}
        header = (
            f"[arxiv:{md.get('paper_id','?')} | {md.get('title','')[:120]} | "
            f"page={md.get('page_num', '?')}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


_CITATION_RE = re.compile(r"\[arxiv:([0-9]{4}\.[0-9]{4,6})\]")


def _extract_cited_ids(text: str) -> list[str]:
    seen: list[str] = []
    for m in _CITATION_RE.finditer(text):
        pid = m.group(1)
        if pid not in seen:
            seen.append(pid)
    return seen


def _build_sources(
    db: Session,
    docs_scores: list[tuple[Document, float]],
    cited_ids: list[str],
    trimmed_ids: set[str],
) -> list[Source]:
    """Prefer cited papers; fallback to all retrieved if LLM cited none."""
    seen: set[str] = set()
    by_pid: dict[str, tuple[Document, float]] = {}
    for d, s in docs_scores:
        pid = (d.metadata or {}).get("paper_id")
        if pid and pid in trimmed_ids and pid not in by_pid:
            by_pid[pid] = (d, s)

    ordered_pids: list[str] = []
    for pid in cited_ids:
        if pid in by_pid and pid not in seen:
            ordered_pids.append(pid)
            seen.add(pid)
    if not ordered_pids:
        ordered_pids = [pid for pid in by_pid.keys()]

    sources: list[Source] = []
    for pid in ordered_pids:
        d, score = by_pid[pid]
        md = d.metadata or {}
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        sources.append(Source(
            paper_id=pid,
            title=(paper.title if paper else md.get("title", "")) or "",
            authors=(paper.authors if paper and paper.authors else []) or [],
            year=(paper.year if paper else md.get("year")) or None,
            primary_category=(paper.primary_category if paper else md.get("primary_category")),
            doi=(paper.doi if paper else md.get("doi")) or None,
            arxiv_url=f"https://arxiv.org/abs/{pid}",
            score=float(score) if score is not None else None,
            page_num=md.get("page_num"),
            chunk_index=md.get("chunk_index"),
            snippet=(d.page_content or "")[:280],
        ))
    return sources


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )


def run_chat(db: Session, req: ChatRequest) -> ChatResponse:
    flt: Optional[ChatFilter] = req.filter
    top_k = req.top_k or settings.retrieval_k
    final_k = req.final_k or settings.final_context_k

    docs_scores = retrieve(req.query, flt=flt, top_k=top_k)
    if not docs_scores:
        return ChatResponse(
            answer="参考资料不足以回答该问题（未检索到相关论文片段）。",
            sources=[],
            used_chunks=0,
        )

    # Trim: final_k chunks for prompt, but keep full retrieval for source building
    trimmed = docs_scores[:final_k]
    trimmed_ids = {(d.metadata or {}).get("paper_id") for d, _ in trimmed}
    context = _format_context([d for d, _ in trimmed])

    llm = _get_llm()
    chain = _prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": req.query, "context": context})

    cited_ids = _extract_cited_ids(answer)
    sources = _build_sources(db, docs_scores, cited_ids, trimmed_ids)

    return ChatResponse(answer=answer, sources=sources, used_chunks=len(trimmed))
