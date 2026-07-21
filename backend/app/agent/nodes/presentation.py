"""Presentation node: transforms the raw agent state into a user-friendly,
productized payload for the UI.

Goals:
- Replace internal tool names with Chinese user-facing labels
- Compute a confidence verdict (high / medium / low) with a human reason
- Build clean source cards (relevance, summary, snippets) from `sources` + context
- Summarize retrieval (counts, main topics, fallback flag) in natural language
- Map each executed step to {name, status, user_message, debug}
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from sqlalchemy.orm import Session

from app.agent.stages import ACTION_LABELS
from app.agent.state import AgentState
from app.agent.telemetry import finalize_fallback_telemetry
from app.models.paper import Paper


_RETRIEVE_COUNT_RE = re.compile(r"found\s+(\d+)\s+chunks", re.IGNORECASE)
_FALLBACK_HINT_RE = re.compile(r"fallback", re.IGNORECASE)
_ARXIV_COUNT_RE = re.compile(r"(\d+)\s+papers", re.IGNORECASE)
_WEB_COUNT_RE = re.compile(r"(\d+)\s+results", re.IGNORECASE)


# Label mapping (internal → user-facing Chinese) — single source of truth
# lives in app.agent.stages; kept as a module attribute for compatibility.
STEP_LABELS = ACTION_LABELS


# ---------------------------------------------------------------------------
# Step → user-facing message + status mapping
# ---------------------------------------------------------------------------
def _step_user_message(action: str, detail: dict, summary: str) -> tuple[str, str]:
    """Return (user_message, status). status ∈ {completed, warning, error}."""
    if action == "retrieve_local":
        # Prefer detail (richer), fall back to parsing the trace's output_summary
        total = detail.get("total")
        if total is None:
            m = _RETRIEVE_COUNT_RE.search(summary or "")
            total = int(m.group(1)) if m else 0
        used_fallback = bool(detail.get("used_fallback")) or bool(_FALLBACK_HINT_RE.search(summary or ""))
        papers = len({h.get("paper_id") for h in (detail.get("hits") or []) if h.get("paper_id")})
        if total == 0:
            return "未找到匹配的文献片段。", "warning"
        msg = f"找到 {total} 个文献片段" + (f"，来自 {papers} 篇论文。" if papers else "。")
        if used_fallback:
            msg = "由于检索词不足，系统使用了扩展检索，" + msg + "结果可能不够精确。"
            return msg, "warning"
        return msg, "completed"

    if action == "evaluate_docs":
        if detail.get("parse_failed"):
            return (
                "资料充分性评估未成功解析，本次回答已降级为低可信度参考。",
                "warning",
            )
        if not detail.get("sufficient"):
            missing = detail.get("missing_aspects") or []
            extra = f"缺少：{ '、'.join(missing[:3]) }" if missing else ""
            return f"资料不足以完整回答问题。{extra}".strip(), "warning"
        return "资料基本足以支撑回答。", "completed"

    if action == "query_rewrite":
        qs = detail.get("queries") or []
        return f"将问题改写为 {len(qs)} 个检索关键词。", "completed"

    if action == "retrieve_arxiv":
        total = detail.get("total")
        if total is None:
            m = _ARXIV_COUNT_RE.search(summary or "")
            total = int(m.group(1)) if m else 0
        return f"从 arXiv 在线查询到 {total} 篇论文。", "completed" if total else "warning"

    if action == "search_web":
        total = detail.get("total")
        if total is None:
            m = _WEB_COUNT_RE.search(summary or "")
            total = int(m.group(1)) if m else 0
        return f"网络资料 {total} 条。", "completed" if total else "warning"

    if action == "reasoning_synthesis":
        return "已基于检索内容生成综合回答。", "completed"

    if action == "self_reflection":
        return "已完成自我校验。", "completed"

    if action == "intent_analysis":
        return "已理解你问题的意图与重点。", "completed"

    if action == "planning":
        return "已制定检索与生成计划。", "completed"

    if action == "re_planning":
        return "根据评估结果补充了检索步骤。", "completed"

    if action.startswith("get_paper_"):
        return "已提取该论文的相关内容。", "completed"

    # --- v2 orchestration stages ---
    if action == "guard":
        flags = detail.get("flags") or []
        if not detail.get("allowed", True):
            return f"问题未通过安全校验（{detail.get('reason', '')}）。", "warning"
        if flags:
            return "检测到可疑指令注入特征，已标记并继续。", "warning"
        return "安全校验通过。", "completed"

    if action == "route":
        labels = detail.get("source_labels") or []
        adjustments = detail.get("adjustments") or []
        msg = "检索来源：" + ("、".join(labels) if labels else "无")
        if adjustments:
            return msg + f"（路由策略调整 {len(adjustments)} 项）。", "completed"
        return msg + "。", "completed"

    if action == "evidence_process":
        before, after = detail.get("before", 0), detail.get("after", 0)
        if before == 0:
            return "没有可处理的证据片段。", "warning"
        return f"证据重排与压缩：{before} → {after} 个片段（{detail.get('papers', 0)} 篇论文）。", "completed"

    if action == "sufficiency_check":
        if detail.get("parse_failed"):
            return "充分性评估未成功解析，本次回答降级为低可信度参考。", "warning"
        if not detail.get("sufficient"):
            missing = detail.get("missing_aspects") or []
            extra = f"缺少:{'、'.join(str(m) for m in missing[:3])}" if missing else ""
            return f"证据不足以完整回答问题。{extra}".strip(), "warning"
        return "证据足以支撑回答。", "completed"

    if action == "groundedness_check":
        if detail.get("passed"):
            return "答案通过有据性校验（引用/完整性/逻辑）。", "completed"
        issues = detail.get("issues") or []
        return ("有据性校验未通过：" + (issues[0] if issues else "存在质量问题") + "。"), "warning"

    if action == "citation_gate":
        removed = detail.get("removed") or []
        resolved = detail.get("resolved")
        resolved_n = len(resolved) if isinstance(resolved, list) else int(resolved or 0)
        if removed:
            return f"解析 {resolved_n} 个引用，剔除 {len(removed)} 个无法核实的引用。", "warning"
        return f"解析并核实 {resolved_n} 个引用。" if resolved_n else "本次回答没有引用。", "completed"

    # Generic
    return summary or "已完成。", "completed"


# ---------------------------------------------------------------------------
# Source cards
# ---------------------------------------------------------------------------
def _build_source_cards(
    db: Session,
    sources_raw: list[dict],
    retrieval_context: list,
    max_cards: int = 5,
) -> list[dict]:
    """For each cited paper, build {paper_id, title, relevance, summary, snippets}.
    Pulls richer per-paper info from MySQL (Paper) when available."""
    # Group chunks per paper_id (for snippets / score)
    chunks_by_paper: dict[str, list[dict]] = defaultdict(list)
    for d in retrieval_context:
        md = d.metadata or {}
        pid = md.get("paper_id") or ""
        if not pid:
            continue
        chunks_by_paper[pid].append({
            "snippet": (d.page_content or "")[:240].strip(),
            "score": md.get("score"),
        })

    cards: list[dict] = []
    seen: set[str] = set()
    iter_list = sources_raw or [{"paper_id": pid} for pid in chunks_by_paper.keys()]
    for s in iter_list:
        pid = s.get("paper_id") or ""
        if not pid or pid in seen:
            continue
        seen.add(pid)
        # Fetch authoritative title/authors from Paper table when possible
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        title = (s.get("title") or (paper.title if paper else "")) or "未知论文"
        snippets = [c["snippet"] for c in chunks_by_paper.get(pid, []) if c["snippet"]][:2]
        # Relevance heuristic: number of hits in retrieval_context
        hit_count = len(chunks_by_paper.get(pid, []))
        if hit_count >= 3:
            relevance = "high"
        elif hit_count == 2:
            relevance = "medium"
        elif hit_count == 1:
            relevance = "low"
        else:
            relevance = "low"
        cards.append({
            "paper_id": pid,
            "title": title,
            "authors": (paper.authors if paper and paper.authors else s.get("authors") or [])[:5],
            "year": paper.year if paper else s.get("year"),
            "primary_category": paper.primary_category if paper else s.get("primary_category"),
            "arxiv_url": f"https://arxiv.org/abs/{pid}",
            "relevance": relevance,
            "hit_count": hit_count,
            "summary": (paper.abstract[:240] + "…") if (paper and paper.abstract) else "",
            "snippets": snippets,
        })
        if len(cards) >= max_cards:
            break
    return cards


# ---------------------------------------------------------------------------
# Retrieval summary
# ---------------------------------------------------------------------------
def _build_retrieval_summary(
    db: Session,
    retrieval_context: list,
    is_fallback: bool,
    cited_papers: int,
) -> dict:
    chunks = [d for d in retrieval_context if (d.metadata or {}).get("paper_id")]
    web_docs = [d for d in retrieval_context if (d.metadata or {}).get("source") == "web_search"]
    paper_ids = [(d.metadata or {}).get("paper_id") for d in chunks]
    unique_papers = sorted(set(p for p in paper_ids if p))
    # Pull primary_category as a simple topic proxy
    topics: Counter = Counter()
    titles: list[str] = []
    for pid in unique_papers[:10]:
        paper = db.query(Paper).filter(Paper.paper_id == pid).one_or_none()
        if paper:
            if paper.primary_category:
                topics[paper.primary_category] += 1
            if paper.title:
                titles.append(paper.title)

    main_topics = [t for t, _ in topics.most_common(3)]
    # Natural-language summary
    if not chunks:
        narrative = "本次未检索到匹配的文献片段。"
    else:
        top_title = titles[0] if titles else None
        narrative = (
            f"系统共检索到 {len(chunks)} 个文献片段，分布于 {len(unique_papers)} 篇论文"
            + (f"，主题集中在 {'、'.join(main_topics)}" if main_topics else "")
            + "。"
        )
        narrative += f"其中 {cited_papers} 篇被最终回答引用并展示为参考论文。"
        if len(unique_papers) > cited_papers:
            narrative += "未展示的论文是候选检索上下文，未被最终答案直接引用。"
        if top_title:
            narrative += f"其中与问题最相关的是《{top_title}》。"
        if web_docs:
            narrative += f"此外还补充查询了 {len(web_docs)} 条网络资料，用于弥补本地论文库不足。"
        if is_fallback:
            narrative += "由于问题较宽泛或检索词不足，系统使用了扩展检索，结果可能不够精确。"

    return {
        "total_chunks": len(chunks),
        "total_papers": len(unique_papers),
        "cited_papers": cited_papers,
        "web_results": len(web_docs),
        "main_topics": main_topics,
        "is_fallback": bool(is_fallback),
        "narrative": narrative,
    }


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------
def _compute_confidence(state: AgentState, sources_count: int) -> tuple[str, str]:
    """Return (level, reason). Level ∈ {high, medium, low}."""
    eval_result = state.get("sufficiency_result") or state.get("evaluator_result") or {}
    parse_failed = bool(state.get("evaluator_parse_failed") or eval_result.get("parse_failed"))
    sufficient = bool(eval_result.get("sufficient"))
    is_fallback = bool(state.get("is_fallback"))
    degraded = bool(state.get("degraded"))
    removed_citations = list(state.get("removed_citations") or [])
    reflection = state.get("reflection_result") or {}
    reflection_passed = bool(reflection.get("passed", True))

    if parse_failed:
        return "low", "资料充分性评估未成功解析，因此本次回答只作为低可信度参考。"
    if sources_count == 0:
        return "low", "未找到可引用的论文来源。"
    if degraded:
        return "low", "补充检索后证据仍不充分，回答已注明局限，请谨慎参考。"
    if is_fallback and not sufficient:
        return "low", "检索使用了扩展查询且资料不足，回答的可靠性较弱。"
    if not sufficient:
        return "low", "检索到的资料不足以完整回答问题。"
    if removed_citations:
        return "medium", f"回答中 {len(removed_citations)} 个引用未通过核实已被剔除，其余内容可信。"
    if is_fallback or not reflection_passed:
        return "medium", "检索结果相关但证据不够完整，建议谨慎使用。"
    if sources_count >= 2 and sufficient and reflection_passed:
        return "high", "检索结果高度相关且通过了有据性校验。"
    return "medium", "检索结果相关但证据有限。"


# ---------------------------------------------------------------------------
# Step list
# ---------------------------------------------------------------------------
def _build_steps_from_traces(state: AgentState) -> list[dict]:
    """Convert raw step_traces into user-facing step entries."""
    traces = state.get("step_traces") or []
    plan = state.get("plan") or []

    # Index detail/params by (action, position) — best-effort match using traces order
    out: list[dict] = []
    for i, trace in enumerate(traces):
        action = trace.get("action", "")
        name = STEP_LABELS.get(action, action)
        # Locate matching plan step (by index in executor traces) for params/reason
        params: dict[str, Any] = dict(trace.get("params") or {})
        reason = str(trace.get("reason") or "")
        # Heuristic: nth executor trace ≈ nth plan step
        _node_level_actions = (
            "intent_analysis", "planning", "reasoning_synthesis", "self_reflection",
            "re_planning", "guard", "route", "evidence_process", "sufficiency_check",
            "groundedness_check", "citation_gate",
        )
        if not params and action not in _node_level_actions:
            # try to find any plan step with same action; consume in order
            for ps in plan:
                if ps.get("action") == action:
                    params = dict(ps.get("params") or {})
                    reason = reason or ps.get("reason", "")
                    break

        # Recover detail from output_summary heuristics
        detail: dict[str, Any] = {
            "raw_summary": trace.get("output_summary", ""),
        }
        detail.update(trace.get("detail") or {})

        # For evaluate_docs: lift from state.evaluator_result
        if action == "evaluate_docs" and not trace.get("detail"):
            er = state.get("evaluator_result") or {}
            detail.update({
                "sufficient": er.get("sufficient"),
                "parse_failed": er.get("parse_failed", False),
                "missing_aspects": er.get("missing_aspects", []),
                "reason": er.get("reason", ""),
            })

        if action == "retrieve_local" and "used_fallback" not in detail:
            detail.update({
                "used_fallback": bool(state.get("is_fallback")),
            })

        user_message, status = _step_user_message(action, detail, trace.get("output_summary", ""))

        out.append({
            "index": i,
            "name": name,
            "action": action,
            "status": status,
            "user_message": user_message,
            "duration_ms": trace.get("duration_ms", 0),
            "debug": {
                "tool": action,
                "params": params,
                "reason": reason,
                "raw_summary": trace.get("output_summary", ""),
                "extra": detail,
            },
        })
    return out


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------
def presentation_node(state: AgentState, *, db: Session) -> dict:
    """Build the structured presentation payload at the end of the graph."""
    answer = state.get("final_answer") or ""
    sources_raw = state.get("sources") or []
    sources_data = [
        s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in sources_raw
    ]
    retrieval_context = state.get("retrieval_context") or []

    source_cards = _build_source_cards(db, sources_data, retrieval_context)
    retrieval_summary = _build_retrieval_summary(
        db, retrieval_context, bool(state.get("is_fallback")), len(source_cards)
    )
    confidence_level, confidence_reason = _compute_confidence(state, len(source_cards))
    steps = _build_steps_from_traces(state)
    telemetry = finalize_fallback_telemetry(state)
    guard_result = state.get("guard_result") or {}
    sufficiency = state.get("sufficiency_result") or {}
    if not guard_result.get("allowed", True):
        response_mode = "refuse"
    elif state.get("synthesis_failed") or sufficiency.get("parse_failed"):
        response_mode = "degraded"
    elif not sufficiency.get("sufficient", False) and (
        state.get("degraded") or not source_cards
    ):
        response_mode = "insufficient"
    else:
        response_mode = "answer"

    presentation = {
        "answer": answer,
        "confidence": confidence_level,
        "confidence_reason": confidence_reason,
        "sources": source_cards,
        "retrieval_summary": retrieval_summary,
        "steps": steps,
        "degraded": bool(state.get("degraded")),
        "removed_citations": list(state.get("removed_citations") or []),
        "fallback_telemetry": telemetry,
        "response_mode": response_mode,
    }
    return {"presentation": presentation, "fallback_telemetry": telemetry}
