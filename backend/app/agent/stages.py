"""Stage event helper — every node self-reports its lifecycle over SSE.

Each pipeline node wraps its work in ``stage(...)`` so the frontend receives
``stage`` events with STABLE ids (no index guessing):

    with stage("guard", "安全校验") as s:
        ...
        s.done("通过", detail={...})

Status vocabulary: start → done | warning | failed | skipped.
Outside a LangGraph run ``emit`` is a no-op, so nodes stay testable.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from app.agent.streaming import emit

# Pipeline stage keys → user-facing Chinese titles (single source of truth).
STAGE_TITLES = {
    "guard": "安全校验",
    "intent": "理解问题",
    "plan": "规划检索",
    "route": "选择检索来源",
    "retrieve": "多源检索",
    "evidence": "筛选与压缩证据",
    "sufficiency": "评估证据充分性",
    "synthesis": "生成回答",
    "groundedness": "校验答案有据",
    "citation": "整理引用",
    "presentation": "组织输出",
}

# Tool/step action names → user-facing Chinese labels (single source of truth,
# shared by executor step events and the presentation layer).
ACTION_LABELS = {
    # executor tool steps
    "query_rewrite": "优化检索关键词",
    "retrieve_local": "检索本地论文库",
    "retrieve_arxiv": "查询 arXiv 在线",
    "search_web": "查询网络资料",
    "get_paper_detail": "查询论文详情",
    "get_paper_chunks": "提取论文片段",
    # pipeline node trace actions (v2 orchestration)
    "guard": "安全校验",
    "route": "选择检索来源",
    "evidence_process": "筛选与压缩证据",
    "sufficiency_check": "评估证据充分性",
    "groundedness_check": "校验答案有据",
    "citation_gate": "整理引用",
    # legacy action names (persisted history compatibility)
    "intent_analysis": "理解你的问题",
    "planning": "制定检索计划",
    "evaluate_docs": "评估资料充分性",
    "reasoning_synthesis": "生成综合回答",
    "self_reflection": "自我校验回答",
    "re_planning": "补充检索计划",
}


def emit_plan(plan: list, revision: int) -> None:
    """Publish the full plan with stable per-step ids (id = plan index).

    Re-emitted whenever the plan changes (planner, route adjustments,
    re_planner) so the frontend can upsert steps by id without guessing.
    """
    emit("plan", {
        "revision": revision,
        "steps": [
            {
                "id": f"step:{i}",
                "action": s["action"],
                "title": ACTION_LABELS.get(s["action"], s["action"]),
                "reason": s.get("reason", ""),
            }
            for i, s in enumerate(plan)
        ],
    })


class StageReporter:
    """Context for one stage execution; call ``done``/``warning``/``failed``."""

    def __init__(self, stage_key: str, stage_id: str, title: str) -> None:
        self.stage = stage_key
        self.id = stage_id
        self.title = title
        self._t0 = time.perf_counter()
        self._closed = False

    def _emit(self, status: str, summary: str = "", detail: Optional[dict] = None) -> None:
        self._closed = True
        emit("stage", {
            "id": self.id,
            "stage": self.stage,
            "status": status,
            "title": self.title,
            "summary": summary,
            "detail": detail or {},
            "duration_ms": round((time.perf_counter() - self._t0) * 1000, 2),
        })

    def done(self, summary: str = "", detail: Optional[dict] = None) -> None:
        self._emit("done", summary, detail)

    def warning(self, summary: str = "", detail: Optional[dict] = None) -> None:
        self._emit("warning", summary, detail)

    def failed(self, summary: str = "", detail: Optional[dict] = None) -> None:
        self._emit("failed", summary, detail)

    def skipped(self, summary: str = "") -> None:
        self._emit("skipped", summary)

    @property
    def closed(self) -> bool:
        return self._closed


class stage:
    """Context manager emitting start on enter; auto-done on clean exit,
    auto-failed on exception (then re-raises)."""

    def __init__(self, stage_key: str, *, stage_id: str | None = None,
                 title: str | None = None, detail: Optional[dict] = None) -> None:
        self._key = stage_key
        self._id = stage_id or stage_key
        self._title = title or STAGE_TITLES.get(stage_key, stage_key)
        self._start_detail = detail

    def __enter__(self) -> StageReporter:
        emit("stage", {
            "id": self._id,
            "stage": self._key,
            "status": "start",
            "title": self._title,
            "detail": self._start_detail or {},
        })
        self._reporter = StageReporter(self._key, self._id, self._title)
        return self._reporter

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            if not self._reporter.closed:
                self._reporter.failed(f"{exc_type.__name__}: {exc}")
            return None  # re-raise
        if not self._reporter.closed:
            self._reporter.done()
        return None


__all__ = ["stage", "StageReporter", "STAGE_TITLES", "ACTION_LABELS", "emit_plan"]
