"""Retry failed rows in an existing traditional-vs-agentic RAG evaluation."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

from eval.agentic_compare_metrics import (  # noqa: E402
    answer_case_metrics,
    compare_system_summaries,
    select_proportional_questions,
    select_stratified_questions,
    summarize_answer_cases,
)
from eval.run_agentic_rag_eval import (  # noqa: E402
    _paired_rows,
    _write_csv,
    _write_jsonl,
    render_report,
    run_agentic_case,
    run_traditional_case,
)
from eval.run_rag_eval import load_questions  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _failed_qids(rows: list[dict]) -> list[str]:
    return [str(row["qid"]) for row in rows if row.get("error")]


def recompute_case_rows(rows: list[dict]) -> list[dict]:
    """Refresh persisted per-question metrics after a scoring-rule update."""
    updated: list[dict] = []
    for raw in rows:
        row = dict(raw)
        metrics = answer_case_metrics(
            qid=str(row["qid"]),
            qtype=str(row.get("type") or "unknown"),
            difficulty=str(row.get("difficulty") or "unknown"),
            expected_paper_ids=list(row.get("expected_pids") or []),
            expected_mode=str(row.get("expected_mode") or "answer"),
            answer=str(row.get("answer_for_metrics") or row.get("answer") or ""),
            source_pids=list(row.get("source_pids") or []),
            context_pids=list(row.get("context_pids") or row.get("source_pids") or []),
            latency_s=float(row.get("latency_s") or 0.0),
            used_chunks=int(row.get("used_chunks") or 0),
            step_count=int(row.get("step_count") or 0),
            retrieval_step_count=int(row.get("retrieval_step_count") or 0),
            error=row.get("error"),
        )
        row.update(metrics)
        updated.append(row)
    return updated


def retry_failed_rows(
    rows: list[dict],
    *,
    rerun_case: Callable[[str, int], dict],
    max_attempts: int,
    sleep_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[list[dict], list[dict]]:
    """Retry only failed rows and replace successful results without reordering."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    replacements: dict[str, dict] = {}
    repair_rows: list[dict] = []
    for qid in _failed_qids(rows):
        last: dict | None = None
        for attempt in range(1, max_attempts + 1):
            candidate = dict(rerun_case(qid, attempt))
            if candidate.get("qid") != qid:
                raise ValueError(f"Retry returned qid {candidate.get('qid')!r} for {qid!r}")
            candidate["repair_attempts"] = attempt
            last = candidate
            if not candidate.get("error"):
                break
            if attempt < max_attempts and sleep_seconds:
                sleep_fn(sleep_seconds * attempt)

        assert last is not None
        replacements[qid] = last
        repair_rows.append(
            {
                "qid": qid,
                "attempts": last["repair_attempts"],
                "recovered": not bool(last.get("error")),
                "final_error": last.get("error"),
            }
        )

    updated = [dict(replacements.get(str(row["qid"]), row)) for row in rows]
    return updated, repair_rows


def _selected_questions(manifest: dict) -> list[dict]:
    questions = load_questions(Path(manifest["dataset"]))
    selection = manifest["selection"]
    if selection.get("sample_size") is not None:
        selected = select_proportional_questions(
            questions,
            sample_size=int(selection["sample_size"]),
        )
    else:
        selected = select_stratified_questions(
            questions,
            per_type=selection.get("per_type"),
        )
    if selection.get("limit") is not None:
        selected = selected[: int(selection["limit"])]
    if len(selected) != manifest["question_count"]:
        raise ValueError(
            "Reconstructed question selection differs from the original run: "
            f"expected {manifest['question_count']}, got {len(selected)}"
        )
    return selected


def _backup_outputs(run_dir: Path, stamp: str) -> list[str]:
    names = (
        "traditional_per_question.jsonl",
        "traditional_per_question.csv",
        "agentic_per_question.jsonl",
        "agentic_per_question.csv",
        "paired_per_question.jsonl",
        "paired_per_question.csv",
        "summary.json",
        "manifest.json",
        "report.md",
    )
    backups: list[str] = []
    for name in names:
        path = run_dir / name
        if path.exists():
            backup = run_dir / f"{name}.before-repair-{stamp}"
            shutil.copy2(path, backup)
            backups.append(backup.name)
    return backups


def _write_outputs(
    *,
    run_dir: Path,
    manifest: dict,
    traditional_rows: list[dict],
    agentic_rows: list[dict],
    repair_log: dict,
) -> tuple[dict, list[dict]]:
    traditional_rows = recompute_case_rows(traditional_rows)
    agentic_rows = recompute_case_rows(agentic_rows)
    paired = _paired_rows(traditional_rows, agentic_rows)
    summaries = {
        "traditional_rag": summarize_answer_cases(traditional_rows),
        "agentic_rag": summarize_answer_cases(agentic_rows),
        "elapsed_initial_s": repair_log["elapsed_initial_s"],
        "repair_elapsed_s": repair_log["elapsed_s"],
        "elapsed_total_s": round(
            repair_log["elapsed_initial_s"] + repair_log["elapsed_s"],
            4,
        ),
    }
    comparisons = compare_system_summaries(
        baseline=summaries["traditional_rag"],
        candidate=summaries["agentic_rag"],
    )

    _write_jsonl(run_dir / "traditional_per_question.jsonl", traditional_rows)
    _write_jsonl(run_dir / "agentic_per_question.jsonl", agentic_rows)
    _write_jsonl(run_dir / "paired_per_question.jsonl", paired)
    _write_csv(run_dir / "traditional_per_question.csv", traditional_rows)
    _write_csv(run_dir / "agentic_per_question.csv", agentic_rows)
    _write_csv(run_dir / "paired_per_question.csv", paired)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "summaries": summaries,
                "comparisons": comparisons,
                "repair": repair_log,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = render_report(
        run_id=manifest["run_id"],
        manifest=manifest,
        summaries=summaries,
        comparisons=comparisons,
    )
    report += (
        "\n\n## Repair\n"
        f"- Retried failed rows at {repair_log['finished_at']} (max attempts: {repair_log['max_attempts']}).\n"
        f"- Remaining errors: traditional={repair_log['remaining_errors']['traditional_rag']}, "
        f"agentic={repair_log['remaining_errors']['agentic_rag']}."
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return summaries, comparisons


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry only failed rows in an Agentic RAG comparison run."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=8.0)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found in {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    questions = _selected_questions(manifest)
    qid_to_item = {str(item["qid"]): item for item in questions}
    qid_to_idx = {qid: idx for idx, qid in enumerate(qid_to_item, start=1)}

    traditional_rows = _load_jsonl(run_dir / "traditional_per_question.jsonl")
    agentic_rows = _load_jsonl(run_dir / "agentic_per_question.jsonl")
    failed_traditional = _failed_qids(traditional_rows)
    failed_agentic = _failed_qids(agentic_rows)
    missing_qids = set(failed_traditional + failed_agentic) - set(qid_to_item)
    if missing_qids:
        raise ValueError(f"Failed rows are not in reconstructed selection: {sorted(missing_qids)}")

    previous_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    elapsed_initial_s = float(previous_summary["summaries"].get("elapsed_total_s", 0.0))
    started = datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    backups = _backup_outputs(run_dir, stamp)
    repair_t0 = time.perf_counter()

    traditional_cfg = manifest["traditional"]

    def rerun_traditional(qid: str, attempt: int) -> dict:
        return run_traditional_case(
            qid_to_item[qid],
            idx=qid_to_idx[qid],
            total=len(questions),
            context_k=int(traditional_cfg["context_k"]),
            retrieval_top_k=traditional_cfg["retrieval_top_k"],
            context_strategy=traditional_cfg["context_strategy"],
            mmr_lambda=float(traditional_cfg["mmr_lambda"]),
        )

    traditional_rows, traditional_repair = retry_failed_rows(
        traditional_rows,
        rerun_case=rerun_traditional,
        max_attempts=args.max_attempts,
        sleep_seconds=args.sleep_sec,
    )

    agentic_repair: list[dict] = []
    if failed_agentic:
        from app.db.mysql import SessionLocal

        db = SessionLocal()
        try:
            def rerun_agentic(qid: str, attempt: int) -> dict:
                return run_agentic_case(
                    qid_to_item[qid],
                    idx=qid_to_idx[qid],
                    total=len(questions),
                    run_id=f"{manifest['run_id']}-repair-{attempt}",
                    db=db,
                    local_only=bool(manifest.get("agentic", {}).get("local_only", True)),
                )

            agentic_rows, agentic_repair = retry_failed_rows(
                agentic_rows,
                rerun_case=rerun_agentic,
                max_attempts=args.max_attempts,
                sleep_seconds=args.sleep_sec,
            )
        finally:
            db.close()

    elapsed_s = round(time.perf_counter() - repair_t0, 4)
    repair_log = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "max_attempts": args.max_attempts,
        "sleep_seconds": args.sleep_sec,
        "elapsed_initial_s": elapsed_initial_s,
        "elapsed_s": elapsed_s,
        "backups": backups,
        "traditional_rag": traditional_repair,
        "agentic_rag": agentic_repair,
        "remaining_errors": {
            "traditional_rag": len(_failed_qids(traditional_rows)),
            "agentic_rag": len(_failed_qids(agentic_rows)),
        },
    }
    manifest.setdefault("repairs", []).append(repair_log)
    _write_outputs(
        run_dir=run_dir,
        manifest=manifest,
        traditional_rows=traditional_rows,
        agentic_rows=agentic_rows,
        repair_log=repair_log,
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(repair_log, ensure_ascii=False, indent=2))
    print(f"Repaired comparison run in {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
