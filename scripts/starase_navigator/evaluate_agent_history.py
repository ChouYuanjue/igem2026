from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.starase_navigator.errors import AppError
from scripts.starase_navigator.serve import NavigatorRuntime


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = ROOT / "results/starase_navigator_runtime/run_events.jsonl"


def _read_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _user_text(row: dict[str, Any]) -> str:
    payload = row.get("input") if isinstance(row.get("input"), dict) else {}
    return str(payload.get("text") or payload.get("user_text") or "").strip()


def _error_code(row: dict[str, Any]) -> str:
    error = row.get("error") if isinstance(row.get("error"), dict) else {}
    return str(error.get("code") or error.get("type") or "").strip()


def historical_failure_cases(
    events: list[dict[str, Any]],
    *,
    error_codes: set[str],
    context_turns: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Build replay cases from real failed run_step events and their prior context."""
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        if str(row.get("event_type") or "") != "run_step":
            continue
        text = _user_text(row)
        session_id = str(row.get("session_id") or "").strip()
        if text and session_id:
            by_session[session_id].append(row)

    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for session_id, rows in by_session.items():
        for index, row in enumerate(rows):
            code = _error_code(row)
            if error_codes and code not in error_codes:
                continue
            if not code:
                continue
            start = max(0, index - max(0, int(context_turns)))
            turn_rows = rows[start : index + 1]
            turns: list[str] = []
            for candidate in turn_rows:
                text = _user_text(candidate)
                if text and (not turns or turns[-1] != text):
                    turns.append(text)
            if not turns:
                continue
            digest = hashlib.sha256(
                json.dumps(turns, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:16]
            if digest in seen:
                continue
            seen.add(digest)
            cases.append(
                {
                    "source_session_id": session_id,
                    "source_error_code": code,
                    "source_started_at_unix": row.get("started_at_unix"),
                    "turns": turns,
                    "case_id": f"history-{digest}",
                }
            )
            if limit > 0 and len(cases) >= limit:
                return cases
    return cases


def _result_summary(result: dict[str, Any]) -> dict[str, Any]:
    execution = (
        result.get("agent_execution")
        if isinstance(result.get("agent_execution"), dict)
        else {}
    )
    steps = [
        {
            "turn": row.get("turn"),
            "kind": row.get("action_kind"),
            "tool": row.get("tool"),
            "status": row.get("status"),
        }
        for row in execution.get("steps") or []
        if isinstance(row, dict)
    ]
    protein = (
        result.get("protein_resolution")
        if isinstance(result.get("protein_resolution"), dict)
        else {}
    )
    reaction = (
        result.get("reaction_resolution")
        if isinstance(result.get("reaction_resolution"), dict)
        else {}
    )
    return {
        "direction": str(result.get("direction") or ""),
        "response_type": str(result.get("response_type") or ""),
        "needs_user_input": bool(result.get("needs_user_input")),
        "protein_id": str(protein.get("recommended_id") or ""),
        "protein_mode": str(protein.get("mode") or ""),
        "reaction_id": str(reaction.get("recommended_id") or ""),
        "reaction_mode": str(reaction.get("mode") or ""),
        "retrieval_plan": dict(result.get("retrieval_plan") or {}),
        "reaction_constraints": dict(result.get("reaction_constraints") or {}),
        "agent_turn_count": int(execution.get("turn_count") or 0),
        "steps": steps,
        "summary": str(result.get("summary") or "")[:1200],
    }


def replay_case(
    runtime: NavigatorRuntime,
    case: dict[str, Any],
    *,
    ui_language: str,
) -> dict[str, Any]:
    session_id = f"eval-{case['case_id']}"
    outputs: list[dict[str, Any]] = []
    for turn_index, text in enumerate(case.get("turns") or [], start=1):
        started = time.perf_counter()
        try:
            result = runtime.agent_resolve(
                str(text),
                ui_language=ui_language,
                session_id=session_id,
            )
        except AppError as exc:
            outputs.append(
                {
                    "turn": turn_index,
                    "input": str(text),
                    "status": "error",
                    "error_code": str(exc.code or ""),
                    "error_detail": str(exc.detail or "")[:1200],
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            break
        outputs.append(
            {
                "turn": turn_index,
                "input": str(text),
                "status": "ok",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": _result_summary(result),
            }
        )
    return {
        **case,
        "replay_session_id": session_id,
        "replay_status": (
            "success"
            if outputs
            and all(row.get("status") == "ok" for row in outputs)
            else "failed"
        ),
        "outputs": outputs,
        "llm_provenance": runtime.deepseek.provenance(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay real Starase run_step failures with their prior conversational "
            "context. The corpus is derived from run_events.jsonl rather than a "
            "hand-written phrase list."
        )
    )
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument(
        "--error-code",
        action="append",
        default=["agent_turn_limit", "agent_repeated_tool_call"],
        help="Historical error code to select; repeat for multiple codes.",
    )
    parser.add_argument("--context-turns", type=int, default=3)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--ui-language", default="zh")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = historical_failure_cases(
        _read_events(args.events),
        error_codes={str(value) for value in args.error_code if str(value)},
        context_turns=max(0, args.context_turns),
        limit=max(0, args.limit),
    )
    if args.list_only:
        print(json.dumps(cases, ensure_ascii=False, indent=2))
        return

    runtime = NavigatorRuntime()
    results = [
        replay_case(runtime, case, ui_language=str(args.ui_language or "zh"))
        for case in cases
    ]
    report = {
        "schema": "starase-agent-history-replay-v1",
        "source": str(args.events),
        "case_count": len(results),
        "success_count": sum(
            row.get("replay_status") == "success" for row in results
        ),
        "failure_count": sum(
            row.get("replay_status") != "success" for row in results
        ),
        "cases": results,
        "llm_provenance": runtime.deepseek.provenance(),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
