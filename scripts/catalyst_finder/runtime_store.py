from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from scripts.catalyst_finder.errors import AppError


class RuntimeStore:
    """Own private runtime telemetry, feedback persistence, and transient run state."""

    def __init__(self, *, feedback_path: Path, run_events_path: Path) -> None:
        self.feedback_path = feedback_path
        self.run_events_path = run_events_path
        self._feedback_lock = threading.Lock()
        self._run_events_lock = threading.Lock()
        self._pending_run_steps: dict[str, list[dict[str, Any]]] = {}
        self._pending_run_started: dict[str, float] = {}

    def record_run_event(
        self,
        *,
        event_type: str,
        run_id: str,
        session_id: str = "",
        step_id: str = "",
        status: str = "success",
        input_data: Any = None,
        output_data: Any = None,
        error: Any = None,
        started_at_unix: float | None = None,
        finished_at_unix: float | None = None,
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not run_id:
            raise AppError("run_id_required", "模型运行缺少 run_id。", HTTPStatus.BAD_REQUEST)
        started = started_at_unix or time.time()
        finished = finished_at_unix or time.time()
        record = {
            "event_id": hashlib.sha256(
                f"{time.time_ns()}|{run_id}|{event_type}".encode("utf-8")
            ).hexdigest()[:16],
            "event_type": event_type,
            "session_id": session_id,
            "run_id": run_id,
            "step_id": step_id,
            "status": status,
            "started_at_unix": started,
            "finished_at_unix": finished,
            "latency_ms": latency_ms
            if latency_ms is not None
            else round(max(0.0, finished - started) * 1000, 2),
            "input": input_data,
            "output": output_data,
            "error": error,
            "metadata": metadata or {},
        }
        self.run_events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._run_events_lock:
            with self.run_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            try:
                os.chmod(self.run_events_path, 0o600)
            except OSError:
                pass
        return {"ok": True, "event_id": record["event_id"]}

    def _prune_pending_run_steps(self, now: float | None = None) -> None:
        current = now or time.time()
        expired = [
            run_id
            for run_id, started in self._pending_run_started.items()
            if current - started > 3600
        ]
        for run_id in expired:
            self._pending_run_started.pop(run_id, None)
            self._pending_run_steps.pop(run_id, None)

    def hold_run_step(self, run_id: str, step: dict[str, Any]) -> None:
        with self._run_events_lock:
            self._prune_pending_run_steps()
            if run_id not in self._pending_run_steps and len(self._pending_run_steps) >= 256:
                oldest_run_id = min(
                    self._pending_run_started,
                    key=self._pending_run_started.get,
                    default=None,
                )
                if oldest_run_id is not None:
                    self._pending_run_started.pop(oldest_run_id, None)
                    self._pending_run_steps.pop(oldest_run_id, None)
            self._pending_run_started.setdefault(run_id, time.time())
            steps = self._pending_run_steps.setdefault(run_id, [])
            steps.append(step)
            if len(steps) > 8:
                del steps[:-8]

    def take_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        with self._run_events_lock:
            self._prune_pending_run_steps()
            self._pending_run_started.pop(run_id, None)
            return self._pending_run_steps.pop(run_id, [])

    def submit_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        rating = str(payload.get("rating") or "").strip()
        category = str(payload.get("category") or "other").strip()
        message = str(payload.get("message") or "").strip()
        contact = str(payload.get("contact") or "").strip()
        if rating not in {"helpful", "neutral", "needs_improvement", ""}:
            raise AppError(
                "feedback_invalid_rating",
                "请选择有效的使用感受。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        if category not in {"results", "interaction", "database", "route", "other"}:
            category = "other"
        if not rating and not message:
            raise AppError(
                "feedback_empty",
                "请至少选择一个使用感受，或写下你的意见。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        if len(message) > 3000:
            raise AppError(
                "feedback_too_long",
                "反馈内容请控制在 3000 字以内。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        if len(contact) > 200:
            raise AppError(
                "feedback_contact_too_long",
                "联系方式过长。",
                HTTPStatus.UNPROCESSABLE_ENTITY,
            )
        raw_context = payload.get("context") if isinstance(payload.get("context"), dict) else {}
        context: dict[str, str] = {}
        for key in ("direction", "target", "route_id", "result_mode", "task_summary"):
            value = str(raw_context.get(key) or "").strip()
            if value:
                context[key] = value[:500]
        now = time.time()
        feedback_id = hashlib.sha256(
            f"{time.time_ns()}|{rating}|{category}|{message}".encode("utf-8")
        ).hexdigest()[:14]
        record = {
            "feedback_id": feedback_id,
            "submitted_at_unix": now,
            "rating": rating or None,
            "category": category,
            "message": message,
            "contact": contact or None,
            "context": context,
        }
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self._feedback_lock:
            with self.feedback_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                os.chmod(self.feedback_path, 0o600)
            except OSError:
                pass
        return {"ok": True, "feedback_id": feedback_id}
