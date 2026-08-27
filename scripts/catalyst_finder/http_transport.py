from __future__ import annotations

import json
import mimetypes
import os
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from scripts.catalyst_finder.errors import AppError

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "frontend/catalyst_finder"

def _safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise AppError("bad_path", "Invalid path", HTTPStatus.BAD_REQUEST)
    return candidate


class Handler(BaseHTTPRequestHandler):
    runtime: Any = None
    max_body_bytes = 64 * 1024

    def version_string(self) -> str:
        return "CatalystFinder"

    def do_GET(self) -> None:  # noqa: N802
        self._handle_get(head_only=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_get(head_only=True)

    def _handle_get(self, *, head_only: bool) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path in {"/health", "/api/status"}:
                self._json(HTTPStatus.OK, self.runtime.status(), head_only=head_only)
                return
            if path == "/api/routes":
                self._json(HTTPStatus.OK, self.runtime._route_catalog, head_only=head_only)
                return
            if path in {"", "/"}:
                self._serve_file(STATIC_ROOT / "index.html", cache=False, head_only=head_only)
                return
            relative = unquote(path.lstrip("/"))
            candidate = _safe_path(STATIC_ROOT, relative)
            if candidate.is_file():
                self._serve_file(
                    candidate,
                    cache=path.startswith("/assets/"),
                    head_only=head_only,
                )
                return
            self._serve_file(STATIC_ROOT / "index.html", cache=False, head_only=head_only)
        except AppError as exc:
            self._error(exc)
        except Exception as exc:
            self._error(AppError("internal_error", "服务暂时不可用。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"))

    def _tracked_call(self, payload: dict[str, Any], event_type: str, operation: Any) -> Any:
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            return operation()
        started = time.time()
        metadata = {
            key: payload.get(key)
            for key in ("card_id", "card_title", "prompt_template", "prompt_source", "edited_after_card_click")
            if payload.get(key) is not None
        }
        event_input = dict(payload)
        for key in ("run_id", "session_id", "step_id", *metadata.keys()):
            event_input.pop(key, None)
        final_prompt = payload.get("text") or payload.get("user_text")
        if final_prompt:
            event_input["final_user_prompt"] = str(final_prompt)
        try:
            output = operation()
        except AppError as exc:
            finished = time.time()
            database_codes = {"rhea_no_match", "rhea_not_found", "rhea_smiles_missing", "rhea_unavailable"}
            step_status = "no_match" if exc.code in {"rhea_no_match", "rhea_not_found", "rhea_smiles_missing"} else "dependency_unavailable" if exc.code == "rhea_unavailable" else "validation_failed" if exc.status < 500 else "system_error"
            step_type = "database_verification" if exc.code in database_codes else event_type
            step_error = {"code": exc.code, "message": exc.message}
            step = {"step_id": str(payload.get("step_id") or ""), "step_type": step_type, "status": step_status, "input": event_input, "output": None, "error": step_error, "started_at_unix": started, "finished_at_unix": finished, "latency_ms": round((finished - started) * 1000, 2)}
            self.runtime.record_run_event(event_type="model_run", run_id=run_id, session_id=str(payload.get("session_id") or ""), status=step_status, input_data={**event_input, "steps": self.runtime.take_run_steps(run_id) + [step]}, error=step_error, started_at_unix=started, finished_at_unix=finished, metadata={**metadata, "failure_stage": step_type})
            raise
        except Exception as exc:
            finished = time.time()
            step = {"step_id": str(payload.get("step_id") or ""), "step_type": event_type, "status": "system_error", "input": event_input, "output": None, "error": {"type": type(exc).__name__, "message": str(exc)}, "started_at_unix": started, "finished_at_unix": finished, "latency_ms": round((finished - started) * 1000, 2)}
            self.runtime.record_run_event(event_type="model_run", run_id=run_id, session_id=str(payload.get("session_id") or ""), status="system_error", input_data={**event_input, "steps": self.runtime.take_run_steps(run_id) + [step]}, error=step["error"], started_at_unix=started, finished_at_unix=finished, metadata=metadata)
            raise
        finished = time.time()
        step = {"step_id": str(payload.get("step_id") or ""), "step_type": event_type, "status": "success", "input": event_input, "output": output, "error": None, "started_at_unix": started, "finished_at_unix": finished, "latency_ms": round((finished - started) * 1000, 2)}
        if event_type == "intent_and_entity_resolution":
            self.runtime.hold_run_step(run_id, step)
            return output
        self.runtime.record_run_event(event_type="model_run", run_id=run_id, session_id=str(payload.get("session_id") or ""), status="success", input_data={**event_input, "steps": self.runtime.take_run_steps(run_id) + [step]}, output_data=output, started_at_unix=started, finished_at_unix=finished, metadata=metadata)
        return output

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        try:
            payload = self._body()
            if parsed.path == "/api/run-events":
                event = self.runtime.record_run_event(
                    event_type=str(payload.get("event_type") or "user_event"),
                    run_id=str(payload.get("run_id") or ""),
                    session_id=str(payload.get("session_id") or ""),
                    step_id=str(payload.get("step_id") or ""),
                    status=str(payload.get("status") or "success"),
                    input_data=payload.get("input"),
                    output_data=payload.get("output"),
                    metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
                )
                self._json(HTTPStatus.CREATED, event)
                return
            if parsed.path == "/api/resolve":
                self._json(HTTPStatus.OK, self.runtime.resolve(str(payload.get("text") or "")))
                return
            if parsed.path == "/api/resolve-protein":
                self._json(HTTPStatus.OK, self.runtime.resolve_protein(str(payload.get("text") or "")))
                return
            if parsed.path == "/api/agent/resolve":
                self._json(
                    HTTPStatus.OK,
                    self._tracked_call(payload, "intent_and_entity_resolution", lambda: self.runtime.agent_resolve(
                        str(payload.get("text") or ""),
                        direction_hint=str(payload.get("direction_hint") or "auto"),
                        conversation_context=payload.get("conversation_context") if isinstance(payload.get("conversation_context"), dict) else {},
                        ui_language=str(payload.get("ui_language") or "en"),
                    )),
                )
                return
            if parsed.path == "/api/warmup/protein-encoder":
                self._json(
                    HTTPStatus.ACCEPTED,
                    self.runtime.prewarm_protein_encoder(background=True),
                )
                return
            if parsed.path == "/api/rank":
                self._json(
                    HTTPStatus.OK,
                    self._tracked_call(payload, "candidate_ranking", lambda: self.runtime.rank(
                        str(payload.get("rhea_id") or ""),
                        reaction_smiles=str(payload.get("reaction_smiles") or ""),
                        query_id=str(payload.get("query_id") or ""),
                        orientation=str(payload.get("orientation") or "forward"),
                        user_text=str(payload.get("user_text") or ""),
                        route_mode=str(payload.get("route_mode") or "intelligent"),
                        top_k=int(payload.get("top_k") or 10),
                        confirmed_seed_ids=[str(value) for value in (payload.get("confirmed_seed_ids") or [])],
                        confirmed_seed_inputs=[
                            dict(value)
                            for value in (payload.get("confirmed_seed_inputs") or [])
                            if isinstance(value, dict)
                        ],
                        conversation_context=payload.get("conversation_context") if isinstance(payload.get("conversation_context"), dict) else {},
                        ui_language=str(payload.get("ui_language") or "en"),
                    )),
                )
                return
            if parsed.path == "/api/rank-reactions":
                self._json(
                    HTTPStatus.OK,
                    self._tracked_call(payload, "reaction_ranking", lambda: self.runtime.rank_reactions(
                        str(payload.get("protein_id") or ""),
                        enzyme_sequence=str(payload.get("enzyme_sequence") or ""),
                        query_id=str(payload.get("query_id") or ""),
                        user_text=str(payload.get("user_text") or ""),
                        route_mode=str(payload.get("route_mode") or "intelligent"),
                        conversation_context=payload.get("conversation_context") if isinstance(payload.get("conversation_context"), dict) else {},
                        ui_language=str(payload.get("ui_language") or "en"),
                    )),
                )
                return
            if parsed.path == "/api/route/design":
                self._json(HTTPStatus.OK, self._tracked_call(payload, "route_design", lambda: self.runtime.design_routes(payload)))
                return
            if parsed.path == "/api/pathway/analyze":
                self._json(HTTPStatus.OK, self._tracked_call(payload, "pathway_analysis", lambda: self.runtime.analyze_pathway(payload)))
                return
            if parsed.path == "/api/feedback":
                self._json(HTTPStatus.CREATED, self.runtime.submit_feedback(payload))
                return
            raise AppError("not_found", "接口不存在。", HTTPStatus.NOT_FOUND)
        except AppError as exc:
            self._error(exc)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._error(AppError("invalid_request", "请求格式不正确。", HTTPStatus.BAD_REQUEST, str(exc)))
        except Exception as exc:
            self._error(AppError("internal_error", "服务暂时不可用。", HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.end_headers()

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > self.max_body_bytes:
            raise AppError("invalid_body", "请求内容为空或过大。", HTTPStatus.BAD_REQUEST)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise AppError("invalid_body", "请求必须是 JSON 对象。", HTTPStatus.BAD_REQUEST)
        return payload

    def _json(self, status: int, payload: Any, *, head_only: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _error(self, exc: AppError) -> None:
        payload = {"error": {"code": exc.code, "message": exc.message}}
        if exc.detail and os.environ.get("CATALYST_FINDER_DEBUG") == "1":
            payload["error"]["detail"] = exc.detail
        self._json(exc.status, payload)

    def _serve_file(self, path: Path, *, cache: bool, head_only: bool = False) -> None:
        if not path.is_file():
            raise AppError("not_found", "页面不存在。", HTTPStatus.NOT_FOUND)
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json", "image/svg+xml"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"catalyst-finder {self.address_string()} {fmt % args}", file=sys.stderr)
