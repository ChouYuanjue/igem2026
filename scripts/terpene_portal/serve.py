from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.database_bridge import DatabaseBridge  # noqa: E402
from scripts.database_bridge.model_catalog import ModelDataCatalog  # noqa: E402

PORTAL_DIST = ROOT / "frontend/terpene_portal/dist"
DATABASE_ROOT = ROOT / "external_repos/igem_database"
DATABASE_DIST = DATABASE_ROOT / "frontend/dist"
DATABASE_COMMIT = "87b507908441bb858ebdffb88a04682dafa99e11"
DATABASE_ORIGIN = "git@github.com:Yifan-Jia123/igem_database.git"


class PortalRuntime:
    def __init__(self) -> None:
        self.database = DatabaseBridge()
        self._engine: Any | None = None
        self._engine_lock = threading.Lock()
        self._catalog: ModelDataCatalog | None = None
        self._catalog_lock = threading.Lock()

    def engine(self) -> Any:
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    from projects.active.terpene_screening.core.engine import RetrievalEngine
                    self._engine = RetrievalEngine(allow_overrides=False)
        return self._engine

    def catalog(self) -> ModelDataCatalog:
        if self._catalog is None:
            with self._catalog_lock:
                if self._catalog is None:
                    self._catalog = ModelDataCatalog(ROOT)
        return self._catalog

    def status(self) -> dict[str, Any]:
        route_version = None
        registry_version = None
        try:
            from projects.active.terpene_screening.core.registry_snapshots import current_snapshot_name
            from projects.active.terpene_screening.core.routing import DEFAULT_ROUTE_MANIFEST, load_route_manifest
            from projects.active.terpene_screening.manage_open_world_registry import DEFAULT_REGISTRY_ROOT
            route_version = load_route_manifest(str(DEFAULT_ROUTE_MANIFEST.resolve())).get("route_version")
            registry_version = current_snapshot_name(DEFAULT_REGISTRY_ROOT) or "legacy"
        except Exception:
            pass
        return {
            "status": "ready",
            "model_api": "production_retrieval_engine",
            "database_mode": self.database.mode,
            "database_commit": DATABASE_COMMIT,
            "database_origin": DATABASE_ORIGIN,
            "route_version": route_version,
            "registry_version": registry_version,
        }


class Handler(BaseHTTPRequestHandler):
    runtime = PortalRuntime()
    max_body_bytes = 12 * 1024 * 1024

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        try:
            if path in {"/", "/portal"}:
                self._head_redirect("/portal/")
                return
            if path == "/database":
                self._head_redirect("/database/")
                return
            if path in {"/health", "/api/portal/status"}:
                body = json.dumps(self.runtime.status(), ensure_ascii=False, default=str).encode("utf-8")
                self._head_response(HTTPStatus.OK, "application/json; charset=utf-8", len(body), cache=False)
                return
            if path.startswith("/portal/"):
                relative = path.removeprefix("/portal/")
                candidate = _safe_path(PORTAL_DIST, unquote(relative)) if relative and not relative.endswith("/") else PORTAL_DIST / "index.html"
                if not candidate.is_file():
                    candidate = PORTAL_DIST / "index.html"
                self._head_file(candidate, cache=relative.startswith("assets/"))
                return
            if path == "/database/" or path.startswith("/database/"):
                self._head_file(DATABASE_DIST / "index.html", cache=False)
                return
            if path.startswith("/assets/"):
                self._head_file(_safe_path(DATABASE_DIST, path.removeprefix("/")), cache=True)
                return
            self._head_response(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", 0, cache=False)
        except Exception:
            self._head_response(HTTPStatus.INTERNAL_SERVER_ERROR, "application/json; charset=utf-8", 0, cache=False)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path in {"/", "/portal"}:
                self._redirect("/portal/")
                return
            if path == "/database":
                self._redirect("/database/")
                return
            if path in {"/health", "/api/portal/status"}:
                self._json(HTTPStatus.OK, self.runtime.status())
                return
            if path.startswith("/api/model-data/"):
                self._model_data(parsed)
                return
            if path.startswith("/api/v1/"):
                self._bridge("GET", self.path)
                return
            if path.startswith("/portal/"):
                relative = path.removeprefix("/portal/")
                self._serve_portal(relative)
                return
            if path == "/database/" or path.startswith("/database/"):
                self._serve_file(DATABASE_DIST / "index.html", cache=False)
                return
            if path.startswith("/assets/"):
                self._serve_safe(DATABASE_DIST, path.removeprefix("/"), cache=True)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": path})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "message": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/model/rank/enzymes":
                payload = json.loads(body.decode("utf-8"))
                self._json(HTTPStatus.OK, self.runtime.engine().rank("rank-enzymes", payload))
                return
            if parsed.path == "/api/model/rank/reactions":
                payload = json.loads(body.decode("utf-8"))
                self._json(HTTPStatus.OK, self.runtime.engine().rank("rank-reactions", payload))
                return
            if parsed.path.startswith("/api/v1/"):
                self._bridge("POST", self.path, body)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found", "path": parsed.path})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "message": str(exc)})

    def _model_data(self, parsed: Any) -> None:
        path = parsed.path
        query = parse_qs(parsed.query)
        catalog = self.runtime.catalog()
        if path == "/api/model-data/summary":
            self._json(HTTPStatus.OK, catalog.summary())
            return
        if path == "/api/model-data/search":
            self._json(HTTPStatus.OK, catalog.search(
                query.get("q", [""])[0],
                query.get("kind", ["all"])[0],
                int(query.get("limit", ["40"])[0]),
            ))
            return
        if path == "/api/model-data/graph":
            self._json(HTTPStatus.OK, catalog.graph(
                query.get("q", [""])[0],
                int(query.get("limit", ["36"])[0]),
                query.get("focus_id", [None])[0],
            ))
            return
        prefix = "/api/model-data/entities/"
        if path.startswith(prefix):
            parts = path.removeprefix(prefix).split("/", 1)
            if len(parts) == 2:
                entity = catalog.entity(parts[0], unquote(parts[1]))
                if entity is not None:
                    self._json(HTTPStatus.OK, entity)
                    return
            self._json(HTTPStatus.NOT_FOUND, {"error": "entity_not_found", "path": path})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "model_data_endpoint_not_found", "path": path})

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > self.max_body_bytes:
            raise ValueError("Invalid request body size")
        return self.rfile.read(length)

    def _bridge(self, method: str, path: str, body: bytes = b"") -> None:
        response = self.runtime.database.handle(method, path, body)
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (response.headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response.body)

    def _serve_portal(self, relative: str) -> None:
        if not PORTAL_DIST.exists():
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "error": "portal_not_built",
                "message": "Build frontend/terpene_portal before starting the portal server.",
            })
            return
        if not relative or relative.endswith("/"):
            self._serve_file(PORTAL_DIST / "index.html", cache=False)
            return
        candidate = _safe_path(PORTAL_DIST, unquote(relative))
        if candidate.is_file():
            self._serve_file(candidate, cache=relative.startswith("assets/"))
        else:
            self._serve_file(PORTAL_DIST / "index.html", cache=False)

    def _serve_safe(self, root: Path, relative: str, *, cache: bool) -> None:
        candidate = _safe_path(root, unquote(relative))
        if not candidate.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "asset_not_found", "path": relative})
            return
        self._serve_file(candidate, cache=cache)

    def _serve_file(self, path: Path, *, cache: bool) -> None:
        if not path.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "file_not_found", "path": str(path.relative_to(ROOT))})
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json", "image/svg+xml"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _head_file(self, path: Path, *, cache: bool) -> None:
        if not path.is_file():
            self._head_response(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", 0, cache=False)
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json", "image/svg+xml"}:
            content_type += "; charset=utf-8"
        self._head_response(HTTPStatus.OK, content_type, path.stat().st_size, cache=cache)

    def _head_response(self, status: int, content_type: str, length: int, *, cache: bool) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

    def _head_redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.end_headers()

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"terpene-portal {self.address_string()} {format % args}", file=sys.stderr)


def _safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError("Path traversal rejected")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the model-first TerpeneNavigator portal and read-only database view.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    if not DATABASE_DIST.is_dir():
        raise SystemExit(f"Pinned database frontend build not found: {DATABASE_DIST}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"url": f"http://{args.host}:{args.port}/portal/", **Handler.runtime.status()}, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
