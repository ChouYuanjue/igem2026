from __future__ import annotations

import argparse
import json
import subprocess
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.engine import RetrievalEngine  # noqa: E402
from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    current_snapshot_name,
)
from projects.active.terpene_screening.core.routing import (  # noqa: E402
    DEFAULT_ROUTE_MANIFEST,
    load_route_manifest,
)
from projects.active.terpene_screening.manage_open_world_registry import (  # noqa: E402
    DEFAULT_REGISTRY_ROOT,
    DEFAULT_REACTION_REGISTRY,
    DEFAULT_PROTEIN_REGISTRY,
    DEFAULT_CURRENT_PROTEINS,
    DEFAULT_DEPLOYMENT,
    registry_status,
)


def registry_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        registry_root=DEFAULT_REGISTRY_ROOT,
        protein_registry=DEFAULT_PROTEIN_REGISTRY,
        reaction_registry=DEFAULT_REACTION_REGISTRY,
        current_protein_dir=DEFAULT_CURRENT_PROTEINS,
        deployment_dir=DEFAULT_DEPLOYMENT,
    )


class Handler(BaseHTTPRequestHandler):
    engine = RetrievalEngine()
    allow_registry_writes = False
    max_body_bytes = 10 * 1024 * 1024

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > self.max_body_bytes:
            raise ValueError("Invalid request body size")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        try:
            if self.path == "/health":
                routes = load_route_manifest(str(DEFAULT_ROUTE_MANIFEST.resolve()))
                self._json(
                    HTTPStatus.OK,
                    {
                        "status": "ready",
                        "route_version": routes["route_version"],
                        "registry_version": current_snapshot_name(DEFAULT_REGISTRY_ROOT) or "legacy",
                    },
                )
                return
            if self.path == "/registry/status":
                self._json(HTTPStatus.OK, registry_status(registry_namespace()))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "message": str(exc)})

    def _registry_write(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.allow_registry_writes:
            raise PermissionError("Registry writes are disabled; start with --allow-registry-writes")
        allowed = {
            "add-enzymes": {"enzyme_id", "sequence", "source_label", "replace"},
            "add-reactions": {"reaction_id", "reaction_smiles", "source_label", "replace"},
        }[command]
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unsupported registry fields: {sorted(unknown)}")
        argv = [
            sys.executable,
            str(ROOT / "projects/active/terpene_screening/manage_open_world_registry.py"),
            command,
        ]
        for key, value in payload.items():
            if value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    argv.append(flag)
            else:
                argv.extend([flag, str(value)])
        completed = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr[-4000:] or completed.stdout[-4000:])
        return json.loads(completed.stdout)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._body()
            if self.path == "/rank/enzymes":
                self._json(HTTPStatus.OK, self.engine.rank("rank-enzymes", payload))
                return
            if self.path == "/rank/reactions":
                self._json(HTTPStatus.OK, self.engine.rank("rank-reactions", payload))
                return
            if self.path == "/registry/enzymes":
                self._json(HTTPStatus.OK, self._registry_write("add-enzymes", payload))
                return
            if self.path == "/registry/reactions":
                self._json(HTTPStatus.OK, self._registry_write("add-reactions", payload))
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        except PermissionError as exc:
            self._json(HTTPStatus.FORBIDDEN, {"error": type(exc).__name__, "message": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": type(exc).__name__, "message": str(exc)})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        print(f"terpene-api {self.address_string()} {format % args}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dependency-free HTTP wrapper around the TPS RetrievalEngine.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-registry-writes", action="store_true")
    parser.add_argument("--allow-model-overrides", action="store_true")
    args = parser.parse_args()
    Handler.engine = RetrievalEngine(allow_overrides=args.allow_model_overrides)
    Handler.allow_registry_writes = args.allow_registry_writes
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"host": args.host, "port": args.port, "registry_writes": args.allow_registry_writes}, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
