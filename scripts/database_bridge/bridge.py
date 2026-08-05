"""Read-only adapter for the pinned igem_database frontend and API.

No file under external_repos/igem_database is modified. The bridge either proxies
an explicitly configured database API or serves a compatibility snapshot for the
upstream UI features that are already implemented.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from scripts.database_bridge import compat_snapshot

SAFE_GET_PREFIXES = (
    "/api/v1/metadata/",
    "/api/v1/graph",
    "/api/v1/search/entries",
    "/api/v1/enzymes/",
    "/api/v1/compounds/",
    "/api/v1/reactions/",
    "/api/v1/assets/",
)
SAFE_POST_PATHS = {"/api/v1/search/pathways"}


@dataclass
class BridgeResponse:
    status: int
    body: bytes
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] | None = None


class DatabaseBridge:
    def __init__(self, upstream_api: str | None = None) -> None:
        self.upstream_api = (upstream_api or os.environ.get("TERPENE_DATABASE_API_URL", "")).rstrip("/")

    @property
    def mode(self) -> str:
        return "proxy" if self.upstream_api else "compatibility_snapshot"

    def handle(self, method: str, raw_path: str, body: bytes = b"") -> BridgeResponse:
        parsed = urllib.parse.urlsplit(raw_path)
        path = parsed.path
        if not self._allowed(method, path):
            return self._json(403, {
                "success": False,
                "error": {
                    "code": "READ_ONLY_BRIDGE",
                    "message": "This bridge only exposes read-only database views already implemented upstream.",
                },
            })
        if self.upstream_api:
            return self._proxy(method, parsed, body)
        return self._snapshot(method, parsed, body)

    def _allowed(self, method: str, path: str) -> bool:
        if method == "GET":
            return any(path.startswith(prefix) for prefix in SAFE_GET_PREFIXES)
        return method == "POST" and path in SAFE_POST_PATHS

    def _proxy(self, method: str, parsed: urllib.parse.SplitResult, body: bytes) -> BridgeResponse:
        url = f"{self.upstream_api}{parsed.path}"
        if parsed.query:
            url += f"?{parsed.query}"
        request = urllib.request.Request(
            url,
            data=body if method == "POST" else None,
            method=method,
            headers={"Content-Type": "application/json", "Accept": "application/json, image/svg+xml"},
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
                return BridgeResponse(
                    status=response.status,
                    body=payload,
                    content_type=response.headers.get_content_type() + "; charset=utf-8",
                    headers={"X-Terpene-Database-Bridge": "proxy"},
                )
        except urllib.error.HTTPError as exc:
            return BridgeResponse(
                status=exc.code,
                body=exc.read(),
                content_type=exc.headers.get_content_type() if exc.headers else "application/json; charset=utf-8",
            )
        except Exception as exc:
            return self._json(502, {
                "success": False,
                "error": {"code": type(exc).__name__, "message": str(exc)},
            })

    def _snapshot(self, method: str, parsed: urllib.parse.SplitResult, body: bytes) -> BridgeResponse:
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/api/v1/metadata/filter-options":
            return self._data({
                "organisms": sorted({edge["card"]["organismName"] for edge in compat_snapshot.EDGES}),
                "sourceTypes": ["swiss_prot", "manual_literature", "ai_literature"],
                "reviewStatuses": ["official", "reviewed", "pending"],
            })
        if path == "/api/v1/graph":
            return self._data(compat_snapshot.graph_payload())
        if path.startswith("/api/v1/graph/edge-groups/") and path.endswith("/edges"):
            group_id = urllib.parse.unquote(path.removeprefix("/api/v1/graph/edge-groups/").removesuffix("/edges"))
            return self._data({"edgeGroupId": group_id, "edges": compat_snapshot.edge_group(group_id)})
        if path == "/api/v1/search/entries":
            return self._data({"items": compat_snapshot.search_entries(query.get("q", [""])[0]), "page": 1, "pageSize": 80, "total": len(compat_snapshot.EDGES), "totalPages": 1})
        if path == "/api/v1/search/pathways" and method == "POST":
            payload = json.loads(body.decode("utf-8") or "{}")
            items = compat_snapshot.pathway(str(payload.get("startCompoundId", "")), str(payload.get("endCompoundId", "")))
            return self._data({"items": items})
        if path.startswith("/api/v1/enzymes/"):
            enzyme_id = urllib.parse.unquote(path.removeprefix("/api/v1/enzymes/"))
            detail = compat_snapshot.enzyme_detail(enzyme_id)
            return self._data(detail) if detail else self._not_found(enzyme_id)
        if path.startswith("/api/v1/compounds/") and path.endswith("/card"):
            compound_id = urllib.parse.unquote(path.removeprefix("/api/v1/compounds/").removesuffix("/card"))
            record = compat_snapshot.compound(compound_id)
            return self._data(record) if record else self._not_found(compound_id)
        if path.startswith("/api/v1/reactions/"):
            reaction_id = urllib.parse.unquote(path.removeprefix("/api/v1/reactions/"))
            edge = next((item for item in compat_snapshot.EDGES if item["reactionId"] == reaction_id), None)
            if not edge:
                return self._not_found(reaction_id)
            detail = compat_snapshot.enzyme_detail(edge["enzymeId"])
            reaction = next(item for item in detail["reactions"] if item["reactionId"] == reaction_id) if detail else None
            return self._data(reaction)
        if path.startswith("/api/v1/assets/compounds/") and path.endswith("/structure.svg"):
            compound_id = urllib.parse.unquote(path.removeprefix("/api/v1/assets/compounds/").removesuffix("/structure.svg"))
            return BridgeResponse(200, _structure_svg(compound_id), "image/svg+xml; charset=utf-8")
        if path.startswith("/api/v1/assets/reactions/") and path.endswith("/atom-map.svg"):
            reaction_id = urllib.parse.unquote(path.removeprefix("/api/v1/assets/reactions/").removesuffix("/atom-map.svg"))
            return BridgeResponse(200, _reaction_svg(reaction_id), "image/svg+xml; charset=utf-8")
        return self._not_found(path)

    def _data(self, payload: Any) -> BridgeResponse:
        return self._json(200, {"success": True, "data": payload, "meta": {"bridgeMode": self.mode}})

    def _not_found(self, identifier: str) -> BridgeResponse:
        return self._json(404, {"success": False, "error": {"code": "NOT_FOUND", "message": f"No compatibility record for {identifier}"}})

    @staticmethod
    def _json(status: int, payload: Any) -> BridgeResponse:
        return BridgeResponse(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _structure_svg(label: str) -> bytes:
    safe = label.replace("&", "&amp;").replace("<", "&lt;")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="220" viewBox="0 0 300 220">
<rect width="300" height="220" rx="18" fill="#f8f7fb"/>
<g fill="none" stroke="#2e756d" stroke-width="5" stroke-linecap="round"><path d="M52 132 L95 96 L140 132 L190 88 L242 128"/><circle cx="95" cy="96" r="7" fill="#e89143" stroke="none"/></g>
<text x="150" y="184" text-anchor="middle" fill="#61706f" font-family="sans-serif" font-size="14">{safe} · compatibility structure</text></svg>'''
    return svg.encode("utf-8")


def _reaction_svg(label: str) -> bytes:
    safe = label.replace("&", "&amp;").replace("<", "&lt;")
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="180" viewBox="0 0 520 180">
<rect width="520" height="180" rx="18" fill="#f8f7fb"/><circle cx="105" cy="90" r="34" fill="#d8f2ed" stroke="#2e756d" stroke-width="3"/><circle cx="415" cy="90" r="34" fill="#fce8cf" stroke="#d58a3f" stroke-width="3"/><path d="M170 90 H350" stroke="#6d69a4" stroke-width="4"/><path d="M350 90 l-18 -10 v20 z" fill="#6d69a4"/><text x="260" y="55" text-anchor="middle" fill="#4f4b72" font-family="sans-serif" font-size="14">{safe}</text></svg>'''
    return svg.encode("utf-8")
