from __future__ import annotations

from datetime import datetime, timezone

from ..http import get_json

ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_entry(pdb_id: str) -> dict:
    obj, _ = get_json(ENTRY.format(pdb_id=pdb_id.lower()))
    info = obj.get("rcsb_entry_info") or {}
    exptl = obj.get("exptl") or []
    methods = sorted(set(str(x.get("method", "")) for x in exptl if x.get("method")))
    resolution = info.get("resolution_combined") or []
    return {
        "pdb_id": pdb_id.upper(),
        "experimental": 1 if methods else 0,
        "method": ";".join(methods),
        "resolution": min(resolution) if resolution else None,
        "fetched_utc": utcnow(),
    }
