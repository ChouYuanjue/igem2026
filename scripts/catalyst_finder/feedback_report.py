from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "results/catalyst_finder_runtime/feedback.jsonl"


def load_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    if not path.exists():
        return rows, invalid
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            invalid += 1
    return rows, invalid


def iso_time(value: Any) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        return ""


def summarize(records: list[dict[str, Any]], invalid: int = 0) -> dict[str, Any]:
    ratings = Counter(str(row.get("rating") or "unspecified") for row in records)
    categories = Counter(str(row.get("category") or "other") for row in records)
    directions = Counter(str((row.get("context") or {}).get("direction") or "unknown") for row in records)
    result_modes = Counter(str((row.get("context") or {}).get("result_mode") or "unknown") for row in records)
    routes = Counter(str((row.get("context") or {}).get("route_id") or "unknown") for row in records)
    contacts = sum(1 for row in records if str(row.get("contact") or "").strip())
    return {
        "total": len(records),
        "invalid_lines": int(invalid),
        "with_contact": contacts,
        "rating": dict(ratings.most_common()),
        "category": dict(categories.most_common()),
        "direction": dict(directions.most_common()),
        "result_mode": dict(result_modes.most_common()),
        "route": dict(routes.most_common(20)),
    }


def public_row(row: dict[str, Any], *, include_contact: bool = False) -> dict[str, Any]:
    context = row.get("context") if isinstance(row.get("context"), dict) else {}
    result = {
        "feedback_id": row.get("feedback_id"),
        "submitted_at": iso_time(row.get("submitted_at_unix")),
        "rating": row.get("rating"),
        "category": row.get("category"),
        "message": row.get("message"),
        "context": context,
    }
    if include_contact:
        result["contact"] = row.get("contact")
    elif row.get("contact"):
        result["contact"] = "[redacted]"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Catalyst Finder feedback report (server-side only)")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--days", type=float, default=None, help="Only include records from the most recent N days")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent records to display")
    parser.add_argument("--include-contact", action="store_true", help="Include contact fields in output")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--raw", action="store_true", help="Emit filtered JSONL records instead of a summary")
    args = parser.parse_args()

    records, invalid = load_records(args.path)
    if args.days is not None:
        cutoff = time.time() - max(0.0, float(args.days)) * 86400.0
        records = [row for row in records if float(row.get("submitted_at_unix") or 0.0) >= cutoff]
    records.sort(key=lambda row: float(row.get("submitted_at_unix") or 0.0))

    if args.raw:
        for row in records[-max(0, args.limit):] if args.limit else records:
            print(json.dumps(public_row(row, include_contact=args.include_contact), ensure_ascii=False))
        return 0

    payload = {
        "path": str(args.path),
        "summary": summarize(records, invalid),
        "recent": [public_row(row, include_contact=args.include_contact) for row in records[-max(0, args.limit):]],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    s = payload["summary"]
    print(f"Feedback: {s['total']} records · {s['with_contact']} with contact · {s['invalid_lines']} invalid lines")
    for label, key in [("Rating", "rating"), ("Category", "category"), ("Direction", "direction"), ("Result mode", "result_mode")]:
        values = s[key]
        rendered = ", ".join(f"{k}={v}" for k, v in values.items()) if values else "—"
        print(f"{label}: {rendered}")
    if payload["recent"]:
        print("Recent:")
        for row in payload["recent"]:
            ctx = row.get("context") or {}
            print(f"- {row.get('submitted_at') or 'unknown time'} · {row.get('rating') or '—'} · {row.get('category') or '—'} · {ctx.get('direction') or '—'}")
            print(f"  {str(row.get('message') or '').replace(chr(10), ' ')[:500]}")
            if row.get("contact"):
                print(f"  contact: {row['contact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
