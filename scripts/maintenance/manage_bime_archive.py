"""Verify or explicitly restore reviewed BiME-Rank historical archive moves.

This tool never decides what is historical. That decision is frozen in
reproducibility/bime_rank/archive_moves.json. Default behavior is read-only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reproducibility/bime_rank/archive_moves.json"
CANONICAL = ROOT / "reproducibility/bime_rank/canonical.json"


def load_moves() -> list[dict]:
    payload = json.loads(MANIFEST.read_text())
    if payload.get("deleted", 0) != 0:
        raise RuntimeError("Archive manifest unexpectedly records deletions")
    return payload["moves"]


def canonical_primaries() -> set[str]:
    payload = json.loads(CANONICAL.read_text())
    return {claim["primary"] for claim in payload["claims"].values()}


def verify() -> int:
    moves = load_moves()
    primaries = canonical_primaries()
    errors: list[str] = []
    total_files = 0
    total_bytes = 0
    for move in moves:
        src = ROOT / move["from"]
        dst = ROOT / move["to"]
        total_files += int(move["files"])
        total_bytes += int(move["bytes"])
        if src.exists():
            errors.append(f"source unexpectedly exists: {move['from']}")
        if not dst.is_dir():
            errors.append(f"archive destination missing: {move['to']}")
        if any(p == move["from"] or p.startswith(move["from"] + "/") for p in primaries):
            errors.append(f"canonical primary lies under archived source: {move['from']}")
        for evidence in move.get("evidence", []):
            p = dst / evidence["file"]
            if not p.is_file():
                errors.append(f"evidence missing: {p.relative_to(ROOT)}")
                continue
            if "sha256" in evidence:
                import hashlib
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                if digest != evidence["sha256"]:
                    errors.append(f"evidence SHA mismatch: {p.relative_to(ROOT)}")
    result = {
        "moves": len(moves),
        "files_recorded": total_files,
        "bytes_recorded": total_bytes,
        "deleted": 0,
        "canonical_primaries_archived": False,
        "ok": not errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        for error in errors[:100]:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


def restore(path: str, apply: bool) -> int:
    moves = load_moves()
    match = next((m for m in moves if m["from"] == path), None)
    if match is None:
        print(f"Not a reviewed archive source path: {path}", file=sys.stderr)
        return 2
    src = ROOT / match["from"]
    dst = ROOT / match["to"]
    if src.exists():
        print(f"Refusing: original path already exists: {match['from']}", file=sys.stderr)
        return 2
    if not dst.is_dir():
        print(f"Refusing: archive destination missing: {match['to']}", file=sys.stderr)
        return 2
    print(f"restore: {match['to']} -> {match['from']}")
    if not apply:
        print("dry-run only; pass --apply to move it back")
        return 0
    src.parent.mkdir(parents=True, exist_ok=True)
    if src.parent.stat().st_dev != dst.parent.stat().st_dev:
        print("Refusing cross-filesystem restore; use an explicit audited copy instead", file=sys.stderr)
        return 2
    os.replace(dst, src)
    print("restored")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--path", required=True, help="Exact original results/... path from archive_moves.json")
    restore_parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        return verify()
    return restore(args.path, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
