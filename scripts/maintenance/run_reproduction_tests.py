from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = ROOT / "reproducibility/bime_rank/source_roles.json"
MOVE_MAP = ROOT / "scripts/maintenance/repository_move_map.json"


def _resolve_test_path(relative: str, move_map: dict[str, str]) -> str:
    """Resolve a historical logical test path to its current repository location."""
    current = str(relative)
    seen: set[str] = set()
    while not (ROOT / current).is_file() and current in move_map:
        if current in seen:
            raise SystemExit(f"repository move-map cycle while resolving test: {relative}")
        seen.add(current)
        current = str(move_map[current])
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Git-defined BiME-Rank project test surface.")
    parser.add_argument(
        "--tier",
        choices=("release", "extended"),
        default="extended",
        help="release: portable project regression only; extended: release plus current/canonical reproduction tests",
    )
    args = parser.parse_args()
    roles = json.loads(ROLES.read_text(encoding="utf-8"))
    tests = list(roles.get("release_regression", []))
    if args.tier == "extended":
        tests.extend(roles.get("extended_reproduction_tests", []))
    tests = sorted(set(tests))
    if not tests:
        raise SystemExit("source_roles.json selected no project tests")
    move_map = json.loads(MOVE_MAP.read_text(encoding="utf-8")) if MOVE_MAP.is_file() else {}
    resolved = [_resolve_test_path(rel, move_map) for rel in tests]
    missing = [
        f"{logical} -> {physical}" if logical != physical else logical
        for logical, physical in zip(tests, resolved)
        if not (ROOT / physical).is_file()
    ]
    if missing:
        raise SystemExit("missing selected project test(s): " + ", ".join(missing))
    resolved = sorted(set(resolved))
    print(
        json.dumps(
            {
                "tier": args.tier,
                "test_files": len(resolved),
                "relocated_test_files": sum(a != b for a, b in zip(tests, [_resolve_test_path(x, move_map) for x in tests])),
            },
            indent=2,
        )
    )
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *resolved], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
