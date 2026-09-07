from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = ROOT / "reproducibility/bime_rank/source_roles.json"


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
    missing = [rel for rel in tests if not (ROOT / rel).is_file()]
    if missing:
        raise SystemExit("missing selected project test(s): " + ", ".join(missing))
    print(json.dumps({"tier": args.tier, "test_files": len(tests)}, indent=2))
    return subprocess.call([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
