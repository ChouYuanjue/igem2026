from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility/terpene_runtime_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"Missing manifest: {MANIFEST}", file=sys.stderr)
        return 1
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    checked = 0
    for relative, expected in payload["files"].items():
        path = ROOT / relative
        if not path.exists():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        checked += 1
        if actual != expected:
            failures.append(f"sha256 mismatch: {relative}: {actual} != {expected}")

    for asset in payload.get("external_assets", []):
        path = ROOT / asset["target"]
        if not path.exists():
            failures.append(
                f"external asset missing: {asset['target']} (restore with scripts/bootstrap_terpene_runtime.sh)"
            )
            continue
        actual = sha256(path)
        checked += 1
        if actual != asset["sha256"]:
            failures.append(
                f"external asset sha256 mismatch: {asset['target']}: {actual} != {asset['sha256']}"
            )

    required_shapes = payload.get("array_shapes", {})
    if required_shapes:
        try:
            import numpy as np
        except ImportError:
            failures.append("numpy is unavailable; array shape checks skipped")
        else:
            for relative, expected_shape in required_shapes.items():
                path = ROOT / relative
                if not path.exists():
                    continue
                if path.suffix == ".npy":
                    shape = list(np.load(path, mmap_mode="r").shape)
                elif path.suffix == ".npz":
                    from scipy import sparse

                    shape = list(sparse.load_npz(path).shape)
                else:
                    continue
                if shape != expected_shape:
                    failures.append(f"shape mismatch: {relative}: {shape} != {expected_shape}")

    if failures:
        print("Terpene runtime verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "valid",
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "checked_files": checked,
                "manifest_version": payload.get("manifest_version"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
