from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "reproducibility/research_release_manifest.json"
GITHUB_BLOB_LIMIT = 100_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return {item.decode("utf-8") for item in completed.stdout.split(b"\0") if item}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Git-scoped research release.")
    parser.add_argument(
        "--portable-only",
        action="store_true",
        help="Require only directly tracked assets and rebuild contracts; do not require large local/external payloads.",
    )
    args = parser.parse_args()

    if not MANIFEST.is_file():
        print(f"Missing release manifest: {MANIFEST}", file=sys.stderr)
        return 1
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tracked = tracked_paths()
    failures: list[str] = []

    if payload.get("release_branch") != "master":
        failures.append(
            f"research release branch must be master, got {payload.get('release_branch')!r}"
        )

    direct = payload.get("direct_git_assets", [])
    for record in direct:
        relative = str(record["path"])
        path = ROOT / relative
        if relative not in tracked:
            failures.append(f"release asset is not Git-tracked: {relative}")
            continue
        if not path.is_file():
            failures.append(f"tracked release asset missing from checkout: {relative}")
            continue
        size = path.stat().st_size
        if size != int(record["bytes"]):
            failures.append(f"size mismatch: {relative}: {size} != {record['bytes']}")
        if size >= GITHUB_BLOB_LIMIT:
            failures.append(f"release asset exceeds GitHub blob limit: {relative}: {size}")
        digest = sha256(path)
        if digest != str(record["sha256"]):
            failures.append(f"sha256 mismatch: {relative}: {digest} != {record['sha256']}")

    canonical = json.loads((ROOT / "reproducibility/bime_rank/canonical.json").read_text())
    declared = payload.get("canonical_claim_primaries", {})
    for claim_id, claim in canonical["claims"].items():
        relative = str(claim["primary"])
        if declared.get(claim_id) != relative:
            failures.append(f"canonical claim manifest drift: {claim_id}")
        if relative not in tracked:
            failures.append(f"canonical primary is not Git-tracked: {claim_id}: {relative}")
        if not (ROOT / relative).is_file():
            failures.append(f"canonical primary missing: {claim_id}: {relative}")

    for item in payload.get("rebuildable_assets", []):
        for key in ("builder", "merge_builder", "metadata_manifest"):
            relative = item.get(key)
            if not relative:
                continue
            if relative not in tracked:
                failures.append(f"rebuild contract {key} is not Git-tracked: {relative}")
            if not (ROOT / relative).is_file():
                failures.append(f"rebuild contract {key} missing: {relative}")
        path = ROOT / str(item["path"])
        if args.portable_only or not path.exists():
            continue
        expected_shape = item.get("expected_shape")
        if expected_shape and path.suffix in {".npy", ".npz"}:
            try:
                import numpy as np
                if path.suffix == ".npy":
                    shape = list(np.load(path, mmap_mode="r").shape)
                else:
                    from scipy import sparse
                    shape = list(sparse.load_npz(path).shape)
                if shape != list(expected_shape):
                    failures.append(f"rebuildable shape mismatch: {item['path']}: {shape} != {expected_shape}")
            except Exception as exc:
                failures.append(f"cannot inspect rebuildable asset {item['path']}: {exc!r}")

    for asset in payload.get("external_assets", []):
        relative = str(asset["target"])
        if relative in tracked:
            failures.append(f"large third-party asset must not be vendored in normal Git: {relative}")
        if args.portable_only:
            continue
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"external asset missing: {relative}")
            continue
        expected_size = asset.get("bytes")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            failures.append(f"external asset size mismatch: {relative}")
        expected_sha = asset.get("sha256")
        if expected_sha and sha256(path) != str(expected_sha):
            failures.append(f"external asset sha256 mismatch: {relative}")

    private_roots = [str(value) for value in payload.get("private_roots", [])]
    for relative in tracked:
        if any(relative.startswith(root) for root in private_roots):
            failures.append(f"private/local path is Git-tracked: {relative}")

    # Unrelated local development branches may exist on the server, but must not be in a research checkout.
    for forbidden in ("src/experimental_touch_cascade/", "configs/experimental_touch/", "projects/planned/"):
        offenders = sorted(path for path in tracked if path.startswith(forbidden))
        if offenders:
            failures.append(f"non-release Git surface remains under {forbidden}: {len(offenders)} files")

    result = {
        "status": "valid" if not failures else "invalid",
        "portable_only": args.portable_only,
        "direct_git_assets": len(direct),
        "direct_git_bytes": sum(int(x["bytes"]) for x in direct),
        "canonical_claims": len(canonical["claims"]),
        "rebuildable_assets": len(payload.get("rebuildable_assets", [])),
        "external_assets": len(payload.get("external_assets", [])),
        "failures": len(failures),
    }
    print(json.dumps(result, indent=2))
    if failures:
        for failure in failures[:200]:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
