from __future__ import annotations

from pathlib import Path


class SeparationError(RuntimeError):
    pass


def _within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_runtime_separation(profile) -> None:
    r = profile.runtime
    if not _within(r.evidence_db, r.allowed_evidence_root):
        raise SeparationError(
            f"evidence_db must stay under allowed_evidence_root: {r.evidence_db} !< {r.allowed_evidence_root}"
        )
    if not _within(r.run_root, r.allowed_run_root):
        raise SeparationError(
            f"run_root must stay under allowed_run_root: {r.run_root} !< {r.allowed_run_root}"
        )
    # Source candidate files must never be reused as runtime databases.
    source_paths = {profile.source.candidates_path.resolve()}
    if profile.source.metadata_path:
        source_paths.add(profile.source.metadata_path.resolve())
    if r.evidence_db.resolve() in source_paths:
        raise SeparationError("evidence_db collides with candidate source")
    if _within(r.run_root, profile.repo_root / "external_repos" / "igem_database"):
        raise SeparationError("run_root may not be inside the production/upstream database repository")
    if _within(r.evidence_db, profile.repo_root / "external_repos" / "igem_database"):
        raise SeparationError("evidence_db may not be inside the production/upstream database repository")


def ensure_runtime_dirs(profile) -> None:
    assert_runtime_separation(profile)
    profile.runtime.evidence_db.parent.mkdir(parents=True, exist_ok=True)
    profile.runtime.run_root.mkdir(parents=True, exist_ok=True)
    profile.runtime.cache_root.mkdir(parents=True, exist_ok=True)
