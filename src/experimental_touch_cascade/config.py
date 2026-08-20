from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceConfig:
    kind: str
    candidates_path: Path
    metadata_path: Path | None
    id_column: str = "enzyme_id"
    sequence_column: str = "sequence"
    sha256_column: str = "sequence_sha256"


@dataclass(frozen=True)
class RuntimeConfig:
    evidence_db: Path
    run_root: Path
    cache_root: Path
    allowed_evidence_root: Path
    allowed_run_root: Path


@dataclass(frozen=True)
class FocusConfig:
    ranking_path: Path | None
    candidate_column: str = "candidate_id"
    group_column: str = "requested_group"
    rank_column: str = "rank"
    stage2_top_k_per_group: int = 200
    stage3_top_k_per_group: int = 50


@dataclass(frozen=True)
class PolicyConfig:
    stage1_uniparc_batch_size: int = 100
    stage1_uniprot_batch_size: int = 80
    stage1_workers: int = 6
    stage2_workers: int = 4
    stage3_workers: int = 3
    stage2_max_candidates: int = 10000
    stage3_max_candidates: int = 2000
    stage3_max_papers_per_candidate: int = 20
    promote_pe_at_most: int = 2
    promote_reviewed: bool = True
    promote_pdb: bool = True
    promote_structured_experiment: bool = True


@dataclass(frozen=True)
class Profile:
    profile_id: str
    repo_root: Path
    source: SourceConfig
    runtime: RuntimeConfig
    focus: FocusConfig
    policy: PolicyConfig
    raw: dict[str, Any]


def _resolve(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p).resolve()


def load_profile(path: str | Path, repo_root: str | Path | None = None) -> Profile:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = Path(repo_root).resolve() if repo_root else path.parents[3].resolve()
    s = raw["source"]
    r = raw["runtime"]
    f = raw.get("focus", {})
    p = raw.get("policy", {})
    return Profile(
        profile_id=str(raw["profile_id"]),
        repo_root=root,
        source=SourceConfig(
            kind=str(s.get("kind", "csv")),
            candidates_path=_resolve(root, s["candidates_path"]),
            metadata_path=_resolve(root, s.get("metadata_path")),
            id_column=str(s.get("id_column", "enzyme_id")),
            sequence_column=str(s.get("sequence_column", "sequence")),
            sha256_column=str(s.get("sha256_column", "sequence_sha256")),
        ),
        runtime=RuntimeConfig(
            evidence_db=_resolve(root, r["evidence_db"]),
            run_root=_resolve(root, r["run_root"]),
            cache_root=_resolve(root, r["cache_root"]),
            allowed_evidence_root=_resolve(root, r["allowed_evidence_root"]),
            allowed_run_root=_resolve(root, r["allowed_run_root"]),
        ),
        focus=FocusConfig(
            ranking_path=_resolve(root, f.get("ranking_path")),
            candidate_column=str(f.get("candidate_column", "candidate_id")),
            group_column=str(f.get("group_column", "requested_group")),
            rank_column=str(f.get("rank_column", "rank")),
            stage2_top_k_per_group=int(f.get("stage2_top_k_per_group", 200)),
            stage3_top_k_per_group=int(f.get("stage3_top_k_per_group", 50)),
        ),
        policy=PolicyConfig(**{k: v for k, v in p.items() if k in PolicyConfig.__annotations__}),
        raw=raw,
    )
