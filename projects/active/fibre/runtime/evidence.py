from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from projects.active.fibre.evidence.pair_scores import (
    ClipzymePairEvidence,
    enzgfm_pair_evidence,
)
from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.model.index import FibreCandidateIndex
from projects.active.fibre.runtime.scientific_evidence import (
    Direction,
    ScientificEvidenceModule,
    apply_live_scientific_evidence_bundle,
    load_admitted_evidence_bundle,
)


DEFAULT_EVIDENCE_RELEASE = ROOT / "projects/active/fibre/release/manifests/score_evidence_v1"


def _default_release_paths(direction: Direction) -> tuple[Path, Path]:
    return (
        DEFAULT_EVIDENCE_RELEASE / f"{direction}_bundle.json",
        DEFAULT_EVIDENCE_RELEASE / f"{direction}_core_calibration.json",
    )


@dataclass(frozen=True)
class RankedEvidenceCandidate:
    identifier: str
    score: float
    core_score: float
    contributions: dict[str, float]


class FibreEvidenceRuntime:
    """Frozen broad retrieval updated only by admitted pair-level evidence."""

    def __init__(
        self,
        *,
        direction: Direction,
        bundle_path: str | Path | None = None,
        core_calibration_path: str | Path | None = None,
        device: str | torch.device = "cuda",
        modules: list[ScientificEvidenceModule] | None = None,
    ) -> None:
        self.direction = direction
        self.device = torch.device(device)
        default_bundle, default_calibration = _default_release_paths(direction)
        bundle_path = default_bundle if bundle_path is None else Path(bundle_path)
        core_calibration_path = (
            default_calibration if core_calibration_path is None else Path(core_calibration_path)
        )
        self.index = FibreCandidateIndex(device=self.device)
        self.bundle = load_admitted_evidence_bundle(Path(bundle_path))
        calibration = json.loads(Path(core_calibration_path).read_text())
        if str(calibration.get("baseline_id") or "") != self.bundle.baseline_id:
            raise ValueError("evidence bundle and broad-core calibration use different baselines")
        if calibration.get("method") != "fixed_global_affine_v1":
            raise ValueError("unsupported broad-core calibration")
        if not bool(calibration.get("ranking_invariant", False)):
            raise ValueError("broad-core calibration must preserve ranking")
        self.core_center = float(calibration["center"])
        self.core_scale = float(calibration["scale"])
        if self.core_scale <= 0:
            raise ValueError("broad-core calibration scale must be positive")
        self.modules = modules or self._default_modules(direction)
        member_names = [member.descriptor.name for member in self.bundle.members]
        module_names = [str(module.name) for module in self.modules]
        if module_names != member_names:
            raise ValueError(f"bundle/modules mismatch: {member_names} != {module_names}")

    def _default_modules(self, direction: Direction) -> list[ScientificEvidenceModule]:
        modules: dict[str, ScientificEvidenceModule] = {
            "clipzyme_structure": ClipzymePairEvidence(device=self.device),
            f"enzgfm_{direction}": enzgfm_pair_evidence(direction, device=self.device),
        }
        return [modules[member.descriptor.name] for member in self.bundle.members]

    def _core_scores(
        self,
        query_id: str,
        candidate_ids: list[str],
    ) -> np.ndarray:
        if self.direction == "r2e":
            q = self.index.reaction_embeddings[self.index.reaction_index[str(query_id)]]
            rows = torch.as_tensor(
                [self.index.protein_index[str(x)] for x in candidate_ids],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                raw = self.index.protein_embeddings.index_select(0, rows) @ q
        else:
            q = self.index.protein_embeddings[self.index.protein_index[str(query_id)]]
            rows = torch.as_tensor(
                [self.index.reaction_index[str(x)] for x in candidate_ids],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                raw = self.index.reaction_embeddings.index_select(0, rows) @ q
        values = raw.float().cpu().numpy().astype(np.float64, copy=False)
        return (values - self.core_center) / self.core_scale

    def score_candidates(
        self,
        query_id: str,
        *,
        candidate_ids: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if candidate_ids is None:
            candidate_ids = (
                list(self.index.protein_ids)
                if self.direction == "r2e"
                else list(self.index.reaction_ids)
            )
        core = self._core_scores(str(query_id), candidate_ids)
        fused, contributions = apply_live_scientific_evidence_bundle(
            core,
            candidate_ids,
            direction=self.direction,
            query_id=str(query_id),
            modules=self.modules,
            bundle=self.bundle,
        )
        return candidate_ids, fused, core, contributions

    def rank(
        self,
        query_id: str,
        *,
        top_k: int = 20,
        candidate_ids: list[str] | None = None,
    ) -> list[RankedEvidenceCandidate]:
        candidate_ids, fused, core, contributions = self.score_candidates(
            query_id, candidate_ids=candidate_ids
        )
        order = np.argsort(-fused, kind="stable")[: int(top_k)]
        names = list(contributions)
        return [
            RankedEvidenceCandidate(
                identifier=candidate_ids[int(i)],
                score=float(fused[int(i)]),
                core_score=float(core[int(i)]),
                contributions={name: float(contributions[name][int(i)]) for name in names},
            )
            for i in order
        ]

    def reaction_to_enzyme(self, reaction_id: str, *, top_k: int = 20) -> list[RankedEvidenceCandidate]:
        if self.direction != "r2e":
            raise ValueError("runtime is not configured for R2E")
        return self.rank(str(reaction_id), top_k=top_k)

    def enzyme_to_reaction(self, protein_id: str, *, top_k: int = 20) -> list[RankedEvidenceCandidate]:
        if self.direction != "e2r":
            raise ValueError("runtime is not configured for E2R")
        return self.rank(str(protein_id), top_k=top_k)
