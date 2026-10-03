from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from projects.active.bridge.evidence.experts import (
    CageFamilyPairEvidence,
    ReactionCenterPairEvidence,
    SeedHomologyPairEvidence,
)
from projects.active.bridge.evidence.pair_scores import (
    ClipzymePairEvidence,
    TpsPairEvidence,
    enzgfm_pair_evidence,
)
from projects.active.bridge.model.index import FibreCandidateIndex
from projects.active.bridge.runtime.scientific_evidence import (
    EvidenceOutput,
    ScientificEvidenceModule,
)
from projects.active.bridge.kernel.evidence_fusion import query_standardize


EXPERT_TYPE_IDS = (
    "functional_foundation",
    "structural_geometry",
    "mechanistic",
    "family_domain",
    "contextual_observational",
)


@dataclass(frozen=True)
class ExpertTypeOutput:
    type_id: str
    score: np.ndarray
    available: np.ndarray
    member_scores: dict[str, np.ndarray]
    member_available: dict[str, np.ndarray]

    @property
    def supported_candidates(self) -> int:
        return int(np.asarray(self.available, dtype=bool).sum())


class R2EExpertTypeRuntime:
    """Five coarse expert channels on top of the frozen Broad candidate space.

    Concrete experts remain visible for audit, but the public interface exposes
    only type-level channels.  This is deliberate: later ablation operates on
    five expert types, not on every concrete expert.
    """

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        enabled_types: Iterable[str] | None = None,
        seed_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        requested = tuple(EXPERT_TYPE_IDS if enabled_types is None else map(str, enabled_types))
        unknown = sorted(set(requested) - set(EXPERT_TYPE_IDS))
        if unknown:
            raise ValueError(f"unknown expert types: {unknown}")
        self.device = torch.device(device)
        self.index = FibreCandidateIndex(device=self.device)
        self.enabled_types = requested
        self.seed_ids = tuple(seed_ids or ())
        self._modules: dict[str, list[ScientificEvidenceModule]] = {}

    def _build(self, type_id: str) -> list[ScientificEvidenceModule]:
        if type_id in self._modules:
            return self._modules[type_id]
        if type_id == "functional_foundation":
            modules: list[ScientificEvidenceModule] = [
                enzgfm_pair_evidence("r2e", device=self.device),
            ]
        elif type_id == "structural_geometry":
            modules = [ClipzymePairEvidence(device=self.device)]
        elif type_id == "mechanistic":
            modules = [ReactionCenterPairEvidence(device=self.device)]
        elif type_id == "family_domain":
            modules = [
                CageFamilyPairEvidence("p450"),
                CageFamilyPairEvidence("phosphatase"),
                CageFamilyPairEvidence("terpene"),
                TpsPairEvidence(),
            ]
        elif type_id == "contextual_observational":
            modules = []
            if self.seed_ids:
                modules.append(
                    SeedHomologyPairEvidence(
                        list(self.seed_ids),
                        index=self.index,
                        device=self.device,
                    )
                )
        else:
            raise ValueError(type_id)
        self._modules[type_id] = modules
        return modules

    @staticmethod
    def _aggregate_type(
        type_id: str,
        outputs: list[tuple[str, EvidenceOutput]],
        candidate_count: int,
    ) -> ExpertTypeOutput:
        if not outputs:
            zeros = np.zeros(candidate_count, dtype=np.float64)
            missing = np.zeros(candidate_count, dtype=bool)
            return ExpertTypeOutput(type_id, zeros, missing, {}, {})

        accumulated = np.zeros(candidate_count, dtype=np.float64)
        counts = np.zeros(candidate_count, dtype=np.int32)
        member_scores: dict[str, np.ndarray] = {}
        member_available: dict[str, np.ndarray] = {}

        for name, output in outputs:
            output.validate(candidate_count)
            available = np.asarray(output.available, dtype=bool)
            raw = np.asarray(output.score, dtype=np.float64)
            z = query_standardize(
                torch.as_tensor(raw, dtype=torch.float64),
                torch.as_tensor(available, dtype=torch.bool),
            ).cpu().numpy()
            accumulated[available] += z[available]
            counts[available] += 1
            member_scores[name] = z
            member_available[name] = available

        available = counts > 0
        score = np.zeros(candidate_count, dtype=np.float64)
        score[available] = accumulated[available] / counts[available]
        return ExpertTypeOutput(
            type_id=type_id,
            score=score,
            available=available,
            member_scores=member_scores,
            member_available=member_available,
        )

    def score_type(
        self,
        type_id: str,
        *,
        query_id: str,
        candidate_ids: list[str],
    ) -> ExpertTypeOutput:
        modules = self._build(type_id)
        outputs: list[tuple[str, EvidenceOutput]] = []
        for module in modules:
            result = module.score(
                direction="r2e",
                query_id=str(query_id),
                candidate_ids=candidate_ids,
            )
            outputs.append((str(module.name), result))
        return self._aggregate_type(type_id, outputs, len(candidate_ids))

    def score_all(
        self,
        *,
        query_id: str,
        candidate_ids: list[str],
    ) -> dict[str, ExpertTypeOutput]:
        return {
            type_id: self.score_type(
                type_id,
                query_id=str(query_id),
                candidate_ids=candidate_ids,
            )
            for type_id in self.enabled_types
        }

    def audit(
        self,
        *,
        query_id: str,
        candidate_ids: list[str],
    ) -> dict[str, object]:
        outputs = self.score_all(query_id=query_id, candidate_ids=candidate_ids)
        return {
            "query_id": str(query_id),
            "candidate_count": len(candidate_ids),
            "types": {
                type_id: {
                    "supported_candidates": output.supported_candidates,
                    "members": {
                        name: int(mask.sum())
                        for name, mask in output.member_available.items()
                    },
                }
                for type_id, output in outputs.items()
            },
        }
