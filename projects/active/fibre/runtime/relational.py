from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from projects.active.fibre.model.assets import FibreAssetStore
from projects.active.fibre.model.checkpoint import load_fibre_checkpoint
from projects.active.fibre.model.index import FibreCandidateIndex


@dataclass(frozen=True)
class RankedCandidate:
    identifier: str
    score: float


class FibreRelationalRuntime:
    """Two-stage open-world retrieval with one final FIBRE relation score.

    Stage 1 scans the full registered universe with a standard dual-encoder index.
    Stage 2 discards index scores and reranks the shortlist exclusively with the
    ERAM-style FIBRE relation model and its currently available plug-in evidence.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        index_checkpoint: str | Path | None = None,
        device: str | torch.device = "cuda",
        shortlist: int = 4096,
        relation_batch: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.shortlist = int(shortlist)
        self.relation_batch = int(relation_batch)
        self.model, self.payload = load_fibre_checkpoint(
            checkpoint,
            device=self.device,
            eval_mode=True,
        )
        self.assets = FibreAssetStore()
        index_kwargs = {"device": self.device}
        if index_checkpoint is not None:
            index_kwargs["checkpoint"] = index_checkpoint
        self.index = FibreCandidateIndex(**index_kwargs)

    @torch.no_grad()
    def _scores(
        self,
        protein_ids: list[str],
        reaction_ids: list[str],
    ) -> np.ndarray:
        values: list[np.ndarray] = []
        names = set(self.model.expert_adapters.keys())
        # Context is query-time evidence and is absent unless a caller explicitly
        # supplies it through a higher-level workflow.
        names.discard("context")
        for start in range(0, len(protein_ids), self.relation_batch):
            p = protein_ids[start : start + self.relation_batch]
            r = reaction_ids[start : start + self.relation_batch]
            kwargs = self.assets.batch(
                p,
                r,
                self.device,
                expert_names=names,
            )
            score = self.model.score(**kwargs)
            values.append(score.float().cpu().numpy())
        return np.concatenate(values, axis=0)

    def reaction_to_enzyme(
        self,
        reaction_id: str,
        *,
        top_k: int = 20,
    ) -> list[RankedCandidate]:
        candidates, _ = self.index.proteins_for_reaction(
            reaction_id,
            k=self.shortlist,
        )
        scores = self._scores(
            candidates,
            [str(reaction_id)] * len(candidates),
        )
        order = np.argsort(-scores, kind="stable")[: int(top_k)]
        return [
            RankedCandidate(candidates[int(i)], float(scores[int(i)]))
            for i in order
        ]

    def enzyme_to_reaction(
        self,
        protein_id: str,
        *,
        top_k: int = 20,
    ) -> list[RankedCandidate]:
        candidates, _ = self.index.reactions_for_protein(
            protein_id,
            k=self.shortlist,
        )
        # UniMol cannot represent a tiny fraction of exceptionally large or
        # invalid reactions. Keep the full-universe index broad, but only pass
        # relation-core-supported candidates to ERAM reranking.
        candidates = [
            rid for rid in candidates if self.assets.tokens.available(rid)
        ]
        scores = self._scores(
            [str(protein_id)] * len(candidates),
            candidates,
        )
        order = np.argsort(-scores, kind="stable")[: int(top_k)]
        return [
            RankedCandidate(candidates[int(i)], float(scores[int(i)]))
            for i in order
        ]
