from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from projects.active.fibre.model.assets import FibreAssetStore
from projects.active.fibre.model.checkpoint import load_fibre_checkpoint
from projects.active.fibre.model.fibre import ExpertEvidence
from projects.active.fibre.model.plugins import attach_expert_plugin
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
        expert_plugins: list[str | Path] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.shortlist = int(shortlist)
        self.relation_batch = int(relation_batch)
        self.model, self.payload = load_fibre_checkpoint(
            checkpoint,
            device=self.device,
            eval_mode=True,
        )
        for plugin in expert_plugins or []:
            attach_expert_plugin(self.model, plugin, device=self.device)
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
        *,
        context_scores: np.ndarray | None = None,
        context_available: np.ndarray | None = None,
    ) -> np.ndarray:
        values: list[np.ndarray] = []
        names = set(self.model.expert_adapters.keys())
        names.discard("context")
        if context_scores is not None and "context" not in self.model.expert_adapters:
            raise RuntimeError("context scores supplied without an attached context expert plugin")
        if context_scores is not None and len(context_scores) != len(protein_ids):
            raise ValueError("context scores must align to scored pairs")
        if context_available is not None and len(context_available) != len(protein_ids):
            raise ValueError("context availability must align to scored pairs")
        for start in range(0, len(protein_ids), self.relation_batch):
            p = protein_ids[start : start + self.relation_batch]
            r = reaction_ids[start : start + self.relation_batch]
            kwargs = self.assets.batch(
                p,
                r,
                self.device,
                expert_names=names,
            )
            if context_scores is not None:
                local = np.asarray(context_scores[start : start + len(p)], dtype=np.float32)
                local_available = (
                    np.ones(len(local), dtype=bool)
                    if context_available is None
                    else np.asarray(context_available[start : start + len(p)], dtype=bool)
                )
                kwargs["evidence"]["context"] = ExpertEvidence(
                    torch.as_tensor(local[:, None], device=self.device),
                    torch.as_tensor(local_available, device=self.device),
                )
            score = self.model.score(**kwargs)
            values.append(score.float().cpu().numpy())
        return np.concatenate(values, axis=0)

    def reaction_to_enzyme(
        self,
        reaction_id: str,
        *,
        top_k: int = 20,
        context_scores: Mapping[str, float] | None = None,
    ) -> list[RankedCandidate]:
        candidates, base_scores = self.index.proteins_for_reaction(
            reaction_id,
            k=self.shortlist,
        )
        if not self.assets.tokens.available(reaction_id):
            return [
                RankedCandidate(candidate, float(score))
                for candidate, score in zip(candidates[: int(top_k)], base_scores[: int(top_k)], strict=True)
            ]
        ctx = None
        ctx_ok = None
        if context_scores is not None:
            ctx = np.asarray([float(context_scores.get(x, 0.0)) for x in candidates], dtype=np.float32)
            ctx_ok = np.asarray([x in context_scores for x in candidates], dtype=bool)
        scores = self._scores(
            candidates,
            [str(reaction_id)] * len(candidates),
            context_scores=ctx,
            context_available=ctx_ok,
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
        context_scores: Mapping[str, float] | None = None,
    ) -> list[RankedCandidate]:
        candidates, base_scores = self.index.reactions_for_protein(
            protein_id,
            k=self.shortlist,
        )
        supported_slots = [
            i for i, rid in enumerate(candidates) if self.assets.tokens.available(rid)
        ]
        supported_ids = [candidates[i] for i in supported_slots]
        relation_scores = (
            self._scores(
                [str(protein_id)] * len(supported_ids),
                supported_ids,
                context_scores=(
                    np.asarray([float(context_scores.get(x, 0.0)) for x in supported_ids], dtype=np.float32)
                    if context_scores is not None else None
                ),
                context_available=(
                    np.asarray([x in context_scores for x in supported_ids], dtype=bool)
                    if context_scores is not None else None
                ),
            )
            if supported_ids
            else np.empty(0, dtype=np.float32)
        )
        relation_order = np.argsort(-relation_scores, kind="stable")
        final = list(candidates)
        score_map = {
            supported_ids[int(local)]: float(relation_scores[int(local)])
            for local in range(len(supported_ids))
        }
        for slot, local in zip(supported_slots, relation_order, strict=True):
            final[slot] = supported_ids[int(local)]
        base_map = {candidate: float(score) for candidate, score in zip(candidates, base_scores, strict=True)}
        return [
            RankedCandidate(candidate, score_map.get(candidate, base_map[candidate]))
            for candidate in final[: int(top_k)]
        ]
