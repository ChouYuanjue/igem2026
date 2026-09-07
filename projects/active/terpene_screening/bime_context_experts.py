from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateContextExpert:
    """Optional candidate-level evidence source consumed by BiME-Rank routing.

    ``scores`` is always aligned to ``candidate_ids`` when available. Missing context
    is represented by ``available=False`` rather than fabricated zero scores, so callers
    can preserve exact zero-shot behavior.
    """

    name: str
    candidate_ids: tuple[str, ...]
    scores: np.ndarray | None
    available: bool
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.available != (self.scores is not None):
            raise ValueError("available must agree with whether scores are present")
        if self.scores is not None:
            arr = np.asarray(self.scores)
            if arr.ndim != 1 or len(arr) != len(self.candidate_ids):
                raise ValueError("context expert scores must be a 1D vector aligned to candidate_ids")
            if not np.isfinite(arr).all():
                raise ValueError("context expert scores must be finite")

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    def audit_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": bool(self.available),
            "source": self.source,
            "candidate_count": self.candidate_count,
            **dict(self.metadata),
        }


def unavailable_context_expert(
    name: str,
    candidate_ids: Sequence[str],
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> CandidateContextExpert:
    return CandidateContextExpert(
        name=name,
        candidate_ids=tuple(map(str, candidate_ids)),
        scores=None,
        available=False,
        source=source,
        metadata=dict(metadata or {}),
    )


def wrap_context_scores(
    name: str,
    candidate_ids: Sequence[str],
    scores: np.ndarray | None,
    *,
    source: str,
    metadata: dict[str, Any] | None = None,
) -> CandidateContextExpert:
    if scores is None:
        return unavailable_context_expert(name, candidate_ids, source=source, metadata=metadata)
    return CandidateContextExpert(
        name=name,
        candidate_ids=tuple(map(str, candidate_ids)),
        scores=np.asarray(scores, dtype=np.float32),
        available=True,
        source=source,
        metadata=dict(metadata or {}),
    )


def protein_seed_similarity_expert(
    protein_features: np.ndarray,
    protein_ids: Sequence[str],
    seed_ids: Iterable[str],
) -> CandidateContextExpert:
    ids = list(map(str, protein_ids))
    index = {value: i for i, value in enumerate(ids)}
    seeds = [str(value) for value in seed_ids if str(value) in index]
    if not seeds:
        return unavailable_context_expert(
            "seed_sequence_similarity",
            ids,
            source="known_enzyme_ids",
            metadata={"seed_count": 0},
        )
    rows = np.asarray([index[value] for value in seeds], dtype=np.int64)
    scores = (np.asarray(protein_features, dtype=np.float32) @ np.asarray(protein_features, dtype=np.float32)[rows].T).max(axis=1)
    return wrap_context_scores(
        "seed_sequence_similarity",
        ids,
        scores,
        source="known_enzyme_ids",
        metadata={"seed_count": len(seeds), "seed_ids": tuple(seeds), "aggregation": "max_cosine"},
    )


def reaction_seed_similarity_expert(
    reaction_embedding_sets: Sequence[np.ndarray],
    reaction_ids: Sequence[str],
    seed_ids: Iterable[str],
) -> CandidateContextExpert:
    ids = list(map(str, reaction_ids))
    index = {value: i for i, value in enumerate(ids)}
    seeds = [str(value) for value in seed_ids if str(value) in index]
    if not seeds:
        return unavailable_context_expert(
            "seed_reaction_similarity",
            ids,
            source="known_reaction_ids",
            metadata={"seed_count": 0},
        )
    if not reaction_embedding_sets:
        raise ValueError("reaction_embedding_sets must not be empty")
    rows = np.asarray([index[value] for value in seeds], dtype=np.int64)
    accumulated = np.zeros(len(ids), dtype=np.float32)
    for embeddings in reaction_embedding_sets:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(ids):
            raise ValueError("reaction embedding set is not aligned to reaction_ids")
        accumulated += (emb @ emb[rows].T).max(axis=1)
    scores = accumulated / float(len(reaction_embedding_sets))
    return wrap_context_scores(
        "seed_reaction_similarity",
        ids,
        scores,
        source="known_reaction_ids",
        metadata={"seed_count": len(seeds), "seed_ids": tuple(seeds), "aggregation": "mean_of_member_max_cosine"},
    )
