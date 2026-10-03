from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _row_stochastic(partition: np.ndarray, *, name: str, atol: float = 1e-8) -> np.ndarray:
    value = np.asarray(partition, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2D partition matrix")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if np.any(value < -atol):
        raise ValueError(f"{name} must be non-negative")
    value = np.maximum(value, 0.0)
    mass = value.sum(axis=1)
    if np.any(np.abs(mass - 1.0) > atol):
        raise ValueError(f"{name} rows must sum to one")
    return value


def restrict_positive_relation(
    reaction_partition: np.ndarray,
    protein_partition: np.ndarray,
    positive_pairs: np.ndarray | Iterable[tuple[int, int]],
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Tensor-product restriction P_R^T A P_E without materializing A.

    Each distinct verified positive pair is one unit atom. Because both factor
    partitions are row-stochastic, restriction conserves total empirical mass.
    Duplicate rows therefore do not create additional entropy mass.
    """
    pr = _row_stochastic(reaction_partition, name="reaction_partition")
    pe = _row_stochastic(protein_partition, name="protein_partition")
    pairs = np.asarray(list(positive_pairs), dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("positive_pairs must have shape [n,2]")
    if len(pairs) == 0:
        return np.zeros((pr.shape[1], pe.shape[1]), dtype=np.float64)
    pairs = np.unique(pairs, axis=0)
    if np.any(pairs[:, 0] < 0) or np.any(pairs[:, 0] >= pr.shape[0]):
        raise ValueError("reaction positive index out of range")
    if np.any(pairs[:, 1] < 0) or np.any(pairs[:, 1] >= pe.shape[0]):
        raise ValueError("protein positive index out of range")

    coarse = np.zeros((pr.shape[1], pe.shape[1]), dtype=np.float64)
    for r, e in pairs:
        coarse += np.outer(pr[int(r)], pe[int(e)])

    expected = float(len(pairs))
    if not np.isclose(float(coarse.sum()), expected, rtol=1e-10, atol=1e-10):
        raise RuntimeError("tensor-atlas restriction failed to conserve positive mass")
    if normalize:
        coarse /= expected
    return coarse


@dataclass(frozen=True)
class TensorAtlasField:
    """One global coarse deformation and its factor partitions."""

    reaction_partition: np.ndarray
    protein_partition: np.ndarray
    coarse_deformation: np.ndarray

    def __post_init__(self) -> None:
        pr = _row_stochastic(self.reaction_partition, name="reaction_partition")
        pe = _row_stochastic(self.protein_partition, name="protein_partition")
        dc = np.asarray(self.coarse_deformation, dtype=np.float64)
        if dc.shape != (pr.shape[1], pe.shape[1]):
            raise ValueError("coarse_deformation shape must equal (reaction_atlas_size, protein_atlas_size)")
        if not np.all(np.isfinite(dc)):
            raise ValueError("coarse_deformation must be finite")
        object.__setattr__(self, "reaction_partition", pr)
        object.__setattr__(self, "protein_partition", pe)
        object.__setattr__(self, "coarse_deformation", dc)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.reaction_partition.shape[0], self.protein_partition.shape[0])

    def value(self, reaction_index: int, protein_index: int) -> float:
        r = int(reaction_index); e = int(protein_index)
        if not 0 <= r < self.shape[0] or not 0 <= e < self.shape[1]:
            raise IndexError("tensor-atlas field index out of range")
        return float(self.reaction_partition[r] @ self.coarse_deformation @ self.protein_partition[e])

    def reaction_to_protein_section(self, reaction_index: int, protein_indices: np.ndarray | None = None) -> np.ndarray:
        r = int(reaction_index)
        if not 0 <= r < self.shape[0]:
            raise IndexError("reaction index out of range")
        ids = np.arange(self.shape[1], dtype=np.int64) if protein_indices is None else np.asarray(protein_indices, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.shape[1]):
            raise IndexError("protein candidate index out of range")
        left = self.reaction_partition[r] @ self.coarse_deformation
        return np.asarray(left @ self.protein_partition[ids].T, dtype=np.float64)

    def protein_to_reaction_section(self, protein_index: int, reaction_indices: np.ndarray | None = None) -> np.ndarray:
        e = int(protein_index)
        if not 0 <= e < self.shape[1]:
            raise IndexError("protein index out of range")
        ids = np.arange(self.shape[0], dtype=np.int64) if reaction_indices is None else np.asarray(reaction_indices, dtype=np.int64).reshape(-1)
        if np.any(ids < 0) or np.any(ids >= self.shape[0]):
            raise IndexError("reaction candidate index out of range")
        right = self.coarse_deformation @ self.protein_partition[e]
        return np.asarray(self.reaction_partition[ids] @ right, dtype=np.float64)

    def overlap_defect(self, pairs: np.ndarray | Iterable[tuple[int, int]]) -> np.ndarray:
        """Numerical R2E/E2R disagreement at identical product-space points."""
        pairs_array = np.asarray(list(pairs), dtype=np.int64)
        if pairs_array.ndim != 2 or pairs_array.shape[1] != 2:
            raise ValueError("pairs must have shape [n,2]")
        out = np.empty(len(pairs_array), dtype=np.float64)
        for i, (r, e) in enumerate(pairs_array):
            from_r = self.reaction_to_protein_section(int(r), np.asarray([e]))[0]
            from_e = self.protein_to_reaction_section(int(e), np.asarray([r]))[0]
            out[i] = from_r - from_e
        return out
