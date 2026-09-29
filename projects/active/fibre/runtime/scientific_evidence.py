from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd


EvidenceKind = Literal[
    "molecular_view",
    "mechanistic",
    "structural",
    "experimental_context",
]
EvidenceRole = Literal["rerank", "retrieve_and_rerank"]
Direction = Literal["r2e", "e2r"]


def _parse_available(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    text = values.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
        "": False,
        "nan": False,
        "none": False,
    }
    unknown = sorted(set(text) - set(mapping))
    if unknown:
        raise ValueError(
            f"evidence availability contains unsupported boolean values: {unknown[:5]}"
        )
    return text.map(mapping).astype(bool)


@dataclass(frozen=True)
class EvidenceOutput:
    """One evidence module evaluated on one query/candidate set."""

    score: np.ndarray
    available: np.ndarray
    quality: np.ndarray | None = None

    def validate(self, candidate_count: int) -> None:
        score = np.asarray(self.score)
        available = np.asarray(self.available)
        if score.shape != (candidate_count,):
            raise ValueError("evidence score must align with candidate_ids")
        if available.shape != (candidate_count,):
            raise ValueError("evidence availability must align with candidate_ids")
        if self.quality is not None and np.asarray(self.quality).shape != (candidate_count,):
            raise ValueError("evidence quality must align with candidate_ids")
        supported = available.astype(bool)
        if supported.any() and not np.isfinite(score[supported]).all():
            raise ValueError("available evidence scores must be finite")
        if self.quality is not None:
            quality = np.asarray(self.quality)
            if supported.any() and not np.isfinite(quality[supported]).all():
                raise ValueError("available evidence quality must be finite")
            if supported.any() and (
                np.any(quality[supported] < 0.0) or np.any(quality[supported] > 1.0)
            ):
                raise ValueError("available evidence quality must lie in [0, 1]")


class ScientificEvidenceModule(Protocol):
    """Minimal self-deployment contract for evidence that changes catalytic rank.

    A module must provide a pairwise ranking signal.  Candidate constraints such
    as host, inventory or expression feasibility belong in the downstream
    experimental-decision layer and intentionally do not implement this
    protocol.
    """

    name: str
    kind: EvidenceKind
    role: EvidenceRole
    directions: tuple[Direction, ...]

    def score(
        self,
        *,
        direction: Direction,
        query_id: str,
        candidate_ids: list[str],
    ) -> EvidenceOutput:
        ...


class CandidateGenerator(Protocol):
    """Optional retrieval capability for a source that can discover candidates."""

    name: str
    directions: tuple[Direction, ...]

    def retrieve(
        self,
        *,
        direction: Direction,
        query_id: str,
        top_k: int,
    ) -> list[str]:
        ...


class CandidateConstraint(Protocol):
    """Experimental eligibility rule kept outside the catalytic score."""

    name: str

    def eligible(
        self,
        *,
        candidate_ids: list[str],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class EvidenceDescriptor:
    """Metadata required before a local scientific predictor can be admitted."""

    name: str
    kind: EvidenceKind
    role: EvidenceRole
    directions: tuple[Direction, ...]
    score_semantics: str
    availability_semantics: str
    quality_semantics: str | None
    provenance: str

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("evidence name is required")
        if self.kind not in {
            "molecular_view",
            "mechanistic",
            "structural",
            "experimental_context",
        }:
            raise ValueError(f"unknown evidence kind: {self.kind}")
        if self.role not in {"rerank", "retrieve_and_rerank"}:
            raise ValueError(f"unknown evidence role: {self.role}")
        if not self.directions:
            raise ValueError("at least one direction is required")
        if any(value not in ("r2e", "e2r") for value in self.directions):
            raise ValueError("unknown evidence direction")
        if not self.score_semantics.strip():
            raise ValueError("score semantics are required")
        if not self.availability_semantics.strip():
            raise ValueError("availability semantics are required")
        if not self.provenance.strip():
            raise ValueError("training/provenance boundary is required")


class TabularEvidenceModule:
    """Zero-code adapter for laboratory or third-party pair scores.

    Required columns are direction, query_id, candidate_id and score. An
    optional quality column may be supplied when the descriptor declares its
    semantics. Repeated query/candidate rows are rejected.
    """

    REQUIRED_COLUMNS = ("direction", "query_id", "candidate_id", "score")

    def __init__(
        self,
        frame: pd.DataFrame,
        descriptor: EvidenceDescriptor,
    ) -> None:
        descriptor.validate()
        missing = [name for name in self.REQUIRED_COLUMNS if name not in frame.columns]
        if missing:
            raise ValueError(f"evidence table missing columns: {missing}")
        work = frame.copy()
        work["direction"] = work["direction"].astype(str)
        work["query_id"] = work["query_id"].astype(str)
        work["candidate_id"] = work["candidate_id"].astype(str)
        work["score"] = pd.to_numeric(work["score"], errors="raise").astype(float)
        if "available" in work.columns:
            work["available"] = _parse_available(work["available"])
        else:
            work["available"] = True
        if "quality" in work.columns:
            work["quality"] = pd.to_numeric(work["quality"], errors="raise").astype(float)
        if work.duplicated(["direction", "query_id", "candidate_id"]).any():
            raise ValueError("evidence table contains duplicate query/candidate rows")
        if not set(work["direction"]).issubset(set(descriptor.directions)):
            raise ValueError("evidence table contains an undeclared direction")
        supported = work["available"].to_numpy(bool)
        if supported.any() and not np.isfinite(work.loc[supported, "score"]).all():
            raise ValueError("available evidence table scores must be finite")
        if (
            "quality" in work.columns
            and supported.any()
            and not np.isfinite(work.loc[supported, "quality"]).all()
        ):
            raise ValueError("available evidence table quality must be finite")
        if "quality" in work.columns and descriptor.quality_semantics is None:
            raise ValueError("quality column requires declared quality semantics")

        self.descriptor = descriptor
        self.name = descriptor.name
        self.kind = descriptor.kind
        self.role = descriptor.role
        self.directions = descriptor.directions
        self._frame = work
        self._groups = {
            (str(direction), str(query)): group.set_index("candidate_id", drop=False)
            for (direction, query), group in work.groupby(
                ["direction", "query_id"],
                sort=False,
            )
        }

    @classmethod
    def from_csv(
        cls,
        path: Path,
        descriptor: EvidenceDescriptor,
    ) -> "TabularEvidenceModule":
        return cls(pd.read_csv(path), descriptor)

    def score(
        self,
        *,
        direction: Direction,
        query_id: str,
        candidate_ids: list[str],
    ) -> EvidenceOutput:
        if direction not in self.directions:
            raise ValueError(f"{self.name} does not support direction {direction}")
        group = self._groups.get((str(direction), str(query_id)))
        values = np.zeros(len(candidate_ids), dtype=np.float64)
        available = np.zeros(len(candidate_ids), dtype=bool)
        quality = (
            np.zeros(len(candidate_ids), dtype=np.float64)
            if self.descriptor.quality_semantics is not None
            else None
        )
        if group is not None:
            row_by_id = group.index
            for index, candidate_id in enumerate(map(str, candidate_ids)):
                if candidate_id not in row_by_id:
                    continue
                row = group.loc[candidate_id]
                row_available = bool(row["available"])
                available[index] = row_available
                if not row_available:
                    continue
                values[index] = float(row["score"])
                if quality is not None and "quality" in group.columns:
                    quality[index] = float(row["quality"])
        return EvidenceOutput(values, available, quality)

    def retrieve(
        self,
        *,
        direction: Direction,
        query_id: str,
        top_k: int,
    ) -> list[str]:
        if self.role != "retrieve_and_rerank":
            raise ValueError(f"{self.name} is not registered as a candidate generator")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        group = self._groups.get((str(direction), str(query_id)))
        if group is None:
            return []
        ordered = group.loc[group["available"].astype(bool)].reset_index(drop=True).sort_values(
            ["score", "candidate_id"],
            ascending=[False, True],
            kind="stable",
        )
        return ordered["candidate_id"].astype(str).head(top_k).tolist()


@dataclass(frozen=True)
class AdmittedEvidence:
    """One validated evidence module and its frozen reliability parameters."""

    descriptor: EvidenceDescriptor
    strength: float
    quality_slope: float = 0.0

    def validate(self) -> None:
        self.descriptor.validate()
        if self.strength < 0 or self.quality_slope < 0:
            raise ValueError("evidence reliability parameters must be non-negative")
        if self.quality_slope > 0 and self.descriptor.quality_semantics is None:
            raise ValueError("quality_slope requires declared quality semantics")


def fuse_admitted_evidence(
    core_score: np.ndarray,
    outputs: list[EvidenceOutput],
    admitted: list[AdmittedEvidence],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply admitted evidence to one frozen query ranking.

    Each evidence channel is z-calibrated on its own available candidates.
    Missing evidence is exactly neutral. Channels add evidence independently;
    adding or removing one channel never renormalises any other channel.
    """

    core = np.asarray(core_score, dtype=np.float64)
    if core.ndim != 1:
        raise ValueError("core_score must be one-dimensional")
    if len(outputs) != len(admitted):
        raise ValueError("outputs and admitted metadata must align")
    fused = core.copy()
    contributions: dict[str, np.ndarray] = {}
    for output, registration in zip(outputs, admitted, strict=True):
        registration.validate()
        calibrated = query_zscore(output)
        reliability = np.full(
            len(core),
            float(registration.strength),
            dtype=np.float64,
        )
        if registration.quality_slope:
            if calibrated.quality is None:
                raise ValueError(
                    f"{registration.descriptor.name} requires quality values"
                )
            reliability += float(registration.quality_slope) * np.asarray(
                calibrated.quality,
                dtype=np.float64,
            )
        reliability = np.maximum(reliability, 0.0)
        contribution = (
            calibrated.score
            * reliability
            * calibrated.available.astype(np.float64)
        )
        fused += contribution
        contributions[registration.descriptor.name] = contribution
    return fused, contributions


def query_zscore(output: EvidenceOutput, *, eps: float = 1e-6) -> EvidenceOutput:
    """Map arbitrary expert score scales onto a common within-query scale.

    Only supported candidates define the mean and variance. Missing evidence is
    represented by score zero after calibration and remains explicitly marked
    unavailable.
    """

    score = np.asarray(output.score, dtype=np.float64)
    available = np.asarray(output.available, dtype=bool)
    output.validate(len(score))
    calibrated = np.zeros_like(score, dtype=np.float64)
    if available.any():
        local = score[available]
        std = max(float(local.std()), eps)
        calibrated[available] = (local - float(local.mean())) / std
    quality = None if output.quality is None else np.asarray(output.quality, dtype=np.float64)
    return EvidenceOutput(
        score=calibrated,
        available=available,
        quality=quality,
    )
