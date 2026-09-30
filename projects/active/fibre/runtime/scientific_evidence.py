from __future__ import annotations

import json
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
ScoreDirection = Literal["higher_is_better", "lower_is_better"]


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
    score_direction: ScoreDirection = "higher_is_better"

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
        if self.score_direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError(f"unknown evidence score direction: {self.score_direction}")
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
        if descriptor.score_direction == "lower_is_better":
            work["score"] = -work["score"]
        if "available" in work.columns:
            work["available"] = _parse_available(work["available"])
        else:
            work["available"] = True
        if "quality" in work.columns:
            work["quality"] = pd.to_numeric(work["quality"], errors="raise").astype(float)
        elif descriptor.quality_semantics is not None:
            raise ValueError(
                "declared quality semantics require a quality column at runtime"
            )
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
        return cls(
            pd.read_csv(
                path,
                dtype={
                    "direction": str,
                    "query_id": str,
                    "candidate_id": str,
                },
            ),
            descriptor,
        )

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
            requested = pd.Index([str(value) for value in candidate_ids])
            aligned = group.reindex(requested)
            row_available = (
                aligned["available"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            )
            available[:] = row_available
            if row_available.any():
                values[row_available] = aligned.loc[
                    row_available,
                    "score",
                ].to_numpy(np.float64)
                if quality is not None and "quality" in group.columns:
                    quality[row_available] = aligned.loc[
                        row_available,
                        "quality",
                    ].to_numpy(np.float64)
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
    baseline_id: str = ""

    def validate(self) -> None:
        self.descriptor.validate()
        if not np.isfinite(self.strength) or not np.isfinite(self.quality_slope):
            raise ValueError("evidence reliability parameters must be finite")
        if self.strength < 0 or self.quality_slope < 0:
            raise ValueError("evidence reliability parameters must be non-negative")
        if self.quality_slope > 0 and self.descriptor.quality_semantics is None:
            raise ValueError("quality_slope requires declared quality semantics")


@dataclass(frozen=True)
class AdmittedEvidenceBundle:
    """Jointly calibrated evidence modules sharing one frozen admission fit."""

    bundle_id: str
    members: tuple[AdmittedEvidence, ...]
    baseline_id: str

    def validate(self) -> None:
        if not self.bundle_id.strip():
            raise ValueError("scientific evidence bundle_id is required")
        if not self.baseline_id.strip():
            raise ValueError("scientific evidence bundle baseline_id is required")
        if not self.members:
            raise ValueError("scientific evidence bundle must contain members")
        names: set[str] = set()
        for member in self.members:
            member.validate()
            if member.descriptor.name in names:
                raise ValueError(
                    f"duplicate scientific evidence bundle member: "
                    f"{member.descriptor.name}"
                )
            names.add(member.descriptor.name)


def load_evidence_descriptor(path: Path) -> EvidenceDescriptor:
    """Load evidence metadata without requiring the module to pass alone.

    A joint bundle may admit a source whose value only appears conditionally
    with another source, so descriptor loading is intentionally separate from
    single-module admission loading.
    """

    payload = json.loads(Path(path).read_text())
    descriptor_payload = dict(payload.get("descriptor") or payload)
    descriptor = EvidenceDescriptor(
        name=str(descriptor_payload["name"]),
        kind=descriptor_payload["kind"],
        role=descriptor_payload["role"],
        directions=tuple(descriptor_payload["directions"]),
        score_semantics=str(descriptor_payload["score_semantics"]),
        availability_semantics=str(descriptor_payload["availability_semantics"]),
        quality_semantics=descriptor_payload.get("quality_semantics"),
        provenance=str(descriptor_payload["provenance"]),
        score_direction=descriptor_payload.get(
            "score_direction",
            "higher_is_better",
        ),
    )
    descriptor.validate()
    return descriptor


def load_admitted_evidence(path: Path) -> AdmittedEvidence:
    """Load one frozen cross-fit admission result for runtime use."""

    payload = json.loads(Path(path).read_text())
    descriptor = load_evidence_descriptor(path)
    final = dict(payload["final"])
    if not bool(final.get("admitted", False)):
        raise ValueError(f"evidence admission did not pass: {path}")
    admitted = AdmittedEvidence(
        descriptor=descriptor,
        strength=float(final["strength"]),
        quality_slope=float(final.get("quality_slope", 0.0)),
        baseline_id=str(dict(payload.get("baseline") or {}).get("id") or ""),
    )
    admitted.validate()
    return admitted


def load_admitted_evidence_bundle(path: Path) -> AdmittedEvidenceBundle:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "fibre-scientific-evidence-bundle-admission-v1":
        raise ValueError(f"unsupported scientific evidence bundle schema: {path}")
    final = dict(payload.get("final") or {})
    if not bool(final.get("admitted", False)):
        raise ValueError(f"scientific evidence bundle did not pass: {path}")
    members: list[AdmittedEvidence] = []
    for entry in payload.get("members") or []:
        descriptor_payload = dict(entry["descriptor"])
        descriptor = EvidenceDescriptor(
            name=str(descriptor_payload["name"]),
            kind=descriptor_payload["kind"],
            role=descriptor_payload["role"],
            directions=tuple(descriptor_payload["directions"]),
            score_semantics=str(descriptor_payload["score_semantics"]),
            availability_semantics=str(descriptor_payload["availability_semantics"]),
            quality_semantics=descriptor_payload.get("quality_semantics"),
            provenance=str(descriptor_payload["provenance"]),
            score_direction=descriptor_payload.get(
                "score_direction",
                "higher_is_better",
            ),
        )
        members.append(
            AdmittedEvidence(
                descriptor=descriptor,
                strength=float(entry["strength"]),
                quality_slope=float(entry.get("quality_slope", 0.0)),
            )
        )
    bundle = AdmittedEvidenceBundle(
        bundle_id=str(payload.get("bundle_id") or ""),
        members=tuple(members),
        baseline_id=str(dict(payload.get("baseline") or {}).get("id") or ""),
    )
    bundle.validate()
    return bundle


def apply_tabular_scientific_evidence(
    core_score: np.ndarray,
    candidate_ids: list[str],
    *,
    direction: Direction,
    query_id: str,
    evidence_csvs: list[Path],
    admission_jsons: list[Path],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[AdmittedEvidence]]:
    """Apply admitted local/third-party evidence to a complete candidate score vector.

    The candidate universe is owned by the core retriever.  Evidence rows absent
    from a module are neutral, so a sparse laboratory table can safely modify only
    the pairs for which it has a real scientific signal.
    """

    if len(evidence_csvs) != len(admission_jsons):
        raise ValueError(
            "scientific evidence CSV and admission JSON counts must match"
        )
    if not evidence_csvs:
        return np.asarray(core_score, dtype=np.float64).copy(), {}, []

    outputs: list[EvidenceOutput] = []
    admitted: list[AdmittedEvidence] = []
    names: set[str] = set()
    for evidence_path, admission_path in zip(
        evidence_csvs,
        admission_jsons,
        strict=True,
    ):
        registration = load_admitted_evidence(admission_path)
        if registration.descriptor.name in names:
            raise ValueError(
                f"duplicate scientific evidence module name: "
                f"{registration.descriptor.name}"
            )
        names.add(registration.descriptor.name)
        if direction not in registration.descriptor.directions:
            raise ValueError(
                f"{registration.descriptor.name} does not support {direction}"
            )
        module = TabularEvidenceModule.from_csv(
            evidence_path,
            registration.descriptor,
        )
        outputs.append(
            module.score(
                direction=direction,
                query_id=str(query_id),
                candidate_ids=candidate_ids,
            )
        )
        admitted.append(registration)

    fused, contributions = fuse_admitted_evidence(
        np.asarray(core_score, dtype=np.float64),
        outputs,
        admitted,
    )
    return fused, contributions, admitted


def apply_tabular_scientific_evidence_bundle(
    core_score: np.ndarray,
    candidate_ids: list[str],
    *,
    direction: Direction,
    query_id: str,
    evidence_csvs: list[Path],
    bundle_json: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], AdmittedEvidenceBundle]:
    """Apply evidence coefficients fitted jointly against the same frozen core.

    Joint admission is required when more than one evidence source changes the
    same ranking. This prevents independently admitted, correlated predictors
    from being counted twice merely because each helped against the bare core.
    """

    bundle = load_admitted_evidence_bundle(bundle_json)
    if len(evidence_csvs) != len(bundle.members):
        raise ValueError(
            "scientific evidence CSV count must match jointly admitted bundle members"
        )
    outputs: list[EvidenceOutput] = []
    for evidence_path, registration in zip(
        evidence_csvs,
        bundle.members,
        strict=True,
    ):
        if direction not in registration.descriptor.directions:
            raise ValueError(
                f"{registration.descriptor.name} does not support {direction}"
            )
        module = TabularEvidenceModule.from_csv(
            evidence_path,
            registration.descriptor,
        )
        outputs.append(
            module.score(
                direction=direction,
                query_id=str(query_id),
                candidate_ids=candidate_ids,
            )
        )
    fused, contributions = fuse_admitted_evidence(
        np.asarray(core_score, dtype=np.float64),
        outputs,
        list(bundle.members),
    )
    return fused, contributions, bundle


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
