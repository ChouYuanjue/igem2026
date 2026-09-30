from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.runtime.scientific_evidence import (
    AdmittedEvidence,
    EvidenceOutput,
    TabularEvidenceModule,
    load_evidence_descriptor,
    query_zscore,
)
from reproducibility.bime_rank.support.evidence_admission import (
    bootstrap_mean_interval,
    core_score_signature,
    fit_nonnegative_pairwise_logistic,
)


REQUIRED_CORE = ("query_id", "candidate_id", "core_score", "label")


def stable_fold(value: str, folds: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor_payload(member: AdmittedEvidence) -> dict[str, object]:
    descriptor = member.descriptor
    return {
        "name": descriptor.name,
        "kind": descriptor.kind,
        "role": descriptor.role,
        "directions": list(descriptor.directions),
        "score_semantics": descriptor.score_semantics,
        "availability_semantics": descriptor.availability_semantics,
        "quality_semantics": descriptor.quality_semantics,
        "provenance": descriptor.provenance,
        "score_direction": descriptor.score_direction,
    }


def prepare_query(
    group: pd.DataFrame,
    modules: list[TabularEvidenceModule],
    members: list[AdmittedEvidence],
    *,
    direction: str,
) -> dict[str, np.ndarray] | None:
    labels = group["label"].to_numpy(np.int8)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) == 0 or len(negatives) == 0:
        return None

    candidate_ids = group["candidate_id"].astype(str).tolist()
    query_id = str(group["query_id"].iloc[0])
    feature_columns: list[np.ndarray] = []
    for module, member in zip(modules, members, strict=True):
        calibrated = query_zscore(
            module.score(
                direction=direction,
                query_id=query_id,
                candidate_ids=candidate_ids,
            )
        )
        feature_columns.append(calibrated.score)
        if member.descriptor.quality_semantics is not None:
            if calibrated.quality is None:
                raise RuntimeError(
                    f"{member.descriptor.name} declared quality but produced none"
                )
            feature_columns.append(
                calibrated.score * np.asarray(calibrated.quality, dtype=np.float64)
            )

    features = np.column_stack(feature_columns)
    core = group["core_score"].to_numpy(np.float64)
    core_diff = (core[positives, None] - core[None, negatives]).reshape(-1)
    evidence_diff = (
        features[positives, None, :] - features[None, negatives, :]
    ).reshape(-1, features.shape[1])
    weight = np.full(
        len(core_diff),
        1.0 / (len(positives) * len(negatives)),
        dtype=np.float64,
    )
    return {
        "core_diff": core_diff,
        "evidence_diff": evidence_diff,
        "weight": weight,
    }


def stack_queries(
    frame: pd.DataFrame,
    modules: list[TabularEvidenceModule],
    members: list[AdmittedEvidence],
    *,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cores: list[np.ndarray] = []
    evidence: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used_queries = 0
    for _, group in frame.groupby("query_id", sort=False):
        prepared = prepare_query(
            group,
            modules,
            members,
            direction=direction,
        )
        if prepared is None:
            continue
        used_queries += 1
        cores.append(prepared["core_diff"])
        evidence.append(prepared["evidence_diff"])
        weights.append(prepared["weight"])
    if not cores:
        raise ValueError("no query contains both explicit positive and explicit negative outcomes")
    return (
        np.concatenate(cores),
        np.concatenate(evidence),
        np.concatenate(weights),
        used_queries,
    )


def pairwise_loss(
    core: np.ndarray,
    evidence: np.ndarray,
    coef: np.ndarray,
    weight: np.ndarray,
) -> float:
    margin = core + evidence @ coef
    normalized = weight / weight.sum()
    return float(np.sum(normalized * np.logaddexp(0.0, -margin)))


def query_improvements(
    frame: pd.DataFrame,
    modules: list[TabularEvidenceModule],
    members: list[AdmittedEvidence],
    *,
    direction: str,
    coefficients: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        prepared = prepare_query(
            group,
            modules,
            members,
            direction=direction,
        )
        if prepared is None:
            continue
        core = prepared["core_diff"]
        evidence = prepared["evidence_diff"]
        weight = prepared["weight"]
        zero = np.zeros(evidence.shape[1], dtype=np.float64)
        values.append(
            pairwise_loss(core, evidence, zero, weight)
            - pairwise_loss(core, evidence, coefficients, weight)
        )
    return np.asarray(values, dtype=np.float64)


def split_member_coefficients(
    coefficients: np.ndarray,
    members: list[AdmittedEvidence],
) -> list[tuple[float, float]]:
    offset = 0
    values: list[tuple[float, float]] = []
    for member in members:
        strength = float(coefficients[offset])
        offset += 1
        quality_slope = 0.0
        if member.descriptor.quality_semantics is not None:
            quality_slope = float(coefficients[offset])
            offset += 1
        values.append((strength, quality_slope))
    if offset != len(coefficients):
        raise RuntimeError("joint evidence coefficient layout drifted")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly cross-fit multiple already-defined scientific evidence modules "
            "against one frozen FIBRE core, preventing correlated modules from being "
            "independently double-counted."
        )
    )
    parser.add_argument("--core-csv", type=Path, required=True)
    parser.add_argument(
        "--evidence-csv",
        type=Path,
        action="append",
        default=[],
        help="Runtime-format evidence CSV. Repeat in bundle member order.",
    )
    parser.add_argument(
        "--member-spec",
        "--member-admission",
        dest="member_spec",
        type=Path,
        action="append",
        default=[],
        help=(
            "JSON carrying the member descriptor. It may be a standalone descriptor "
            "or a previous admission result; previous coefficients/status are ignored."
        ),
    )
    parser.add_argument("--direction", choices=("r2e", "e2r"), required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--l1",
        type=float,
        default=1e-3,
        help=(
            "Non-negative L1 penalty; duplicating an evidence channel cannot "
            "reduce its total regularization cost."
        ),
    )
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument(
        "--baseline-id",
        default=None,
        help=(
            "Identifier for the exact frozen core score source used in --core-csv. "
            "If omitted, the core CSV SHA256 is used."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if len(args.evidence_csv) < 2:
        raise ValueError("joint admission requires at least two evidence modules")
    if len(args.evidence_csv) != len(args.member_spec):
        raise ValueError("--evidence-csv and --member-spec counts must match")

    members = [
        AdmittedEvidence(
            descriptor=load_evidence_descriptor(path),
            strength=0.0,
        )
        for path in args.member_spec
    ]
    names = [member.descriptor.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("joint evidence bundle member names must be unique")
    for member in members:
        if args.direction not in member.descriptor.directions:
            raise ValueError(
                f"{member.descriptor.name} does not support {args.direction}"
            )

    modules = [
        TabularEvidenceModule.from_csv(path, member.descriptor)
        for path, member in zip(args.evidence_csv, members, strict=True)
    ]

    frame = pd.read_csv(
        args.core_csv,
        dtype={"query_id": str, "candidate_id": str},
    )
    missing = [name for name in REQUIRED_CORE if name not in frame.columns]
    if missing:
        raise ValueError(f"core training table missing columns: {missing}")
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["core_score"] = pd.to_numeric(
        frame["core_score"],
        errors="raise",
    ).astype(float)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain explicit 0/1 experimental outcomes")
    if not np.isfinite(frame["core_score"]).all():
        raise ValueError("core_score must be finite")
    if frame.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("core training table contains duplicate query/candidate rows")

    if "fold" in frame.columns:
        if frame.groupby("query_id")["fold"].nunique().max() != 1:
            raise ValueError("all candidates of one query must share a fold")
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    else:
        frame["fold"] = frame["query_id"].map(
            {
                query: stable_fold(query, args.folds)
                for query in frame["query_id"].unique()
            }
        )

    folds = sorted(set(int(value) for value in frame["fold"].unique()))
    if len(folds) < 2:
        raise ValueError("joint cross-fitting requires at least two populated folds")

    crossfit: list[dict[str, object]] = []
    oof_query_improvements: list[np.ndarray] = []
    for holdout in folds:
        train = frame[frame["fold"] != holdout]
        test = frame[frame["fold"] == holdout]
        core_train, evidence_train, weight_train, train_queries = stack_queries(
            train,
            modules,
            members,
            direction=args.direction,
        )
        fit = fit_nonnegative_pairwise_logistic(
            core_train,
            evidence_train,
            sample_weight=weight_train,
            l1=float(args.l1),
            l2=float(args.l2),
        )
        core_test, evidence_test, weight_test, test_queries = stack_queries(
            test,
            modules,
            members,
            direction=args.direction,
        )
        zero = np.zeros(evidence_test.shape[1], dtype=np.float64)
        core_loss = pairwise_loss(core_test, evidence_test, zero, weight_test)
        fused_loss = pairwise_loss(
            core_test,
            evidence_test,
            fit.coefficients,
            weight_test,
        )
        local_query_improvement = query_improvements(
            test,
            modules,
            members,
            direction=args.direction,
            coefficients=fit.coefficients,
        )
        oof_query_improvements.append(local_query_improvement)
        crossfit.append(
            {
                "holdout": int(holdout),
                "train_queries": int(train_queries),
                "test_queries": int(test_queries),
                "coefficients": fit.coefficients.tolist(),
                "core_pairwise_log_loss": core_loss,
                "fused_pairwise_log_loss": fused_loss,
                "improvement": core_loss - fused_loss,
                "query_mean_improvement": float(
                    local_query_improvement.mean()
                ),
                "query_fraction_positive": float(
                    np.mean(local_query_improvement > 0.0)
                ),
            }
        )

    core_all, evidence_all, weight_all, queries_all = stack_queries(
        frame,
        modules,
        members,
        direction=args.direction,
    )
    final_fit = fit_nonnegative_pairwise_logistic(
        core_all,
        evidence_all,
        sample_weight=weight_all,
        l1=float(args.l1),
        l2=float(args.l2),
    )
    improvements = np.asarray(
        [row["improvement"] for row in crossfit],
        dtype=np.float64,
    )
    member_coefficients = split_member_coefficients(
        final_fit.coefficients,
        members,
    )
    query_stability = bootstrap_mean_interval(
        np.concatenate(oof_query_improvements),
    )
    admitted = bool(
        np.all(improvements > 0.0)
        and query_stability.lower_95 > 0.0
        and any(
            strength > 0.0 or quality_slope > 0.0
            for strength, quality_slope in member_coefficients
        )
    )

    core_signature = core_score_signature(frame)
    core_file_sha256 = file_sha256(args.core_csv)
    identity = {
        "direction": args.direction,
        "core_score_sha256": core_signature,
        "evidence_sha256": [file_sha256(path) for path in args.evidence_csv],
        "member_names": names,
    }
    bundle_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    baseline_id = str(
        args.baseline_id
        or f"core-score-sha256:{core_signature}"
    )

    payload = {
        "schema": "fibre-scientific-evidence-bundle-admission-v1",
        "bundle_id": bundle_id,
        "direction": args.direction,
        "baseline": {
            "id": baseline_id,
            "core_score_sha256": core_signature,
            "core_csv_sha256": core_file_sha256,
            "meaning": (
                "All member coefficients are jointly valid only for this frozen "
                "core score semantics."
            ),
        },
        "regularization": {
            "l1": float(args.l1),
            "l2": float(args.l2),
            "reason": (
                "The bundle defaults to L1 because, with non-negative additive "
                "strengths, duplicating a channel cannot lower the total L1 cost."
            ),
        },
        "data": {
            "core_csv": str(args.core_csv),
            "core_score_sha256": core_signature,
            "core_csv_sha256": core_file_sha256,
            "rows": int(len(frame)),
            "queries": int(frame["query_id"].nunique()),
            "usable_queries": int(queries_all),
            "explicit_positive_rows": int((frame["label"] == 1).sum()),
            "explicit_negative_rows": int((frame["label"] == 0).sum()),
            "evidence_csvs": [
                {
                    "path": str(path),
                    "sha256": digest,
                }
                for path, digest in zip(
                    args.evidence_csv,
                    identity["evidence_sha256"],
                    strict=True,
                )
            ],
        },
        "members": [
            {
                "descriptor": descriptor_payload(member),
                "strength": strength,
                "quality_slope": quality_slope,
                "active": bool(strength > 0.0 or quality_slope > 0.0),
            }
            for member, (strength, quality_slope) in zip(
                members,
                member_coefficients,
                strict=True,
            )
        ],
        "crossfit": crossfit,
        "oof_query_stability": {
            "n": query_stability.n,
            "mean_improvement": query_stability.mean,
            "median_improvement": query_stability.median,
            "fraction_positive": query_stability.fraction_positive,
            "bootstrap_95": [
                query_stability.lower_95,
                query_stability.upper_95,
            ],
            "bootstrap_seed": query_stability.seed,
            "bootstrap_replicates": query_stability.replicates,
        },
        "final": {
            "coefficients": final_fit.coefficients.tolist(),
            "all_holdouts_improved_pairwise_log_loss": bool(
                np.all(improvements > 0.0)
            ),
            "query_bootstrap_lower_95_positive": bool(
                query_stability.lower_95 > 0.0
            ),
            "admitted": admitted,
        },
        "note": (
            "All member coefficients are fit jointly against the same frozen core. "
            "This bundle, rather than independently fitted member weights, must be "
            "used when multiple evidence modules change the same ranking."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
