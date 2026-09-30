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
    ADMISSION_POLICY,
    SINGLE_ADMISSION_SCHEMA,
    EvidenceDescriptor,
    EvidenceOutput,
    query_zscore,
)
from reproducibility.bime_rank.support.evidence_admission import (
    bootstrap_mean_interval,
    core_score_signature,
    fit_nonnegative_pairwise_logistic,
)


REQUIRED = ("query_id", "candidate_id", "core_score", "evidence_score", "label")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool_series(values: pd.Series) -> pd.Series:
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
        raise ValueError(f"available contains unsupported boolean values: {unknown[:5]}")
    return text.map(mapping).astype(bool)


def stable_fold(value: str, folds: int) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % folds


def prepare_query(frame: pd.DataFrame, use_quality: bool) -> dict[str, np.ndarray] | None:
    labels = frame["label"].to_numpy(np.int8)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    if len(positives) == 0 or len(negatives) == 0:
        return None

    available = frame["available"].to_numpy(bool)
    quality = frame["quality"].to_numpy(float) if use_quality else None
    calibrated = query_zscore(
        EvidenceOutput(
            score=frame["evidence_score"].to_numpy(float),
            available=available,
            quality=quality,
        )
    )
    core = frame["core_score"].to_numpy(float)
    evidence = calibrated.score
    features = [evidence]
    if use_quality:
        features.append(evidence * np.asarray(calibrated.quality, dtype=float))
    feature_matrix = np.column_stack(features)

    core_diff = (core[positives, None] - core[None, negatives]).reshape(-1)
    evidence_diff = (
        feature_matrix[positives, None, :] - feature_matrix[None, negatives, :]
    ).reshape(-1, feature_matrix.shape[1])
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
    *,
    use_quality: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    cores: list[np.ndarray] = []
    evidence: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    used_queries = 0
    for _, group in frame.groupby("query_id", sort=False):
        prepared = prepare_query(group, use_quality)
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
    w = weight / weight.sum()
    return float(np.sum(w * np.logaddexp(0.0, -margin)))


def query_improvements(
    frame: pd.DataFrame,
    *,
    use_quality: bool,
    coefficients: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for _, group in frame.groupby("query_id", sort=False):
        prepared = prepare_query(group, use_quality)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit one local scientific evidence table against a frozen FIBRE core."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--kind",
        required=True,
        choices=("molecular_view", "mechanistic", "structural", "experimental_context"),
    )
    parser.add_argument("--direction", required=True, choices=("r2e", "e2r"))
    parser.add_argument(
        "--role",
        choices=("rerank", "retrieve_and_rerank"),
        default="rerank",
        help="Whether the evidence source only reranks or can also discover candidates.",
    )
    parser.add_argument("--score-semantics", required=True)
    parser.add_argument(
        "--score-direction",
        choices=("higher_is_better", "lower_is_better"),
        default="higher_is_better",
    )
    parser.add_argument("--availability-semantics", required=True)
    parser.add_argument("--quality-semantics")
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument(
        "--baseline-id",
        default=None,
        help=(
            "Identifier for the exact frozen core score source used to make "
            "core_score. If omitted, a SHA256 of the sorted query/candidate/core "
            "score relation is used."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    descriptor = EvidenceDescriptor(
        name=str(args.name),
        kind=args.kind,
        role=args.role,
        directions=(args.direction,),
        score_semantics=str(args.score_semantics),
        availability_semantics=str(args.availability_semantics),
        quality_semantics=(
            str(args.quality_semantics) if args.quality_semantics is not None else None
        ),
        provenance=str(args.provenance),
        score_direction=args.score_direction,
    )
    descriptor.validate()

    frame = pd.read_csv(args.csv)
    missing = [name for name in REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"training table missing columns: {missing}")
    frame = frame.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["core_score"] = pd.to_numeric(frame["core_score"], errors="raise").astype(float)
    frame["evidence_score"] = pd.to_numeric(
        frame["evidence_score"], errors="raise"
    ).astype(float)
    if descriptor.score_direction == "lower_is_better":
        frame["evidence_score"] = -frame["evidence_score"]
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain explicit 0/1 experimental outcomes")
    if "available" not in frame.columns:
        frame["available"] = True
    else:
        frame["available"] = parse_bool_series(frame["available"])
    use_quality = args.quality_semantics is not None
    if use_quality:
        if "quality" not in frame.columns:
            raise ValueError("--quality-semantics requires a quality column")
        frame["quality"] = pd.to_numeric(frame["quality"], errors="raise").astype(float)
    elif "quality" in frame.columns:
        raise ValueError("quality column requires --quality-semantics")

    if not np.isfinite(frame["core_score"]).all():
        raise ValueError("core_score must be finite")
    if not np.isfinite(frame.loc[frame["available"], "evidence_score"]).all():
        raise ValueError("available evidence_score values must be finite")
    if use_quality and not np.isfinite(frame.loc[frame["available"], "quality"]).all():
        raise ValueError("available quality values must be finite")
    if use_quality and (
        (frame.loc[frame["available"], "quality"] < 0.0).any()
        or (frame.loc[frame["available"], "quality"] > 1.0).any()
    ):
        raise ValueError("available quality values must lie in [0, 1]")

    if "fold" in frame.columns:
        fold_by_query = (
            frame[["query_id", "fold"]]
            .drop_duplicates()
            .set_index("query_id")["fold"]
            .astype(int)
            .to_dict()
        )
        if frame.groupby("query_id")["fold"].nunique().max() != 1:
            raise ValueError("all candidates of one query must share a fold")
    else:
        fold_by_query = {
            query: stable_fold(query, args.folds)
            for query in frame["query_id"].unique()
        }
        frame["fold"] = frame["query_id"].map(fold_by_query)

    folds = sorted(set(int(value) for value in frame["fold"].unique()))
    if len(folds) < 2:
        raise ValueError("cross-fitting requires at least two populated folds")

    oof: list[dict[str, object]] = []
    oof_query_improvements: list[np.ndarray] = []
    for holdout in folds:
        train = frame[frame["fold"] != holdout]
        test = frame[frame["fold"] == holdout]
        core_train, evidence_train, weight_train, train_queries = stack_queries(
            train,
            use_quality=use_quality,
        )
        fit = fit_nonnegative_pairwise_logistic(
            core_train,
            evidence_train,
            sample_weight=weight_train,
            l2=float(args.l2),
        )
        core_test, evidence_test, weight_test, test_queries = stack_queries(
            test,
            use_quality=use_quality,
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
            use_quality=use_quality,
            coefficients=fit.coefficients,
        )
        oof_query_improvements.append(local_query_improvement)
        oof.append(
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
        use_quality=use_quality,
    )
    final_fit = fit_nonnegative_pairwise_logistic(
        core_all,
        evidence_all,
        sample_weight=weight_all,
        l2=float(args.l2),
    )
    improvements = np.asarray([row["improvement"] for row in oof], dtype=float)
    query_stability = bootstrap_mean_interval(
        np.concatenate(oof_query_improvements),
    )
    admitted = bool(
        np.all(improvements > 0.0)
        and query_stability.lower_95 > 0.0
        and np.any(final_fit.coefficients > 0.0)
    )
    training_table_sha256 = file_sha256(args.csv)
    core_signature = core_score_signature(frame)
    baseline_id = str(
        args.baseline_id
        or f"core-score-sha256:{core_signature}"
    )

    payload = {
        "schema": SINGLE_ADMISSION_SCHEMA,
        "admission_policy": ADMISSION_POLICY,
        "baseline": {
            "id": baseline_id,
            "core_score_sha256": core_signature,
            "training_table_sha256": training_table_sha256,
            "meaning": (
                "Evidence coefficients are valid only for the frozen core score "
                "semantics identified by this baseline."
            ),
        },
        "descriptor": {
            "name": descriptor.name,
            "kind": descriptor.kind,
            "role": descriptor.role,
            "directions": list(descriptor.directions),
            "score_semantics": descriptor.score_semantics,
            "availability_semantics": descriptor.availability_semantics,
            "quality_semantics": descriptor.quality_semantics,
            "provenance": descriptor.provenance,
            "score_direction": descriptor.score_direction,
        },
        "data": {
            "path": str(args.csv),
            "rows": int(len(frame)),
            "queries": int(frame["query_id"].nunique()),
            "usable_queries": int(queries_all),
            "explicit_positive_rows": int((frame["label"] == 1).sum()),
            "explicit_negative_rows": int((frame["label"] == 0).sum()),
        },
        "crossfit": oof,
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
            "strength": float(final_fit.coefficients[0]),
            "quality_slope": (
                float(final_fit.coefficients[1]) if use_quality else 0.0
            ),
            "all_holdouts_improved_pairwise_log_loss": bool(np.all(improvements > 0.0)),
            "query_bootstrap_lower_95_positive": bool(
                query_stability.lower_95 > 0.0
            ),
            "admitted": admitted,
        },
        "note": (
            "Only explicit positive/negative outcomes are used. Unlabelled pairs are "
            "not silently converted into negatives."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
