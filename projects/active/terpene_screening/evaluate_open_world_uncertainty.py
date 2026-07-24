from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_marts_adapted_neighbor_hybrid import (  # noqa: E402
    enzyme_to_reaction_transfer,
)
from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (  # noqa: E402
    reaction_features as zero_shot_reaction_features,
    reaction_similarity as zero_shot_reaction_similarity,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
    E2R_TOP10_RRF_CONSTANT,
    E2R_TOP10_RRF_PRIMARY_WEIGHT,
    E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
    E2R_TOP10_SECONDARY_NEIGHBOR_K,
    choose_retrieval_scores,
    ensemble_query_diagnostics,
    rank_positions,
    reciprocal_rank_fusion_members,
    reciprocal_rank_fusion_scores,
)
from projects.active.terpene_screening.train_dual_tower_cold import (  # noqa: E402
    ModelConfig,
    TerpeneDualTower,
    rank_metrics,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_R2E_SHORT = ROOT / "results/terpene_marts_domain_adaptation_r2e075"
DEFAULT_R2E_LONG = ROOT / "results/terpene_marts_domain_adaptation_cartesian_pu"
DEFAULT_E2R = ROOT / "results/terpene_marts_domain_adaptation_freeze_reaction"
DEFAULT_E2R_SECONDARY = ROOT / "results/terpene_marts_domain_adaptation_hardneg128_e50"
DEFAULT_OUTPUT = ROOT / "results/terpene_open_world_uncertainty_rrf_e2r"
DEFAULT_BUDGETS = (3, 10, 20)
NOVELTY_FEATURES = ["query_nearest_train_similarity"]
ENSEMBLE_FEATURES = [
    "ensemble_top1_vote_fraction",
    "ensemble_top1_rank_std",
    "ensemble_top1_score_std",
    "ensemble_top1_margin_z",
    "ensemble_topk_jaccard",
    "ensemble_topk_vote_mean",
    "ensemble_boundary_margin_z",
]
FEATURE_COLUMNS = NOVELTY_FEATURES + ENSEMBLE_FEATURES
SELECTED_FEATURES = {
    ("enzyme_to_reaction", 3): NOVELTY_FEATURES,
    ("enzyme_to_reaction", 10): FEATURE_COLUMNS,
    ("enzyme_to_reaction", 20): FEATURE_COLUMNS,
    ("reaction_to_enzyme", 20): ENSEMBLE_FEATURES,
}


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def load_models(model_dir: Path, split_id: str, device: torch.device) -> list[TerpeneDualTower]:
    paths = sorted((model_dir / "models").glob(f"adapted_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No adapted models for {split_id} under {model_dir}")
    models = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = TerpeneDualTower(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def encode_models(
    models: list[TerpeneDualTower],
    protein_features: np.ndarray,
    reaction_features: np.ndarray,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    proteins, reactions = [], []
    with torch.no_grad():
        for model in models:
            proteins.append(model.encode_proteins(protein_tensor).cpu().numpy())
            reactions.append(model.encode_reactions(reaction_tensor).cpu().numpy())
    return proteins, reactions


def full_reaction_similarity(reaction_table: pd.DataFrame) -> np.ndarray:
    cached = DEFAULT_CACHE / "reaction_zero_shot_similarity.npy"
    if cached.exists():
        matrix = np.load(cached).astype(np.float32)
        if matrix.shape == (len(reaction_table), len(reaction_table)):
            return matrix
    features = [
        zero_shot_reaction_features(str(value))
        for value in reaction_table["reaction_smiles"].astype(str)
    ]
    matrix = np.eye(len(features), dtype=np.float32)
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            value = float(zero_shot_reaction_similarity(features[left], features[right]))
            matrix[left, right] = value
            matrix[right, left] = value
    np.save(cached, matrix)
    return matrix


def topk_diagnostics(
    member_scores: np.ndarray,
    candidate_ids: list[str],
    budget: int,
    consensus_scores: np.ndarray | None = None,
) -> dict[str, float]:
    base = ensemble_query_diagnostics(
        member_scores,
        candidate_ids,
        set(),
        budget,
        consensus_scores=consensus_scores,
    )
    mean_scores = (
        np.asarray(consensus_scores, dtype=np.float64)
        if consensus_scores is not None
        else member_scores.mean(axis=0).astype(np.float64)
    )
    order = np.lexsort((np.asarray(candidate_ids), -mean_scores))
    effective_k = min(budget, len(candidate_ids))
    selected = order[:effective_k]
    member_ranks = np.stack(
        [rank_positions(scores, candidate_ids, set()) for scores in member_scores]
    )
    vote_mean = float(np.mean(member_ranks[:, selected] <= effective_k)) if len(selected) else 0.0
    score_scale = float(np.std(mean_scores))
    if effective_k < len(order) and score_scale > 0:
        boundary_margin = float(
            (mean_scores[order[effective_k - 1]] - mean_scores[order[effective_k]]) / score_scale
        )
    else:
        boundary_margin = 0.0
    return {
        **base,
        "ensemble_topk_vote_mean": vote_mean,
        "ensemble_boundary_margin_z": boundary_margin,
    }


def nearest_train_protein_similarity(
    query_row: int,
    train_rows: np.ndarray,
    protein_features: np.ndarray,
) -> float:
    if not len(train_rows):
        return float("nan")
    return float(np.max(protein_features[train_rows] @ protein_features[query_row]))


def nearest_train_reaction_similarity(
    query_row: int,
    train_rows: np.ndarray,
    similarity: np.ndarray,
) -> float:
    if not len(train_rows):
        return float("nan")
    return float(np.max(similarity[query_row, train_rows]))


def query_record(
    *,
    split_id: str,
    direction: str,
    budget: int,
    query_id: str,
    positives: set[str],
    candidate_ids: list[str],
    member_scores: np.ndarray,
    nearest_similarity: float,
    consensus_scores: np.ndarray | None = None,
) -> dict[str, object]:
    ensemble = (
        np.asarray(consensus_scores, dtype=np.float32)
        if consensus_scores is not None
        else member_scores.mean(axis=0)
    )
    metrics = rank_metrics(ensemble, candidate_ids, positives, set(), (budget,))
    return {
        "split_id": split_id,
        "direction": direction,
        "budget": budget,
        "query_id": query_id,
        "n_positives": len(positives),
        "hit": int(metrics[f"hit_at_{budget}"]),
        "positive_recall": float(metrics[f"positive_recall_at_{budget}"]),
        "best_positive_rank": metrics["best_positive_rank"],
        "reciprocal_rank": metrics["reciprocal_rank"],
        "query_nearest_train_similarity": nearest_similarity,
        **topk_diagnostics(
            member_scores, candidate_ids, budget, consensus_scores=consensus_scores
        ),
    }


def evaluate_queries(
    cache_dir: Path,
    r2e_short_dir: Path,
    r2e_long_dir: Path,
    e2r_dir: Path,
    e2r_secondary_dir: Path,
    budgets: tuple[int, ...],
    device: torch.device,
) -> pd.DataFrame:
    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].astype(str).str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].astype(str).str.lower().eq("true")
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    reaction_similarity = full_reaction_similarity(reaction_table)
    records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train_pairs = pairs[
                (pairs["protein_fold"] != protein_fold)
                & (pairs["reaction_fold"] != reaction_fold)
            ].copy()
            test_pairs = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ].copy()
            if test_pairs.empty:
                continue
            train_protein_ids = sorted(set(train_pairs["Entry"].astype(str)))
            train_reaction_ids = sorted(set(train_pairs["rhea_id"].astype(str)))
            train_protein_rows = np.asarray(
                [protein_to_row[value] for value in train_protein_ids], dtype=np.int64
            )
            train_reaction_rows = np.asarray(
                [reaction_to_row[value] for value in train_reaction_ids], dtype=np.int64
            )
            train_by_protein = {
                protein_id: sorted(set(group["rhea_id"].astype(str)))
                for protein_id, group in train_pairs.groupby("Entry")
            }

            model_cache: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
            for label, directory in {
                "r2e_short": r2e_short_dir,
                "r2e_long": r2e_long_dir,
                "e2r": e2r_dir,
                "e2r_secondary": e2r_secondary_dir,
            }.items():
                models = load_models(directory, split_id, device)
                model_cache[label] = encode_models(
                    models, protein_features, reaction_features, device
                )

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                reaction_row = reaction_to_row[reaction_id]
                nearest = nearest_train_reaction_similarity(
                    reaction_row, train_reaction_rows, reaction_similarity
                )
                for budget in budgets:
                    label = "r2e_short" if budget <= 10 else "r2e_long"
                    protein_embeddings, reaction_embeddings = model_cache[label]
                    member_scores = np.stack(
                        [
                            reaction_embeddings[index][reaction_row]
                            @ protein_embeddings[index].T
                            for index in range(len(protein_embeddings))
                        ]
                    ).astype(np.float32)
                    records.append(
                        query_record(
                            split_id=split_id,
                            direction="reaction_to_enzyme",
                            budget=budget,
                            query_id=reaction_id,
                            positives=positives,
                            candidate_ids=protein_ids,
                            member_scores=member_scores,
                            nearest_similarity=nearest,
                        )
                    )

            protein_embeddings, reaction_embeddings = model_cache["e2r"]
            secondary_proteins, secondary_reactions = model_cache["e2r_secondary"]
            if len(protein_embeddings) != len(secondary_proteins):
                raise ValueError("Primary and secondary E2R fold ensembles differ in size")
            for protein_id, group in test_pairs.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                protein_row = protein_to_row[protein_id]
                nearest = nearest_train_protein_similarity(
                    protein_row, train_protein_rows, protein_features
                )
                transfer = enzyme_to_reaction_transfer(
                    protein_id,
                    train_by_protein,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    reaction_embeddings,
                    topk_neighbors=5,
                )
                secondary_transfer = enzyme_to_reaction_transfer(
                    protein_id,
                    train_by_protein,
                    protein_features,
                    protein_to_row,
                    reaction_to_row,
                    secondary_reactions,
                    topk_neighbors=E2R_TOP10_SECONDARY_NEIGHBOR_K,
                )
                direct_members = np.stack(
                    [
                        protein_embeddings[index][protein_row]
                        @ reaction_embeddings[index].T
                        for index in range(len(protein_embeddings))
                    ]
                ).astype(np.float32)
                secondary_direct_members = np.stack(
                    [
                        secondary_proteins[index][protein_row]
                        @ secondary_reactions[index].T
                        for index in range(len(secondary_proteins))
                    ]
                ).astype(np.float32)
                for budget in budgets:
                    direct_weight = {
                        3: 0.75,
                        10: E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
                        20: 0.75,
                    }[budget]
                    routed = []
                    for direct in direct_members:
                        scores, _ = choose_retrieval_scores(
                            direct,
                            None,
                            reaction_ids,
                            "neighbor_hybrid",
                            neighbor_scores=transfer,
                            hybrid_direct_weight=direct_weight,
                        )
                        routed.append(scores)
                    routed_members = np.stack(routed).astype(np.float32)
                    consensus_scores: np.ndarray | None = None
                    if budget == 10:
                        secondary_routed = []
                        for direct in secondary_direct_members:
                            scores, _ = choose_retrieval_scores(
                                direct,
                                None,
                                reaction_ids,
                                "neighbor_hybrid",
                                neighbor_scores=secondary_transfer,
                                hybrid_direct_weight=E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
                            )
                            secondary_routed.append(scores)
                        secondary_routed_members = np.stack(secondary_routed).astype(np.float32)
                        primary_consensus, _ = choose_retrieval_scores(
                            direct_members.mean(axis=0),
                            None,
                            reaction_ids,
                            "neighbor_hybrid",
                            neighbor_scores=transfer,
                            hybrid_direct_weight=E2R_TOP10_PRIMARY_DIRECT_WEIGHT,
                        )
                        secondary_consensus, _ = choose_retrieval_scores(
                            secondary_direct_members.mean(axis=0),
                            None,
                            reaction_ids,
                            "neighbor_hybrid",
                            neighbor_scores=secondary_transfer,
                            hybrid_direct_weight=E2R_TOP10_SECONDARY_DIRECT_WEIGHT,
                        )
                        consensus_scores = reciprocal_rank_fusion_scores(
                            primary_consensus,
                            secondary_consensus,
                            reaction_ids,
                            E2R_TOP10_RRF_PRIMARY_WEIGHT,
                            E2R_TOP10_RRF_CONSTANT,
                        )
                        routed_members = reciprocal_rank_fusion_members(
                            routed_members,
                            secondary_routed_members,
                            reaction_ids,
                            E2R_TOP10_RRF_PRIMARY_WEIGHT,
                            E2R_TOP10_RRF_CONSTANT,
                        )
                    records.append(
                        query_record(
                            split_id=split_id,
                            direction="enzyme_to_reaction",
                            budget=budget,
                            query_id=protein_id,
                            positives=positives,
                            candidate_ids=reaction_ids,
                            member_scores=routed_members,
                            nearest_similarity=nearest,
                            consensus_scores=consensus_scores,
                        )
                    )
    return pd.DataFrame(records)


def fit_calibrator(
    frame: pd.DataFrame,
    feature_columns: list[str],
    bootstrap_seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    x = frame[feature_columns].astype(float)
    y = frame["hit"].astype(int).to_numpy()
    groups = frame["query_id"].astype(str).to_numpy()
    if len(np.unique(y)) < 2:
        return {"deployable": False, "reason": "single target class"}, frame.assign(reliability_score=np.nan)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    n_splits = min(5, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    probabilities = cross_val_predict(
        pipeline,
        x,
        y,
        groups=groups,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    pipeline.fit(x, y)
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    roc_auc = float(roc_auc_score(y, probabilities))
    average_precision = float(average_precision_score(y, probabilities))
    brier = float(brier_score_loss(y, probabilities))
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_auc = []
    for _ in range(2000):
        indices = rng.integers(0, len(y), len(y))
        if len(np.unique(y[indices])) < 2:
            continue
        bootstrap_auc.append(float(roc_auc_score(y[indices], probabilities[indices])))
    auc_ci_low, auc_ci_high = np.quantile(bootstrap_auc, [0.025, 0.975])
    thresholds = {
        "low": float(np.quantile(probabilities, 0.25)),
        "high": float(np.quantile(probabilities, 0.75)),
    }
    scored = frame.copy()
    scored["reliability_score"] = probabilities
    scored["risk_tier"] = np.where(
        probabilities >= thresholds["high"],
        "higher_evidence",
        np.where(probabilities < thresholds["low"], "lower_evidence", "intermediate"),
    )
    deployment = {
        "deployable": bool(auc_ci_low > 0.5),
        "feature_columns": feature_columns,
        "imputer_statistics": imputer.statistics_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "coefficient": classifier.coef_[0].astype(float).tolist(),
        "intercept": float(classifier.intercept_[0]),
        "thresholds": thresholds,
        "cross_validated": {
            "roc_auc": roc_auc,
            "roc_auc_ci_low": float(auc_ci_low),
            "roc_auc_ci_high": float(auc_ci_high),
            "average_precision": average_precision,
            "base_hit_rate": float(np.mean(y)),
            "brier_score": brier,
        },
    }
    return deployment, scored


def selective_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(np.ceil(len(scored) * coverage)))
        selected = scored.sort_values("reliability_score", ascending=False).head(count)
        rows.append(
            {
                "coverage": coverage,
                "n_queries": len(selected),
                "hit_rate": float(selected["hit"].mean()),
                "mean_positive_recall": float(selected["positive_recall"].mean()),
                "median_best_positive_rank": float(selected["best_positive_rank"].median()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate open-world retrieval uncertainty on strict double-cold folds.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--r2e-short-dir", type=Path, default=DEFAULT_R2E_SHORT)
    parser.add_argument("--r2e-long-dir", type=Path, default=DEFAULT_R2E_LONG)
    parser.add_argument("--e2r-dir", type=Path, default=DEFAULT_E2R)
    parser.add_argument(
        "--e2r-secondary-dir", type=Path, default=DEFAULT_E2R_SECONDARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    query_features = evaluate_queries(
        args.cache_dir.resolve(),
        args.r2e_short_dir.resolve(),
        args.r2e_long_dir.resolve(),
        args.e2r_dir.resolve(),
        args.e2r_secondary_dir.resolve(),
        budgets,
        torch.device(args.device),
    )
    query_features.to_csv(output_dir / "query_uncertainty_features.csv", index=False)

    calibrators: dict[str, object] = {}
    scored_frames = []
    selective_frames = []
    for calibration_index, ((direction, budget), group) in enumerate(
        query_features.groupby(["direction", "budget"], sort=True)
    ):
        key = f"{direction}_top{budget}"
        feature_columns = list(SELECTED_FEATURES.get((direction, int(budget)), FEATURE_COLUMNS))
        calibrator, scored = fit_calibrator(
            group.reset_index(drop=True),
            feature_columns,
            bootstrap_seed=20260723 + calibration_index,
        )
        calibrators[key] = calibrator
        scored_frames.append(scored)
        selective = selective_table(scored) if calibrator.get("deployable") else pd.DataFrame()
        if not selective.empty:
            selective.insert(0, "budget", budget)
            selective.insert(0, "direction", direction)
            selective_frames.append(selective)

    scored_all = pd.concat(scored_frames, ignore_index=True)
    scored_all.to_csv(output_dir / "query_uncertainty_scored.csv", index=False)
    selective_all = pd.concat(selective_frames, ignore_index=True) if selective_frames else pd.DataFrame()
    selective_all.to_csv(output_dir / "selective_performance.csv", index=False)
    calibration_path = output_dir / "calibrators.json"
    calibration_path.write_text(json.dumps(calibrators, indent=2), encoding="utf-8")

    summary_rows = []
    for key, value in calibrators.items():
        metrics = value.get("cross_validated", {})
        summary_rows.append(
            {
                "calibrator": key,
                "deployable": value.get("deployable", False),
                **metrics,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    print(summary.to_string(index=False))
    if not selective_all.empty:
        print(selective_all.to_string(index=False))
    print(json.dumps({"calibrators": str(calibration_path), "n_query_rows": len(query_features)}, indent=2))


if __name__ == "__main__":
    main()
