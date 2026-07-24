from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HORIZYN_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZYN_ROOT))

from horizyn.lightning_module import HorizynLitModule  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402
from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (  # noqa: E402
    build_horizyn_fingerprints,
    encode_horizyn_reactions,
    normalize_rows,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_MAPPING = ROOT / "data/terpene_horizyn_exact_sequence_mapping.csv"
DEFAULT_H5 = HORIZYN_ROOT / "data/sota/prots_t5.h5"
DEFAULT_CONFIG = HORIZYN_ROOT / "configs/sota.yaml"
DEFAULT_DEV_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_dev.ckpt"
DEFAULT_INF_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_inf.ckpt"
DEFAULT_OUTPUT = ROOT / "results/terpene_horizyn_prott5_bridge_double_cold"
DEFAULT_FINGERPRINT_CACHE = ROOT / "data/terpene_horizyn_adapter_v2"
DEFAULT_BUDGETS = (3, 10, 20)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_rows(left)
    right = normalize_rows(right)
    return np.sum(left * right, axis=1)


def load_prott5_vectors(
    h5_path: Path,
    mapping: pd.DataFrame,
    protein_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    with h5py.File(h5_path, "r") as handle:
        ids = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["ids"][:]]
        id_to_row = {value: index for index, value in enumerate(ids)}
        vectors = handle["vectors"]
        matrix = np.full((len(protein_ids), vectors.shape[1]), np.nan, dtype=np.float32)
        audit_rows: list[dict[str, object]] = []
        for row in mapping.itertuples(index=False):
            protein_id = str(row.protein_id)
            horizyn_id = str(row.horizyn_id)
            if protein_id not in protein_to_row or horizyn_id not in id_to_row:
                continue
            protein_row = protein_to_row[protein_id]
            matrix[protein_row] = vectors[id_to_row[horizyn_id]]
            audit_rows.append(
                {
                    "protein_id": protein_id,
                    "horizyn_id": horizyn_id,
                    "protein_row": protein_row,
                    "horizyn_row": id_to_row[horizyn_id],
                    "enzyme_seen": getattr(row, "enzyme_seen", ""),
                }
            )
    matched = np.isfinite(matrix).all(axis=1)
    return matrix, matched, pd.DataFrame(audit_rows)


def choose_ridge_alpha(
    protein_features: np.ndarray,
    prott5_vectors: np.ndarray,
    matched: np.ndarray,
    protein_folds: np.ndarray,
    alphas: tuple[float, ...],
    development_fold: int,
) -> tuple[float, pd.DataFrame]:
    train = matched & (protein_folds != development_fold)
    validation = matched & (protein_folds == development_fold)
    if train.sum() == 0 or validation.sum() == 0:
        raise ValueError("Bridge alpha selection requires matched proteins in train and development folds")
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, fit_intercept=True),
        )
        model.fit(protein_features[train], prott5_vectors[train])
        prediction = model.predict(protein_features[validation]).astype(np.float32)
        cosine = cosine_rows(prediction, prott5_vectors[validation])
        rows.append(
            {
                "alpha": alpha,
                "train_proteins": int(train.sum()),
                "development_proteins": int(validation.sum()),
                "mean_cosine": float(cosine.mean()),
                "median_cosine": float(np.median(cosine)),
                "mean_squared_error": float(
                    np.mean((prediction - prott5_vectors[validation]) ** 2)
                ),
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["mean_cosine", "mean_squared_error", "alpha"],
        ascending=[False, True, True],
    )
    return float(results.iloc[0]["alpha"]), results


def fit_bridge(
    protein_features: np.ndarray,
    prott5_vectors: np.ndarray,
    matched: np.ndarray,
    alpha: float,
) -> tuple[object, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, fit_intercept=True),
    )
    model.fit(protein_features[matched], prott5_vectors[matched])
    prediction = model.predict(protein_features).astype(np.float32)
    return model, prediction


def encode_horizyn_proteins(
    input_vectors: np.ndarray,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    module = HorizynLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    encoder = module.model.target_encoder.to(device).eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(input_vectors), batch_size):
            tensor = torch.as_tensor(
                input_vectors[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(encoder(tensor).float().cpu().numpy())
    return normalize_rows(np.concatenate(outputs, axis=0))


def evaluate_method(
    records: list[dict[str, object]],
    method: str,
    scores: np.ndarray,
    pairs: pd.DataFrame,
    protein_ids: list[str],
    reaction_ids: list[str],
    budgets: tuple[int, ...],
) -> None:
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & pairs["protein_seen"].str.lower().eq("false")
                & pairs["reaction_seen"].str.lower().eq("false")
            ]
            for reaction_id, group in test.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                metrics = rank_metrics(
                    scores[reaction_to_row[reaction_id]],
                    protein_ids,
                    positives,
                    set(),
                    budgets,
                )
                records.append(
                    {
                        "split_id": split_id,
                        "method": method,
                        "direction": "reaction_to_enzyme",
                        "query_id": reaction_id,
                        **metrics,
                    }
                )
            for protein_id, group in test.groupby("Entry", sort=True):
                positives = set(group["rhea_id"].astype(str))
                metrics = rank_metrics(
                    scores[:, protein_to_row[protein_id]],
                    reaction_ids,
                    positives,
                    set(),
                    budgets,
                )
                records.append(
                    {
                        "split_id": split_id,
                        "method": method,
                        "direction": "enzyme_to_reaction",
                        "query_id": protein_id,
                        **metrics,
                    }
                )


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "n_queries": ("query_id", "size"),
        "mean_reciprocal_rank": ("reciprocal_rank", "mean"),
        "median_best_positive_rank": ("best_positive_rank", "median"),
    }
    for budget in budgets:
        aggregations[f"hit_probability_at_{budget}"] = (f"hit_at_{budget}", "mean")
        aggregations[f"positive_recall_at_{budget}"] = (
            f"positive_recall_at_{budget}",
            "mean",
        )
    return frame.groupby(["method", "direction"]).agg(**aggregations).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ESM-C to ProtT5 bridge with official Horizyn encoders."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dev-checkpoint", type=Path, default=DEFAULT_DEV_CHECKPOINT)
    parser.add_argument("--inf-checkpoint", type=Path, default=DEFAULT_INF_CHECKPOINT)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_FINGERPRINT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--development-fold", type=int, default=4)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    alphas = tuple(float(value) for value in args.alphas.split(",") if value)
    device = torch.device(args.device)

    protein_features = np.load(cache_dir / "protein_features.npy").astype(np.float32)
    protein_table = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reaction_table = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    protein_ids = protein_table["protein_id"].astype(str).tolist()
    reaction_ids = reaction_table["reaction_id"].astype(str).tolist()
    mapping = pd.read_csv(args.mapping, dtype=str).fillna("")
    prott5, matched, mapping_audit = load_prott5_vectors(
        args.h5.resolve(), mapping, protein_ids
    )
    fold_map = pairs.groupby("Entry")["protein_fold"].first().to_dict()
    protein_folds = np.asarray(
        [int(fold_map.get(protein_id, args.development_fold)) for protein_id in protein_ids]
    )
    best_alpha, alpha_results = choose_ridge_alpha(
        protein_features,
        prott5,
        matched,
        protein_folds,
        alphas,
        args.development_fold,
    )
    _, bridge_prediction = fit_bridge(
        protein_features, prott5, matched, best_alpha
    )
    direct_or_bridge = bridge_prediction.copy()
    direct_or_bridge[matched] = prott5[matched]
    mapping_audit.to_csv(output_dir / "protein_mapping_audit.csv", index=False)
    alpha_results.to_csv(output_dir / "bridge_alpha_selection.csv", index=False)
    np.save(output_dir / "bridge_predicted_prott5.npy", bridge_prediction)
    np.save(output_dir / "direct_or_bridge_prott5.npy", direct_or_bridge)

    fingerprints, fingerprint_audit = build_horizyn_fingerprints(
        reaction_table,
        args.config.resolve(),
        args.fingerprint_cache.resolve(),
    )
    fingerprint_audit.to_csv(output_dir / "reaction_fingerprint_audit.csv", index=False)

    records: list[dict[str, object]] = []
    checkpoint_specs = {
        "dev": args.dev_checkpoint.resolve(),
        "inf": args.inf_checkpoint.resolve(),
    }
    protein_inputs = {
        "bridge_all": bridge_prediction,
        "direct_or_bridge": direct_or_bridge,
    }
    for checkpoint_label, checkpoint in checkpoint_specs.items():
        reaction_embeddings = encode_horizyn_reactions(
            fingerprints, checkpoint, device, args.batch_size
        )
        np.save(
            output_dir / f"{checkpoint_label}_reaction_embeddings.npy",
            reaction_embeddings,
        )
        for protein_label, protein_input in protein_inputs.items():
            protein_embeddings = encode_horizyn_proteins(
                protein_input, checkpoint, device, args.batch_size
            )
            np.save(
                output_dir / f"{checkpoint_label}_{protein_label}_protein_embeddings.npy",
                protein_embeddings,
            )
            scores = reaction_embeddings @ protein_embeddings.T
            evaluate_method(
                records,
                f"horizyn_{checkpoint_label}_{protein_label}",
                scores,
                pairs,
                protein_ids,
                reaction_ids,
                budgets,
            )

    query_metrics = pd.DataFrame(records)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics = aggregate(query_metrics, budgets)
    metrics.to_csv(output_dir / "metrics.csv", index=False)

    train_pairs = pd.read_csv(
        HORIZYN_ROOT / "data/sota/train_pairs.csv", dtype=str
    ).fillna("")
    test_pairs = pd.read_csv(
        HORIZYN_ROOT / "data/sota/test_pairs.csv", dtype=str
    ).fillna("")
    mapped_ids = set(mapping_audit["horizyn_id"].astype(str))
    overlap = {
        "mapped_in_horizyn_train_pairs": len(mapped_ids & set(train_pairs["protein_id"])),
        "mapped_in_horizyn_test_pairs": len(mapped_ids & set(test_pairs["protein_id"])),
    }
    summary = {
        "n_proteins": len(protein_ids),
        "n_exact_sequence_prott5": int(matched.sum()),
        "n_bridged_prott5": int((~matched).sum()),
        "selected_ridge_alpha": best_alpha,
        "development_fold": args.development_fold,
        "bridge_selection": "Ridge alpha selected by ProtT5 reconstruction cosine on protein fold 4; no reaction labels used.",
        "horizyn_checkpoint_warning": (
            "Horizyn is externally pretrained and may contain overlapping proteins or reactions. "
            "Results are external-pretraining transfer, not a clean from-scratch comparison."
        ),
        "protein_overlap_audit": overlap,
        "fingerprint_success": int(
            fingerprint_audit["success"].astype(str).str.lower().eq("true").sum()
        ),
        "budgets": budgets,
        "device": str(device),
        "outputs": {
            "metrics": str(output_dir / "metrics.csv"),
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "bridge_alpha_selection": str(output_dir / "bridge_alpha_selection.csv"),
            "protein_mapping_audit": str(output_dir / "protein_mapping_audit.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(alpha_results.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
