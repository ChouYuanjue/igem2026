from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_open_world_uncertainty import (  # noqa: E402
    ENSEMBLE_FEATURES,
    fit_calibrator,
    full_reaction_similarity,
    nearest_train_reaction_similarity,
    query_record,
    selective_table,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    ExactResidualReactionDualTower,
)
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig  # noqa: E402

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_MODELS = ROOT / "results/terpene_horizyn_residual_canonical_exact"
DEFAULT_AUX = ROOT / "results/terpene_horizyn_adapter_full/horizyn_reaction_embeddings.npy"
DEFAULT_OUTPUT = ROOT / "results/terpene_exact_residual_uncertainty"
DEFAULT_BUDGETS = (10, 20)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1.0
    return matrix / denominator


def load_split_models(
    model_dir: Path,
    split_id: str,
    device: torch.device,
) -> list[ExactResidualReactionDualTower]:
    paths = sorted((model_dir / "models").glob(f"residual_{split_id}_model*.pt"))
    if not paths:
        raise FileNotFoundError(f"No exact residual models for {split_id} under {model_dir}")
    models: list[ExactResidualReactionDualTower] = []
    for path in paths:
        payload = torch.load(path, map_location=device, weights_only=False)
        model = ExactResidualReactionDualTower(
            ModelConfig(**payload["base_model_config"]),
            int(payload["aux_input_dim"]),
            int(payload["aux_hidden_dim"]),
            float(payload["gate_init"]),
            bool(payload.get("vector_gate", False)),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        models.append(model)
    return models


def evaluate_queries(
    cache_dir: Path,
    model_dir: Path,
    auxiliary_path: Path,
    budgets: tuple[int, ...],
    device: torch.device,
) -> pd.DataFrame:
    protein_features = normalize_rows(np.load(cache_dir / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    auxiliary_features = normalize_rows(np.load(auxiliary_path).astype(np.float32))
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
    if len(auxiliary_features) != len(reaction_ids):
        raise ValueError("Exact auxiliary matrix and reaction table differ in length")
    reaction_similarity = full_reaction_similarity(reaction_table)
    protein_tensor = torch.as_tensor(protein_features, dtype=torch.float32, device=device)
    reaction_tensor = torch.as_tensor(reaction_features, dtype=torch.float32, device=device)
    auxiliary_tensor = torch.as_tensor(auxiliary_features, dtype=torch.float32, device=device)
    records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            train_pairs = pairs[
                pairs["protein_fold"].ne(protein_fold)
                & pairs["reaction_fold"].ne(reaction_fold)
            ]
            test_pairs = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            if test_pairs.empty:
                continue
            train_reaction_ids = sorted(set(train_pairs["rhea_id"].astype(str)))
            train_reaction_rows = np.asarray(
                [reaction_to_row[value] for value in train_reaction_ids], dtype=np.int64
            )
            models = load_split_models(model_dir, split_id, device)
            member_proteins: list[np.ndarray] = []
            member_reactions: list[np.ndarray] = []
            with torch.no_grad():
                for model in models:
                    member_proteins.append(model.encode_proteins(protein_tensor).cpu().numpy())
                    member_reactions.append(
                        model.encode_reactions(reaction_tensor, auxiliary_tensor).cpu().numpy()
                    )

            for reaction_id, group in test_pairs.groupby("rhea_id", sort=True):
                positives = set(group["Entry"].astype(str))
                reaction_row = reaction_to_row[reaction_id]
                nearest = nearest_train_reaction_similarity(
                    reaction_row,
                    train_reaction_rows,
                    reaction_similarity,
                )
                member_scores = np.stack(
                    [
                        member_reactions[index][reaction_row]
                        @ member_proteins[index].T
                        for index in range(len(models))
                    ]
                ).astype(np.float32)
                for budget in budgets:
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
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate exact-residual R2E uncertainty on canonical strict double-cold folds.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--auxiliary", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    features = evaluate_queries(
        args.cache_dir.resolve(),
        args.model_dir.resolve(),
        args.auxiliary.resolve(),
        budgets,
        torch.device(args.device),
    )
    features.to_csv(output_dir / "query_uncertainty_features.csv", index=False)

    calibrators: dict[str, object] = {}
    scored_frames: list[pd.DataFrame] = []
    selective_frames: list[pd.DataFrame] = []
    for index, (budget, group) in enumerate(features.groupby("budget", sort=True)):
        key = f"reaction_to_enzyme_top{int(budget)}"
        calibrator, scored = fit_calibrator(
            group.reset_index(drop=True),
            list(ENSEMBLE_FEATURES),
            bootstrap_seed=20260723 + index,
        )
        calibrator["model_family"] = "horizyn_reaction_residual_exact"
        calibrator["benchmark_manifest"] = str(
            (args.cache_dir.resolve() / "marts_pair_folds.csv")
        )
        calibrators[key] = calibrator
        scored_frames.append(scored)
        if calibrator.get("deployable"):
            selective = selective_table(scored)
            selective.insert(0, "budget", int(budget))
            selective.insert(0, "direction", "reaction_to_enzyme")
            selective_frames.append(selective)

    pd.concat(scored_frames, ignore_index=True).to_csv(
        output_dir / "query_uncertainty_scored.csv", index=False
    )
    selective = (
        pd.concat(selective_frames, ignore_index=True)
        if selective_frames
        else pd.DataFrame()
    )
    selective.to_csv(output_dir / "selective_performance.csv", index=False)
    (output_dir / "calibrators.json").write_text(
        json.dumps(calibrators, indent=2), encoding="utf-8"
    )
    summary_rows = []
    for key, value in calibrators.items():
        summary_rows.append(
            {
                "calibrator": key,
                "deployable": value.get("deployable", False),
                **value.get("cross_validated", {}),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "calibration_summary.csv", index=False)
    print(summary.to_string(index=False))
    if not selective.empty:
        print(selective.to_string(index=False))


if __name__ == "__main__":
    main()
