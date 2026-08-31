from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "projects/active/terpene_screening/train_cleanroom_rhea_retriever.py"
EVAL = ROOT / "projects/active/terpene_screening/evaluate_broad_rhea_benchmark.py"
FREEZE = ROOT / "projects/active/terpene_screening/CLEANROOM_ENZGFM_RDKITPLUS_OUTER_V1.json"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
PROTEIN = ROOT / "data/external/enzgfm_current/general_merged_esmc_enzgfm_equalblock_v1"
REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
GATE_A = ROOT / "results/enzgfm_gate_a_bidirectional_v1/selection"
GATE_B = ROOT / "results/enzgfm_gate_b_rdkitplus_v1/selection"
DEFAULT_MODELS = ROOT / "results/enzgfm_rdkitplus_outer_models_v1"
DEFAULT_EVAL = ROOT / "results/enzgfm_rdkitplus_outer_eval_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def validate_freeze(freeze: dict[str, object]) -> None:
    development = dict(freeze["development_selection"])
    expected = {
        GATE_A / "summary.json": development["gate_a_selection_summary_sha256"],
        GATE_A / "candidate_summary.csv": development["gate_a_candidate_summary_sha256"],
        GATE_B / "summary.json": development["gate_b_selection_summary_sha256"],
        GATE_B / "candidate_summary.csv": development["gate_b_candidate_summary_sha256"],
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Frozen Gate evidence changed or is missing: {path}")
    gate_a = json.loads((GATE_A / "summary.json").read_text(encoding="utf-8"))
    gate_b = json.loads((GATE_B / "summary.json").read_text(encoding="utf-8"))
    if gate_a.get("selected_candidate") != "esmc_enzgfm_equalblock":
        raise RuntimeError("Gate A selected candidate differs from frozen outer protocol")
    if gate_b.get("selected_candidate") != "rdkitplus":
        raise RuntimeError("Gate B selected candidate differs from frozen outer protocol")
    for path in [PROTEIN / "entries.csv", PROTEIN / "embeddings.npy", PROTEIN / "manifest.json", REACTION / "manifest.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)


def model_complete(path: Path) -> bool:
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("dev_fold") == -1 and int(payload.get("n_train_pairs") or 0) > 0


def eval_complete(root: Path, cell: str) -> bool:
    summary = root / cell / "summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return payload.get("cell") == cell and "reaction_to_enzyme" in (payload.get("metrics") or {}) and "enzyme_to_reaction" in (payload.get("metrics") or {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered EnzGFM+RDKit+ outer matrix with per-cell train-only retraining followed by full-candidate evaluation.")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell", action="append", default=None, help="Optional subset; must belong to the frozen cell list.")
    args = parser.parse_args()

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    validate_freeze(freeze)
    frozen_cells = list(freeze["outer_cells"])
    cells = list(args.cell) if args.cell else frozen_cells
    unknown = sorted(set(cells) - set(frozen_cells))
    if unknown:
        raise ValueError(f"Cells are not in the frozen outer protocol: {unknown}")
    training = dict(freeze["outer_training"])
    models_root = args.models_root.resolve(); eval_root = args.eval_root.resolve()
    models_root.mkdir(parents=True, exist_ok=True); eval_root.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        cell_dir = BENCH / cell
        train_pairs = cell_dir / "train_pairs.csv"
        manifest = cell_dir / "manifest.json"
        if not train_pairs.is_file() or not manifest.is_file():
            raise FileNotFoundError(cell_dir)
        model_dir = models_root / cell
        if not model_complete(model_dir):
            command = [
                args.python, str(TRAIN),
                "--associations-csv", str(train_pairs),
                "--schema-dir", str(REACTION),
                "--reaction-feature-dir", str(REACTION),
                "--protein-feature-dir", str(PROTEIN),
                "--output-dir", str(model_dir),
                "--dev-fold", "-1", "--folds", "5",
                "--epochs", str(training["epochs"]),
                "--steps-per-epoch", str(training["steps_per_epoch"]),
                "--reaction-batch-size", str(training["reaction_batch_size"]),
                "--protein-batch-size", str(training["protein_batch_size"]),
                "--neighbor-k", str(training["neighbor_k"]),
                "--dev-neighbor-reactions", "10",
                "--hard-negatives", str(training["hard_negatives"]),
                "--random-negatives", str(training["random_negatives"]),
                "--hard-negative-ramp-epochs", str(training["hard_negative_ramp_epochs"]),
                "--hidden-dim", str(training["hidden_dim"]),
                "--embedding-dim", str(training["embedding_dim"]),
                "--dropout", str(training["dropout"]),
                "--learning-rate", str(training["learning_rate"]),
                "--weight-decay", str(training["weight_decay"]),
                "--temperature", str(training["temperature"]),
                "--topk", str(training["topk"]),
                "--topk-weight", str(training["topk_weight"]),
                "--margin", str(training["margin"]),
                "--r2e-weight", str(training["r2e_weight"]),
                "--reaction-novelty-repeat", str(training["reaction_novelty_repeat"]),
                "--seed", str(training["seed"]),
                "--device", args.device,
            ]
            run(command)
        else:
            print(f"SKIP trained {cell}", flush=True)

        if not eval_complete(eval_root, cell):
            run([
                args.python, str(EVAL),
                "--cell", cell,
                "--benchmark-root", str(BENCH),
                "--protein-feature-dir", str(PROTEIN),
                "--r2e-model-dir", str(model_dir),
                "--e2r-model-dir", str(model_dir),
                "--r2e-reaction-feature-dir", str(REACTION),
                "--e2r-reaction-feature-dir", str(REACTION),
                "--output-dir", str(eval_root),
                "--device", args.device,
            ])
        else:
            print(f"SKIP evaluated {cell}", flush=True)

    metadata = {
        "freeze": str(FREEZE.resolve()),
        "freeze_sha256": sha256(FREEZE),
        "cells": cells,
        "protein_feature_dir": str(PROTEIN.resolve()),
        "reaction_feature_dir": str(REACTION.resolve()),
        "models_root": str(models_root),
        "eval_root": str(eval_root),
    }
    (eval_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
