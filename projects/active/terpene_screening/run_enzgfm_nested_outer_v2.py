from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "projects/active/terpene_screening/train_cleanroom_rhea_retriever.py"
SELECT = ROOT / "projects/active/terpene_screening/select_cleanroom_bidirectional_multifold.py"
EVAL = ROOT / "projects/active/terpene_screening/evaluate_broad_rhea_benchmark.py"
FREEZE = ROOT / "projects/active/terpene_screening/CLEANROOM_ENZGFM_NESTED_OUTER_V2.json"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
BASE_SCHEMA = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
BASE_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1"
RDKIT_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
PROTEINS = {
    "esmc": ROOT / "data/catalyst_candidate_universes/general_merged/proteins",
    "enzgfm": ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1",
    "esmc_enzgfm_equalblock": ROOT / "data/external/enzgfm_current/general_merged_esmc_enzgfm_equalblock_v1",
}
DEFAULT_SELECTION = ROOT / "results/enzgfm_nested_outer_v2_selection"
DEFAULT_MODELS = ROOT / "results/enzgfm_nested_outer_v2_models"
DEFAULT_EVAL = ROOT / "results/enzgfm_nested_outer_v2_eval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def complete_fold(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    dev = payload.get("dev_metrics") or {}
    return "common_ir_r2e" in dev and "common_ir_e2r" in dev


def selected(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get("selected_candidate") or "")
    if not value:
        raise RuntimeError(f"No selected candidate in {path}")
    return value


def train_command(
    *, python: str, associations: Path, protein: Path, reaction: Path, schema: Path,
    output: Path, dev_fold: int, recipe: dict[str, object], device: str,
) -> list[str]:
    return [
        python, str(TRAIN),
        "--associations-csv", str(associations),
        "--schema-dir", str(schema),
        "--reaction-feature-dir", str(reaction),
        "--protein-feature-dir", str(protein),
        "--output-dir", str(output),
        "--dev-fold", str(dev_fold), "--folds", "5",
        "--epochs", str(recipe["epochs"]),
        "--steps-per-epoch", str(recipe["steps_per_epoch"]),
        "--reaction-batch-size", str(recipe["reaction_batch_size"]),
        "--protein-batch-size", str(recipe["protein_batch_size"]),
        "--neighbor-k", str(recipe["neighbor_k"]),
        "--dev-neighbor-reactions", str(recipe["dev_neighbor_reactions"]),
        "--hard-negatives", str(recipe["hard_negatives"]),
        "--random-negatives", str(recipe["random_negatives"]),
        "--hard-negative-ramp-epochs", str(recipe["hard_negative_ramp_epochs"]),
        "--hidden-dim", str(recipe["hidden_dim"]),
        "--embedding-dim", str(recipe["embedding_dim"]),
        "--dropout", str(recipe["dropout"]),
        "--learning-rate", str(recipe["learning_rate"]),
        "--weight-decay", str(recipe["weight_decay"]),
        "--temperature", str(recipe["temperature"]),
        "--topk", str(recipe["topk"]),
        "--topk-weight", str(recipe["topk_weight"]),
        "--margin", str(recipe["margin"]),
        "--r2e-weight", str(recipe["r2e_weight"]),
        "--reaction-novelty-repeat", str(recipe["reaction_novelty_repeat"]),
        "--seed", str(recipe["seed"]),
        "--device", device,
    ]


def run_selector(runs: list[tuple[str, int, Path]], output: Path, python: str) -> None:
    summary = output / "summary.json"
    if summary.is_file():
        return
    command = [python, str(SELECT), "--output-dir", str(output)]
    for candidate, fold, path in runs:
        command += ["--run", f"{candidate}:{fold}:{path}"]
    run(command)


def final_complete(path: Path, train_pairs: Path, protein: Path, reaction: Path) -> bool:
    summary = path / "summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (
        payload.get("dev_fold") == -1
        and Path(str(payload.get("association_source"))).resolve() == train_pairs.resolve()
        and Path(str(payload.get("protein_feature_dir"))).resolve() == protein.resolve()
        and Path(str(payload.get("reaction_feature_dir"))).resolve() == reaction.resolve()
    )


def eval_complete(root: Path, cell: str, protein: Path, model: Path) -> bool:
    summary = root / cell / "summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (
        payload.get("cell") == cell
        and Path(str(payload.get("protein_feature_dir"))).resolve() == protein.resolve()
        and Path(str(payload.get("r2e_model_dir"))).resolve() == model.resolve()
        and "reaction_to_enzyme" in (payload.get("metrics") or {})
        and "enzyme_to_reaction" in (payload.get("metrics") or {})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested train-only selection and final full-candidate outer evaluation for the frozen EnzGFM representation family.")
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cell", action="append", default=None, help="Optional frozen-cell subset; selection/evaluation recipe is unchanged.")
    args = parser.parse_args()

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    cells = list(args.cell) if args.cell else list(freeze["outer_cells"])
    unknown = sorted(set(cells) - set(freeze["outer_cells"]))
    if unknown:
        raise ValueError(f"Not in frozen v2 outer cells: {unknown}")
    if list(freeze["protein_candidates"]) != list(PROTEINS):
        raise RuntimeError("Frozen protein candidate list differs from runner")
    recipe = dict(freeze["training_recipe"])
    folds = [int(value) for value in freeze["inner_selection"]["folds"]]
    for path in [FREEZE, BASE_REACTION / "manifest.json", RDKIT_REACTION / "manifest.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for directory in PROTEINS.values():
        for name in ["entries.csv", "embeddings.npy"]:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)

    selection_root = args.selection_root.resolve(); models_root = args.models_root.resolve(); eval_root = args.eval_root.resolve()
    selection_root.mkdir(parents=True, exist_ok=True); models_root.mkdir(parents=True, exist_ok=True); eval_root.mkdir(parents=True, exist_ok=True)

    for cell in cells:
        train_pairs = BENCH / cell / "train_pairs.csv"
        if not train_pairs.is_file():
            raise FileNotFoundError(train_pairs)
        cell_selection = selection_root / cell
        gate_a = cell_selection / "gate_a"
        gate_a_runs: list[tuple[str, int, Path]] = []
        for candidate, protein in PROTEINS.items():
            for fold in folds:
                target = gate_a / candidate / f"fold{fold}"
                summary = target / "summary.json"
                if not complete_fold(summary):
                    run(train_command(
                        python=args.python, associations=train_pairs, protein=protein,
                        reaction=BASE_REACTION, schema=BASE_SCHEMA, output=target,
                        dev_fold=fold, recipe=recipe, device=args.device,
                    ))
                gate_a_runs.append((candidate, fold, summary))
        run_selector(gate_a_runs, gate_a / "selection", args.python)
        protein_name = selected(gate_a / "selection" / "summary.json")
        protein = PROTEINS[protein_name]

        gate_b = cell_selection / "gate_b"
        gate_b_runs: list[tuple[str, int, Path]] = []
        for fold in folds:
            gate_b_runs.append(("base_reaction", fold, gate_a / protein_name / f"fold{fold}" / "summary.json"))
            target = gate_b / "rdkitplus" / f"fold{fold}"
            summary = target / "summary.json"
            if not complete_fold(summary):
                run(train_command(
                    python=args.python, associations=train_pairs, protein=protein,
                    reaction=RDKIT_REACTION, schema=RDKIT_REACTION, output=target,
                    dev_fold=fold, recipe=recipe, device=args.device,
                ))
            gate_b_runs.append(("rdkitplus", fold, summary))
        run_selector(gate_b_runs, gate_b / "selection", args.python)
        reaction_name = selected(gate_b / "selection" / "summary.json")
        reaction = RDKIT_REACTION if reaction_name == "rdkitplus" else BASE_REACTION
        schema = RDKIT_REACTION if reaction_name == "rdkitplus" else BASE_SCHEMA

        model_dir = models_root / cell
        if not final_complete(model_dir, train_pairs, protein, reaction):
            run(train_command(
                python=args.python, associations=train_pairs, protein=protein,
                reaction=reaction, schema=schema, output=model_dir,
                dev_fold=-1, recipe=recipe, device=args.device,
            ))

        # The evaluator is the first component in this workflow that reads test_pairs.csv.
        if not eval_complete(eval_root, cell, protein, model_dir):
            run([
                args.python, str(EVAL),
                "--cell", cell,
                "--benchmark-root", str(BENCH),
                "--protein-feature-dir", str(protein),
                "--r2e-model-dir", str(model_dir),
                "--e2r-model-dir", str(model_dir),
                "--r2e-reaction-feature-dir", str(reaction),
                "--e2r-reaction-feature-dir", str(reaction),
                "--output-dir", str(eval_root),
                "--device", args.device,
            ])

        metadata = {
            "cell": cell,
            "freeze": str(FREEZE.resolve()),
            "freeze_sha256": sha256(FREEZE),
            "train_pairs": str(train_pairs.resolve()),
            "train_pairs_sha256": sha256(train_pairs),
            "selected_protein": protein_name,
            "selected_reaction": reaction_name,
            "gate_a_selection": str((gate_a / "selection" / "summary.json").resolve()),
            "gate_b_selection": str((gate_b / "selection" / "summary.json").resolve()),
            "final_model": str(model_dir.resolve()),
            "evaluation": str((eval_root / cell).resolve()),
        }
        cell_selection.mkdir(parents=True, exist_ok=True)
        (cell_selection / "nested_selection_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
