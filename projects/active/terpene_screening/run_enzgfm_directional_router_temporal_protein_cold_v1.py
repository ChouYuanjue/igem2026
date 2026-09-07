from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.run_enzgfm_nested_outer_v2 import (
    complete_fold,
    final_complete,
    train_command,
)

TRAIN = ROOT / "projects/active/terpene_screening/train_cleanroom_rhea_retriever.py"
SELECT = ROOT / "projects/active/terpene_screening/select_cleanroom_directional_multifold.py"
EVAL = ROOT / "projects/active/terpene_screening/evaluate_broad_rhea_benchmark.py"
FREEZE = ROOT / "projects/active/terpene_screening/CLEANROOM_ENZGFM_DIRECTIONAL_ROUTER_TEMPORAL_PROTEIN_COLD_V1.json"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
CELL = "temporal_post2020_protein_cold"
BASE_SCHEMA = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
BASE_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1"
RDKIT_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
PROTEINS = {
    "esmc": ROOT / "data/catalyst_candidate_universes/general_merged/proteins",
    "enzgfm": ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1",
    "esmc_enzgfm_equalblock": ROOT / "data/external/enzgfm_current/general_merged_esmc_enzgfm_equalblock_v1",
}
REACTIONS = {
    "base_reaction": (BASE_REACTION, BASE_SCHEMA),
    "rdkitplus": (RDKIT_REACTION, RDKIT_REACTION),
}
PARENT_SELECTION = ROOT / "results/enzgfm_nested_outer_v2_selection" / CELL
DEFAULT_SELECTION = ROOT / "results/enzgfm_directional_router_temporal_protein_cold_v1_selection"
DEFAULT_MODELS = ROOT / "results/enzgfm_directional_router_temporal_protein_cold_v1_models"
DEFAULT_EVAL = ROOT / "results/enzgfm_directional_router_temporal_protein_cold_v1_eval"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def candidate_label(protein: str, reaction: str) -> str:
    if "__" in protein or "__" in reaction:
        raise ValueError("candidate names may not contain double underscore")
    return f"{protein}__{reaction}"


def parse_candidate(value: str) -> tuple[str, str]:
    protein, reaction = value.rsplit("__", 1)
    if protein not in PROTEINS or reaction not in REACTIONS:
        raise ValueError(f"unknown directional candidate {value!r}")
    return protein, reaction


def selected(path: Path, key: str) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = str(payload.get(key) or "")
    if not value:
        raise RuntimeError(f"No {key} in {path}")
    return value


def parent_base_summary(protein: str, fold: int) -> Path:
    return PARENT_SELECTION / "gate_a" / protein / f"fold{fold}" / "summary.json"


def final_model_complete(path: Path, train_pairs: Path, protein: Path, reaction: Path) -> bool:
    return final_complete(path, train_pairs, protein, reaction)


def evaluation_complete(path: Path, *, r2e_model: Path, e2r_model: Path, r2e_protein: Path, e2r_protein: Path) -> bool:
    summary = path / CELL / "summary.json"
    if not summary.is_file():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (
        payload.get("cell") == CELL
        and Path(str(payload.get("r2e_model_dir"))).resolve() == r2e_model.resolve()
        and Path(str(payload.get("e2r_model_dir"))).resolve() == e2r_model.resolve()
        and Path(str(payload.get("r2e_protein_feature_dir"))).resolve() == r2e_protein.resolve()
        and Path(str(payload.get("e2r_protein_feature_dir"))).resolve() == e2r_protein.resolve()
        and "reaction_to_enzyme" in (payload.get("metrics") or {})
        and "enzyme_to_reaction" in (payload.get("metrics") or {})
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen direction-specific EnzGFM/RDKit+ factorial expert routing for untouched temporal protein-cold outer."
    )
    parser.add_argument("--selection-root", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["target_cell"] != CELL or freeze.get("outer_test_metrics_used_for_design_or_selection") is not False:
        raise RuntimeError("freeze contract mismatch")
    if list(freeze["protein_candidates"]) != list(PROTEINS):
        raise RuntimeError("frozen protein candidates differ from runner")
    if list(freeze["reaction_candidates"]) != list(REACTIONS):
        raise RuntimeError("frozen reaction candidates differ from runner")
    recipe = dict(freeze["training_recipe"])
    folds = [int(value) for value in freeze["inner_selection"]["folds"]]
    train_pairs = BENCH / CELL / "train_pairs.csv"
    test_pairs = BENCH / CELL / "test_pairs.csv"
    if not train_pairs.is_file() or not test_pairs.is_file():
        raise FileNotFoundError("benchmark cell is incomplete")

    selection_root = args.selection_root.resolve()
    models_root = args.models_root.resolve()
    eval_root = args.eval_root.resolve()
    selection_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, int, Path]] = []
    for protein_name, protein_dir in PROTEINS.items():
        for reaction_name, (reaction_dir, schema_dir) in REACTIONS.items():
            label = candidate_label(protein_name, reaction_name)
            for fold in folds:
                if reaction_name == "base_reaction":
                    summary = parent_base_summary(protein_name, fold)
                    if not complete_fold(summary):
                        raise RuntimeError(f"parent train-only Gate-A summary missing/incomplete: {summary}")
                else:
                    target = selection_root / "factorial" / label / f"fold{fold}"
                    summary = target / "summary.json"
                    if not complete_fold(summary):
                        run(
                            train_command(
                                python=args.python,
                                associations=train_pairs,
                                protein=protein_dir,
                                reaction=reaction_dir,
                                schema=schema_dir,
                                output=target,
                                dev_fold=fold,
                                recipe=recipe,
                                device=args.device,
                            )
                        )
                runs.append((label, fold, summary))

    selector_dir = selection_root / "selection"
    selector_summary = selector_dir / "summary.json"
    if not selector_summary.is_file():
        command = [args.python, str(SELECT), "--output-dir", str(selector_dir)]
        for label, fold, summary in runs:
            command += ["--run", f"{label}:{fold}:{summary}"]
        run(command)

    r2e_label = selected(selector_summary, "selected_r2e_candidate")
    e2r_label = selected(selector_summary, "selected_e2r_candidate")
    r2e_protein_name, r2e_reaction_name = parse_candidate(r2e_label)
    e2r_protein_name, e2r_reaction_name = parse_candidate(e2r_label)
    r2e_protein = PROTEINS[r2e_protein_name]
    e2r_protein = PROTEINS[e2r_protein_name]
    r2e_reaction, r2e_schema = REACTIONS[r2e_reaction_name]
    e2r_reaction, e2r_schema = REACTIONS[e2r_reaction_name]

    if r2e_label == e2r_label:
        shared = models_root / "shared"
        if not final_model_complete(shared, train_pairs, r2e_protein, r2e_reaction):
            run(
                train_command(
                    python=args.python, associations=train_pairs, protein=r2e_protein,
                    reaction=r2e_reaction, schema=r2e_schema, output=shared,
                    dev_fold=-1, recipe=recipe, device=args.device,
                )
            )
        r2e_model = e2r_model = shared
    else:
        r2e_model = models_root / "r2e"
        e2r_model = models_root / "e2r"
        if not final_model_complete(r2e_model, train_pairs, r2e_protein, r2e_reaction):
            run(
                train_command(
                    python=args.python, associations=train_pairs, protein=r2e_protein,
                    reaction=r2e_reaction, schema=r2e_schema, output=r2e_model,
                    dev_fold=-1, recipe=recipe, device=args.device,
                )
            )
        if not final_model_complete(e2r_model, train_pairs, e2r_protein, e2r_reaction):
            run(
                train_command(
                    python=args.python, associations=train_pairs, protein=e2r_protein,
                    reaction=e2r_reaction, schema=e2r_schema, output=e2r_model,
                    dev_fold=-1, recipe=recipe, device=args.device,
                )
            )

    if not evaluation_complete(
        eval_root, r2e_model=r2e_model, e2r_model=e2r_model,
        r2e_protein=r2e_protein, e2r_protein=e2r_protein,
    ):
        run(
            [
                args.python, str(EVAL),
                "--cell", CELL,
                "--benchmark-root", str(BENCH),
                "--r2e-protein-feature-dir", str(r2e_protein),
                "--e2r-protein-feature-dir", str(e2r_protein),
                "--r2e-model-dir", str(r2e_model),
                "--e2r-model-dir", str(e2r_model),
                "--r2e-reaction-feature-dir", str(r2e_reaction),
                "--e2r-reaction-feature-dir", str(e2r_reaction),
                "--output-dir", str(eval_root),
                "--device", args.device,
            ]
        )

    metadata = {
        "cell": CELL,
        "freeze": str(FREEZE.resolve()),
        "freeze_sha256": sha256(FREEZE),
        "train_pairs": str(train_pairs.resolve()),
        "train_pairs_sha256": sha256(train_pairs),
        "test_pairs_sha256": sha256(test_pairs),
        "selector": str(selector_summary.resolve()),
        "selected_r2e_candidate": r2e_label,
        "selected_e2r_candidate": e2r_label,
        "r2e_model": str(r2e_model.resolve()),
        "e2r_model": str(e2r_model.resolve()),
        "evaluation": str((eval_root / CELL).resolve()),
        "outer_test_metrics_used_for_selection": False,
    }
    (selection_root / "directional_router_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
