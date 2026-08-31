from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "projects/active/terpene_screening/train_cleanroom_rhea_retriever.py"
SELECT = ROOT / "projects/active/terpene_screening/select_cleanroom_bidirectional_multifold.py"
ASSOCIATIONS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
SCHEMA = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1"
CANDIDATES = {
    "esmc": ROOT / "data/catalyst_candidate_universes/general_merged/proteins",
    "enzgfm": ROOT / "data/external/enzgfm_current/clean2023_650m_mean",
    "esmc_enzgfm_equalblock": ROOT / "data/external/enzgfm_current/clean2023_esmc_enzgfm_equalblock",
}
DEFAULT_OUTPUT = ROOT / "results/enzgfm_gate_a_bidirectional_v1"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def complete_summary(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    dev = payload.get("dev_metrics") or {}
    return "common_ir_r2e" in dev and "common_ir_e2r" in dev


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen EnzGFM Gate A on clean2023 folds 0/1/2 and select with balanced bidirectional metrics.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name, directory in CANDIDATES.items():
        for required in [directory / "entries.csv", directory / "embeddings.npy"]:
            if not required.is_file():
                raise FileNotFoundError(required)
        for fold in [0, 1, 2]:
            target = output / name / f"fold{fold}"
            summary = target / "summary.json"
            if complete_summary(summary) and not args.force:
                print(f"SKIP complete {name} fold{fold}", flush=True)
                continue
            command = [
                args.python, str(TRAIN),
                "--associations-csv", str(ASSOCIATIONS),
                "--schema-dir", str(SCHEMA),
                "--reaction-feature-dir", str(REACTION),
                "--protein-feature-dir", str(directory),
                "--output-dir", str(target),
                "--dev-fold", str(fold), "--folds", "5",
                "--epochs", "8", "--steps-per-epoch", "60",
                "--reaction-batch-size", "64", "--protein-batch-size", "48",
                "--neighbor-k", "32", "--dev-neighbor-reactions", "10",
                "--hard-negatives", "80", "--random-negatives", "8",
                "--hard-negative-ramp-epochs", "0",
                "--hidden-dim", "768", "--embedding-dim", "320", "--dropout", "0.1",
                "--learning-rate", "3e-4", "--weight-decay", "1e-4",
                "--temperature", "0.035", "--topk", "10", "--topk-weight", "1.0",
                "--margin", "0.12", "--r2e-weight", "0.70",
                "--reaction-novelty-repeat", "0",
                "--seed", "20260723", "--device", args.device,
            ]
            run(command)
    selector = [args.python, str(SELECT), "--output-dir", str(output / "selection")]
    for name in CANDIDATES:
        for fold in [0, 1, 2]:
            selector += ["--run", f"{name}:{fold}:{output / name / f'fold{fold}' / 'summary.json'}"]
    run(selector)


if __name__ == "__main__":
    main()
