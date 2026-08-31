from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "projects/active/terpene_screening/train_cleanroom_rhea_retriever.py"
SELECT = ROOT / "projects/active/terpene_screening/select_cleanroom_bidirectional_multifold.py"
ASSOCIATIONS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
BASE_SCHEMA = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
BASE_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1"
RDKIT_REACTION = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1"
PROTEIN_CANDIDATES = {
    "esmc": ROOT / "data/catalyst_candidate_universes/general_merged/proteins",
    "enzgfm": ROOT / "data/external/enzgfm_current/clean2023_650m_mean",
    "esmc_enzgfm_equalblock": ROOT / "data/external/enzgfm_current/clean2023_esmc_enzgfm_equalblock",
}
DEFAULT_GATE_A = ROOT / "results/enzgfm_gate_a_bidirectional_v1"
DEFAULT_OUTPUT = ROOT / "results/enzgfm_gate_b_rdkitplus_v1"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def complete_summary(path: Path) -> bool:
    if not path.is_file(): return False
    dev=(json.loads(path.read_text(encoding="utf-8")).get("dev_metrics") or {})
    return "common_ir_r2e" in dev and "common_ir_e2r" in dev


def main() -> None:
    parser=argparse.ArgumentParser(description="Frozen Gate B: test Gate-A-selected protein representation with versus without RDKit+ on the same clean internal folds.")
    parser.add_argument("--gate-a-dir",type=Path,default=DEFAULT_GATE_A)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--python",default=str(ROOT/".venv/bin/python"))
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    gate_a=args.gate_a_dir.resolve(); output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    selection=json.loads((gate_a/"selection/summary.json").read_text(encoding="utf-8"))
    selected=str(selection["selected_candidate"])
    if selected not in PROTEIN_CANDIDATES:
        raise ValueError(f"Unknown Gate-A selected candidate: {selected}")
    protein_dir=PROTEIN_CANDIDATES[selected]
    # Base-reaction results are exactly Gate A for the selected protein representation.
    # Only train the RDKit+ counterpart; selector receives both result sets.
    for fold in [0,1,2]:
        target=output/"rdkitplus"/f"fold{fold}"
        if complete_summary(target/"summary.json") and not args.force:
            print(f"SKIP complete rdkitplus fold{fold}",flush=True); continue
        command=[
            args.python,str(TRAIN),
            "--associations-csv",str(ASSOCIATIONS),
            "--schema-dir",str(RDKIT_REACTION),
            "--reaction-feature-dir",str(RDKIT_REACTION),
            "--protein-feature-dir",str(protein_dir),
            "--output-dir",str(target),
            "--dev-fold",str(fold),"--folds","5",
            "--epochs","8","--steps-per-epoch","60",
            "--reaction-batch-size","64","--protein-batch-size","48",
            "--neighbor-k","32","--dev-neighbor-reactions","10",
            "--hard-negatives","80","--random-negatives","8","--hard-negative-ramp-epochs","0",
            "--hidden-dim","768","--embedding-dim","320","--dropout","0.1",
            "--learning-rate","3e-4","--weight-decay","1e-4","--temperature","0.035",
            "--topk","10","--topk-weight","1.0","--margin","0.12","--r2e-weight","0.70",
            "--reaction-novelty-repeat","0","--seed","20260723","--device",args.device,
        ]
        run(command)
    selector=[args.python,str(SELECT),"--output-dir",str(output/"selection")]
    for fold in [0,1,2]:
        selector += ["--run",f"base_reaction:{fold}:{gate_a/selected/f'fold{fold}'/'summary.json'}"]
        selector += ["--run",f"rdkitplus:{fold}:{output/'rdkitplus'/f'fold{fold}'/'summary.json'}"]
    run(selector)
    metadata={
        "gate_a_selected_candidate":selected,
        "protein_feature_dir":str(protein_dir),
        "base_reaction_results":str(gate_a/selected),
        "rdkitplus_results":str(output/"rdkitplus"),
        "outer_labels_used":False,
    }
    (output/"gate_b_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

if __name__=="__main__": main()
