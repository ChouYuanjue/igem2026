from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "reproducibility/bime_rank/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_enzymecage_internal_reaction_features_v1 as builder
from prepare_enzymecage_full_outer_reaction_probe_v1 import valid_center_queries

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
INTERNAL_PROBE = ROOT / "results/enzymecage_family_response_v1/internal_folds/fold0/pairs.csv"
INTERNAL_CONFIG = ROOT / "results/enzymecage_family_response_v1/internal_folds/fold0/probe.yaml"
OUT_ROOT = ROOT / "results/enzymecage_relation_unseen_reaction_assets_v1"
FEATURE = OUT_ROOT / "feature/reaction"
PROBE = OUT_ROOT / "probe_pairs.csv"
CONFIG = OUT_ROOT / "probe.yaml"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE.mkdir(parents=True, exist_ok=True)

    q = set(pd.read_csv(TARGETS, usecols=["reaction_id"], dtype=str).reaction_id.astype(str))
    meta = pd.read_csv(REACTIONS, dtype=str).fillna("")
    frame = meta[meta.reaction_id.astype(str).isin(q)][["reaction_id", "reaction_smiles"]].drop_duplicates("reaction_id")
    missing = sorted(q - set(frame.reaction_id.astype(str)))
    if missing:
        raise RuntimeError(f"missing reaction metadata: {missing[:10]}")
    frame = frame.rename(columns={"reaction_smiles": "CANO_RXN_SMILES"}).sort_values("reaction_id", kind="stable").reset_index(drop=True)

    placeholder = pd.read_csv(INTERNAL_PROBE, dtype=str).fillna("").iloc[0]
    probe = pd.DataFrame({
        "reaction_id": frame.reaction_id.astype(str),
        "protein_id": str(placeholder["protein_id"]),
        "UniprotID": str(placeholder["UniprotID"]),
        "sequence": str(placeholder["sequence"]),
        "CANO_RXN_SMILES": frame.CANO_RXN_SMILES.astype(str),
        "Label": 0.0,
    })

    builder.OUT = FEATURE
    (FEATURE / "drfp").mkdir(parents=True, exist_ok=True)
    (FEATURE / "reacting_center").mkdir(parents=True, exist_ok=True)

    drfp, drfp_fallbacks = builder.generate_drfp(frame)
    with (FEATURE / "drfp/rxn2fp.pkl").open("wb") as fh:
        pickle.dump(drfp, fh)
    pd.DataFrame(drfp_fallbacks).to_csv(FEATURE / "drfp/fallbacks.csv", index=False)

    centers, center_failures = builder.generate_centers(frame)
    valid_ids, invalid_centers = valid_center_queries(frame, centers)
    with (FEATURE / "reacting_center/reacting_center.pkl").open("wb") as fh:
        pickle.dump(centers, fh)
    pd.DataFrame(center_failures).to_csv(FEATURE / "reacting_center/failures.csv", index=False)

    molecules = builder.unique_molecules(frame)
    mol_index, conformer_failures = builder.generate_conformations(molecules)
    pd.DataFrame(conformer_failures).to_csv(FEATURE / "molecule_conformation/failures.csv", index=False)
    failed_mols = {str(x["smiles"]) for x in conformer_failures}
    failed_conformer_queries = set()
    if failed_mols:
        for rid, reaction in frame[["reaction_id", "CANO_RXN_SMILES"]].itertuples(index=False):
            left, right = str(reaction).split(">>")
            used = {p.replace("*", "C") for p in left.split(".") + right.split(".") if p}
            if used & failed_mols:
                failed_conformer_queries.add(str(rid))

    valid_ids -= failed_conformer_queries
    probe = probe[probe.reaction_id.astype(str).isin(valid_ids)].copy()
    probe.to_csv(PROBE, index=False)

    raw = yaml.safe_load(INTERNAL_CONFIG.read_text())
    raw["data_path"] = str(PROBE.resolve())
    raw["rxn_fp"] = str((FEATURE / "drfp/rxn2fp.pkl").resolve())
    raw["mol_conformation"] = str((FEATURE / "molecule_conformation").resolve())
    raw["reaction_center"] = str((FEATURE / "reacting_center/reacting_center.pkl").resolve())
    CONFIG.write_text(yaml.safe_dump(raw, sort_keys=False))

    summary = {
        "schema": "enzymecage-relation-unseen-reaction-assets-v1",
        "status": "completed",
        "queries": int(len(frame)),
        "probe_queries": int(len(probe)),
        "unavailable_queries": int(len(frame) - len(probe)),
        "invalid_reaction_center_queries": sorted(invalid_centers),
        "failed_conformer_queries": sorted(failed_conformer_queries),
        "molecules": int(len(mol_index)),
        "conformer_failures": int(len(conformer_failures)),
        "drfp_fallbacks": int(len(drfp_fallbacks)),
        "labels_used": False,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
