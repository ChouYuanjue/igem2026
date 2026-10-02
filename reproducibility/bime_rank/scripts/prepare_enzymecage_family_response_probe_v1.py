from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/fibre_vs_enzymecage_external_families_v1"
CAGE = ROOT / "external_repos/EnzymeCAGE"
OUT = ROOT / "results/enzymecage_family_response_v1"
FAMILIES = {
    "p450": ("P450", CAGE / "dataset/external-test-set/p450/feature"),
    "phosphatase": ("Phosphatase", CAGE / "dataset/external-test-set/phosphatase/feature"),
    "terpene": ("Terpene", CAGE / "dataset/external-test-set/terpene/feature"),
}
TOPK = 64


def main() -> None:
    summary = {"schema": "enzymecage-family-response-probe-v1", "topk": TOPK, "families": {}}
    for family, (label, feature) in FAMILIES.items():
        source = pd.read_csv(
            SOURCE / f"{family}_broad_pair_scores.csv.gz",
            dtype={"UniprotID": str},
        ).fillna("")
        query_col = "CANO_RXN_SMILES"
        probe = (
            source.sort_values(
                [query_col, "broad_score", "UniprotID"],
                ascending=[True, False, True],
                kind="stable",
            )
            .groupby(query_col, sort=False)
            .head(TOPK)
            .copy()
        )
        # Extraction never reads the target labels. Keep a zero placeholder to satisfy
        # the author CAGE dataset class and remove any accidental label dependence.
        probe["Label"] = 0.0
        official = pd.read_csv(
            CAGE / f"dataset/external-test-set/{family}/test_{label}.csv",
            dtype={"UniprotID": str},
        ).fillna("")
        sequence_map = (
            official[["UniprotID", "sequence"]]
            .drop_duplicates("UniprotID", keep="first")
            .set_index("UniprotID")["sequence"]
            .to_dict()
        )
        probe["sequence"] = probe["UniprotID"].map(sequence_map).fillna("")
        if probe["sequence"].eq("").any():
            raise RuntimeError(f"{family}: missing sequences after official-table merge")
        keep = ["UniprotID", "sequence", query_col, "Label"]
        probe = probe[keep].drop_duplicates([query_col, "UniprotID"], keep="first")
        out = OUT / f"{family}_broad_top{TOPK}_probe"
        out.mkdir(parents=True, exist_ok=True)
        data_path = out / "pairs.csv"
        probe.to_csv(data_path, index=False)

        config = {
            "model": "EnzymeCAGE",
            "interaction_method": "geo-enhanced-interaction",
            "rxn_inner_interaction": True,
            "pocket_inner_interaction": True,
            "use_prods_info": False,
            "use_structure": True,
            "use_drfp": True,
            "use_esm": True,
            "esm_model": "ESM-C_600M",
            "batch_size": 256,
            "data_path": str(data_path.resolve()),
            "rxn_fp": str((feature / "reaction/drfp/rxn2fp.pkl").resolve()),
            "mol_conformation": str((feature / "reaction/molecule_conformation").resolve()),
            "reaction_center": str((feature / "reaction/reacting_center/reacting_center.pkl").resolve()),
            "protein_gvp_feat": str((feature / "protein/gvp_feature/gvp_protein_feature.pt").resolve()),
            "esm_mean_feature": str((feature / "protein/ESM-C_600M/protein_level/seq2feature.pkl").resolve()),
            "esm_node_feature": str((feature / "protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt").resolve()),
        }
        config_path = out / "probe.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        summary["families"][family] = {
            "queries": int(probe[query_col].nunique()),
            "rows": int(len(probe)),
            "source_rows": int(len(source)),
            "config": str(config_path.relative_to(ROOT)),
        }

    (OUT / "probe_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
