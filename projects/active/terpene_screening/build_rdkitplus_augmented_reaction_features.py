from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
for path in (ROOT, HORIZYN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from horizyn.datasets.csv import CSVDataset  # noqa: E402
from horizyn.datasets.fingerprints.rdkit_plus import RDKitPlusFingerprintDataset  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Append official Horizyn RDKit+ structural fingerprints to the registered Catalyst reaction feature library.")
    parser.add_argument("--reactions", type=Path, default=ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv")
    parser.add_argument("--base-feature-dir", type=Path, default=ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1")
    parser.add_argument("--base-schema-dir", type=Path, default=ROOT / "results/terpene_production_models/marts_adapted_drfp_pu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1")
    args = parser.parse_args()

    source = args.reactions.resolve(); base_dir = args.base_feature_dir.resolve(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reactions = pd.read_csv(source, dtype=str).fillna("")
    base_entries = pd.read_csv(base_dir / "entries.csv", dtype={"reaction_id": str}).sort_values("row")
    base = np.load(base_dir / "reaction_feature_matrix.npy").astype(np.float32)
    if len(base_entries) != len(base):
        raise ValueError("base entries/features differ in row count")
    lookup = reactions.set_index("reaction_id")["reaction_smiles"].to_dict()
    missing = [rid for rid in base_entries["reaction_id"].astype(str) if rid not in lookup]
    if missing:
        raise ValueError(f"reaction source misses registered IDs: {missing[:5]}")

    ordered_csv = out / "_ordered_reactions.csv"
    pd.DataFrame({
        "reaction_id": base_entries["reaction_id"].astype(str),
        "reaction_smiles": [lookup[rid] for rid in base_entries["reaction_id"].astype(str)],
    }).to_csv(ordered_csv, index=False)
    dataset = CSVDataset(str(ordered_csv), key_column="reaction_id", columns=["reaction_smiles"])
    fp = RDKitPlusFingerprintDataset(
        reaction_dataset=dataset,
        vec_dim=1024,
        mol_fp_type="morgan",
        rxn_fp_type="struct",
        use_chirality=False,
        standardize=True,
        standardize_hypervalent=True,
        standardize_remove_hs=True,
        standardize_kekulize=False,
        standardize_uncharge=True,
        standardize_metals=True,
    )
    rdkit = np.zeros((len(base_entries), 1024), dtype=np.float32)
    audit: list[dict[str, object]] = []
    for row, rid in enumerate(base_entries["reaction_id"].astype(str)):
        try:
            rdkit[row] = fp[rid].detach().cpu().numpy().astype(np.float32)
            audit.append({"row": row, "reaction_id": rid, "status": "valid", "warning": ""})
        except Exception as exc:
            audit.append({"row": row, "reaction_id": rid, "status": "zero_fallback", "warning": str(exc)[:1000]})
    augmented = np.concatenate([base, rdkit], axis=1).astype(np.float32)
    np.save(out / "reaction_feature_matrix.npy", augmented)
    base_entries.to_csv(out / "entries.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)

    schema = json.loads((args.base_schema_dir.resolve() / "feature_schema.json").read_text(encoding="utf-8"))
    old_dim = int(schema.get("reaction_feature_dimension") or base.shape[1])
    if old_dim != base.shape[1]:
        raise ValueError(f"base schema width {old_dim} != matrix width {base.shape[1]}")
    schema["reaction_feature_dimension"] = int(augmented.shape[1])
    schema["reaction_feature_mode_extension"] = "append_horizyn_rdkitplus_struct_morgan1024"
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    contract = dict(base_manifest.get("contract") or {})
    contract["reaction_feature_dimension"] = int(augmented.shape[1])
    manifest = {
        "version": "general-merged-reaction-features-rdkitplus-v1",
        "base_feature_dir": str(base_dir),
        "base_feature_manifest_sha256": sha256_file(base_dir / "manifest.json"),
        "reaction_source": str(source),
        "reaction_source_sha256": sha256_file(source),
        "feature_dimension": int(augmented.shape[1]),
        "base_dimension": int(base.shape[1]),
        "rdkitplus_dimension": 1024,
        "rdkitplus_provenance": "external/horizyn RDKitPlusFingerprintDataset",
        "rdkitplus_license": "PolyForm Noncommercial 1.0.0; invoked as external research dependency, source not copied",
        "rdkitplus_settings": {
            "mol_fp_type": "morgan", "radius": 3, "rxn_fp_type": "struct", "use_chirality": False,
            "standardize": True, "standardize_uncharge": True,
        },
        "reaction_count": int(len(base_entries)),
        "valid_rdkitplus": int(sum(row["status"] == "valid" for row in audit)),
        "zero_fallback_rdkitplus": int(sum(row["status"] != "valid" for row in audit)),
        "contract": contract,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ordered_csv.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
