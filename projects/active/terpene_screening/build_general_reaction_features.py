from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from projects.active.terpene_screening.rank_open_world import (
    encode_reaction_with_audit,
    load_feature_schema,
)

UNIVERSE_ROOT = ROOT / "data/catalyst_candidate_universes/general_merged"
REACTIONS = UNIVERSE_ROOT / "reactions.csv"
REFERENCE_DEPLOYMENT = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu_e2r"
OUTPUT = UNIVERSE_ROOT / "reaction_features/drfp_categorical_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_contract(schema: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in schema.items()
        if key
        not in {
            "reaction_ids",
            "n_reactions_without_parseable_smiles",
            "production_training_pairs",
            "protein_ids_file",
        }
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reactions = pd.read_csv(REACTIONS, dtype=str).fillna("")
    if reactions["reaction_id"].duplicated().any():
        raise ValueError("general_merged reactions.csv contains duplicate reaction IDs")
    if reactions["reaction_smiles"].eq("").any():
        missing = reactions.loc[reactions["reaction_smiles"].eq(""), "reaction_id"].head().tolist()
        raise ValueError(f"general_merged reactions lack SMILES: {missing}")

    schema = load_feature_schema(REFERENCE_DEPLOYMENT)
    reference_ids = [str(value) for value in schema["reaction_ids"]]
    reference_matrix = np.load(REFERENCE_DEPLOYMENT / "reaction_feature_matrix.npy").astype(np.float32)
    if len(reference_ids) != len(reference_matrix):
        raise ValueError("reference deployment reaction schema and matrix disagree")
    reference_by_id = {value: index for index, value in enumerate(reference_ids)}

    reaction_ids = reactions["reaction_id"].astype(str).tolist()
    missing_reference = sorted(set(reference_ids) - set(reaction_ids))
    if missing_reference:
        raise ValueError(
            f"general_merged is missing {len(missing_reference)} reference reactions: {missing_reference[:10]}"
        )

    matrix = np.lib.format.open_memmap(
        OUTPUT / "reaction_feature_matrix.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(reactions), int(reference_matrix.shape[1])),
    )
    audit_rows: list[dict[str, object]] = []
    copied = 0
    encoded = 0
    for output_row, record in enumerate(reactions.itertuples(index=False)):
        reaction_id = str(record.reaction_id)
        reaction_smiles = str(record.reaction_smiles)
        reference_row = reference_by_id.get(reaction_id)
        if reference_row is not None:
            matrix[output_row] = reference_matrix[reference_row]
            copied += 1
            audit_rows.append(
                {
                    "row": output_row,
                    "reaction_id": reaction_id,
                    "source": "copied_reference_feature",
                    "status": "precomputed",
                    "drfp_status": "precomputed",
                    "fallback_used": False,
                    "warning": "",
                }
            )
            continue
        feature, audit = encode_reaction_with_audit(
            reaction_smiles,
            schema,
            failure_policy="fallback",
        )
        if feature.shape != (reference_matrix.shape[1],):
            raise ValueError(
                f"reaction feature width mismatch for {reaction_id}: {feature.shape}"
            )
        matrix[output_row] = feature
        encoded += 1
        audit_rows.append(
            {
                "row": output_row,
                "reaction_id": reaction_id,
                "source": "encoded_general_reaction",
                "status": audit.status,
                "drfp_status": audit.drfp_status,
                "fallback_used": bool(audit.fallback_used),
                "warning": audit.warning,
            }
        )
    del matrix

    entries = pd.DataFrame({"row": range(len(reaction_ids)), "reaction_id": reaction_ids})
    entries.to_csv(OUTPUT / "entries.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUTPUT / "audit.csv", index=False)

    built = np.load(OUTPUT / "reaction_feature_matrix.npy", mmap_mode="r")
    if built.shape != (len(reaction_ids), reference_matrix.shape[1]):
        raise ValueError("written general reaction matrix has unexpected shape")
    built_index = {value: index for index, value in enumerate(reaction_ids)}
    for reaction_id, reference_row in reference_by_id.items():
        if not np.array_equal(built[built_index[reaction_id]], reference_matrix[reference_row]):
            raise ValueError(f"reference reaction feature drift: {reaction_id}")

    schema_contract = _schema_contract(schema)
    manifest = {
        "version": "general-merged-reaction-features-v1",
        "candidate_universe": "general_merged",
        "feature_mode": str(schema.get("feature_mode") or ""),
        "feature_dimension": int(reference_matrix.shape[1]),
        "reaction_count": len(reaction_ids),
        "reference_reaction_count": len(reference_ids),
        "reference_features_copied": copied,
        "new_features_encoded": encoded,
        "reaction_source_sha256": _sha256(REACTIONS),
        "reference_schema_sha256": hashlib.sha256(
            json.dumps(schema_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "reference_deployment": str(REFERENCE_DEPLOYMENT.relative_to(ROOT)),
        "contract": schema_contract,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
