from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.rank_open_world import (
    encode_reaction_with_audit,
    load_feature_schema,
)
from projects.active.terpene_screening.fair_benchmark import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCHEMA_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"


def stable_query_id(prefix: str, reaction_smiles: str) -> str:
    digest = hashlib.sha256(str(reaction_smiles).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Catalyst 2115d query-reaction features while reading only a specified reaction-SMILES column. "
            "No enzyme/protein/label column is loaded."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reaction-column", default="CANO_RXN_SMILES")
    parser.add_argument("--query-prefix", required=True)
    parser.add_argument("--source-layer", required=True)
    parser.add_argument("--schema-dir", type=Path, default=DEFAULT_SCHEMA_DIR)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--feature-output-dir", type=Path, required=True)
    args = parser.parse_args()

    input_path = args.input.resolve()
    schema_dir = args.schema_dir.resolve()
    registry_path = args.registry_output.resolve()
    output = args.feature_output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    # Intentionally load only reaction chemistry. Target enzyme IDs, sequences,
    # labels, candidate scores, and evaluation truth never enter this process.
    chemistry = pd.read_csv(input_path, usecols=[args.reaction_column], dtype=str).fillna("")
    reactions = sorted({str(value) for value in chemistry[args.reaction_column] if str(value)})
    if not reactions:
        raise ValueError("no non-empty reaction SMILES found")
    query_ids = [stable_query_id(args.query_prefix, reaction) for reaction in reactions]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("stable query-ID collision")

    registry = pd.DataFrame(
        {
            "reaction_smiles": reactions,
            "reaction_id": query_ids,
            "source_layer": args.source_layer,
        }
    )
    registry.to_csv(registry_path, index=False)

    schema = load_feature_schema(schema_dir)
    expected_dim = int(schema.get("reaction_feature_dimension") or 0)
    if expected_dim <= 0:
        raise ValueError("reference schema has no positive reaction_feature_dimension")
    matrix = np.zeros((len(registry), expected_dim), dtype=np.float32)
    audits: list[dict[str, object]] = []
    for row, record in enumerate(registry.itertuples(index=False)):
        feature, audit = encode_reaction_with_audit(
            str(record.reaction_smiles), schema, failure_policy="fallback"
        )
        if feature.shape != (expected_dim,):
            raise ValueError(f"feature width drift for {record.reaction_id}: {feature.shape}")
        matrix[row] = feature
        audits.append(
            {
                "row": row,
                "reaction_id": str(record.reaction_id),
                "status": audit.status,
                "drfp_status": audit.drfp_status,
                "fallback_used": bool(audit.fallback_used),
                "warning": audit.warning,
            }
        )
    np.save(output / "reaction_feature_matrix.npy", matrix)
    pd.DataFrame({"row": range(len(query_ids)), "reaction_id": query_ids}).to_csv(
        output / "entries.csv", index=False
    )
    pd.DataFrame(audits).to_csv(output / "audit.csv", index=False)

    schema_out = dict(schema)
    schema_out["reaction_ids"] = query_ids
    schema_out["reaction_feature_dimension"] = expected_dim
    schema_out["labels_read"] = False
    (output / "feature_schema.json").write_text(
        json.dumps(schema_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    contract = {
        "reaction_feature_dimension": expected_dim,
        "feature_mode": str(schema.get("feature_mode") or ""),
        "drfp_dimension": int(schema.get("drfp_dimension") or 0),
    }
    manifest = {
        "version": "label-blind-query-reaction-base2115-v1",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "reaction_column_loaded_exclusively": args.reaction_column,
        "query_prefix": args.query_prefix,
        "source_layer": args.source_layer,
        "reaction_count": len(query_ids),
        "feature_dimension": expected_dim,
        "fallback_count": int(sum(bool(row["fallback_used"]) for row in audits)),
        "schema_source": str(schema_dir),
        "registry": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "labels_read": False,
        "protein_columns_read": False,
        "candidate_scores_read": False,
        "contract": contract,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
