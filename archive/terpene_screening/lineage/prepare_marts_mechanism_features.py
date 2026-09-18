from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STEPS = ROOT / "data/terpene_marts/marts_mechanism_steps.tsv"
DEFAULT_PAIRS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_mechanism_features_v1"


def normalize_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_") or "unknown"


def carbocation_count(value: object) -> int:
    text = str(value)
    return len(re.findall(r"\[[^\]]*\+[^\]]*\]", text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build auditable MARTS mechanism-step features and a non-learned similarity baseline.")
    parser.add_argument("--steps", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    steps = pd.read_csv(args.steps, sep="\t", dtype=str).fillna("")
    required = {"Mechanism_marts_id", "Reaction_type", "step_index", "step_reaction_smiles", "Evidence"}
    missing = required - set(steps.columns)
    if missing:
        raise ValueError(f"Mechanism step file missing columns: {sorted(missing)}")
    steps["step_index_numeric"] = pd.to_numeric(steps["step_index"], errors="coerce")
    steps["reaction_type_token"] = steps["Reaction_type"].map(normalize_token)
    steps["evidence_token"] = steps["Evidence"].map(normalize_token)
    steps["carbocation_mentions"] = steps["step_reaction_smiles"].map(carbocation_count)
    steps = steps.sort_values(["Mechanism_marts_id", "step_index_numeric", "step_index"]).reset_index(drop=True)

    vocabulary = sorted(steps["reaction_type_token"].unique())
    evidence_vocabulary = sorted(steps["evidence_token"].unique())
    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    mechanism_ids: list[str] = []
    for mechanism_id, group in steps.groupby("Mechanism_marts_id", sort=True):
        counts = group["reaction_type_token"].value_counts()
        vector = np.asarray([float(counts.get(token, 0)) for token in vocabulary], dtype=np.float32)
        vectors.append(vector)
        mechanism_ids.append(str(mechanism_id))
        first = group.iloc[0]
        last = group.iloc[-1]
        row: dict[str, object] = {
            "mechanism_marts_id": str(mechanism_id),
            "n_steps": len(group),
            "max_step_index": float(group["step_index_numeric"].max()) if group["step_index_numeric"].notna().any() else np.nan,
            "reaction_type_sequence": ">".join(group["reaction_type_token"].astype(str)),
            "unique_reaction_types": int(group["reaction_type_token"].nunique()),
            "evidence_sequence": ">".join(group["evidence_token"].astype(str)),
            "carbocation_mentions": int(group["carbocation_mentions"].sum()),
            "start_substrate_smiles": str(first.get("Substrate_smiles", "")),
            "final_product_smiles": str(last.get("Product_smiles", "")),
            "start_substrate_name": str(first.get("Substrate_name", "")),
            "final_product_name": str(last.get("Product_name", "")),
            "publication": str(first.get("Publication", "")),
        }
        for token in vocabulary:
            row[f"count_{token}"] = int(counts.get(token, 0))
        rows.append(row)
    matrix = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.where(norms == 0, 1.0, norms)
    cosine = normalized @ normalized.T
    binary = matrix > 0
    intersection = binary.astype(np.float32) @ binary.astype(np.float32).T
    union = binary.sum(axis=1)[:, None] + binary.sum(axis=1)[None, :] - intersection
    jaccard = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    features = pd.DataFrame(rows)
    ids = pd.DataFrame({"row": np.arange(len(mechanism_ids)), "mechanism_marts_id": mechanism_ids})
    steps.to_csv(output / "mechanism_steps_normalized.tsv", sep="\t", index=False)
    features.to_csv(output / "mechanism_features.csv", index=False)
    ids.to_csv(output / "mechanism_ids.csv", index=False)
    np.save(output / "reaction_type_count_matrix.npy", matrix)
    np.save(output / "mechanism_cosine_similarity.npy", cosine.astype(np.float32))
    np.save(output / "mechanism_jaccard_similarity.npy", jaccard.astype(np.float32))

    pairs = pd.read_csv(args.pairs, sep="\t", dtype=str).fillna("")
    pair_coverage = pairs["mechanism_marts_id"].astype(str).isin(set(mechanism_ids)) if "mechanism_marts_id" in pairs else pd.Series(False, index=pairs.index)
    unresolved = pairs.loc[~pair_coverage, [column for column in ["enzyme_id", "reaction_signature", "mechanism_marts_id", "publication"] if column in pairs]].copy()
    unresolved.to_csv(output / "pairs_without_mechanism_features.csv", index=False)
    summary = {
        "status": "ready",
        "n_steps": len(steps),
        "n_mechanisms": len(mechanism_ids),
        "reaction_type_vocabulary": vocabulary,
        "evidence_vocabulary": evidence_vocabulary,
        "pair_rows": len(pairs),
        "pair_rows_with_mechanism_features": int(pair_coverage.sum()),
        "pair_coverage_fraction": float(pair_coverage.mean()) if len(pairs) else 0.0,
        "feature_shape": list(matrix.shape),
        "promotion_policy": "auxiliary_residual_or_kernel_only_until_frozen_R2E_confirmation",
        "outputs": {
            "features": str(output / "mechanism_features.csv"),
            "cosine_similarity": str(output / "mechanism_cosine_similarity.npy"),
            "jaccard_similarity": str(output / "mechanism_jaccard_similarity.npy"),
            "coverage_gaps": str(output / "pairs_without_mechanism_features.csv"),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
