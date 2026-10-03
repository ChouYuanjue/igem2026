from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.runtime.cli import (
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)


ASSET_ROOT = ROOT.parent / "igem2026"
CENTER_ROOT = (
    ASSET_ROOT
    / "results/cleanroom_internal_reaction_center_bounded_v3"
)
PROTEINS = (
    ASSET_ROOT
    / "data/catalyst_candidate_universes/general_merged/proteins"
)
REACTIONS = (
    ASSET_ROOT
    / "data/catalyst_candidate_universes/general_merged/reaction_features/"
    "drfp_categorical_rdkitplus_center_v1"
)
MAPPING = (
    ASSET_ROOT
    / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
)
OUT = ROOT / "results/fibre_reaction_center_scientific_evidence_v1"
FOLDS = (0, 1, 2)


def encode_rows(
    model: torch.nn.Module,
    matrix: np.ndarray,
    rows: np.ndarray,
    *,
    side: str,
    device: torch.device,
    chunk_size: int = 8192,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), chunk_size):
            batch = torch.as_tensor(
                matrix[rows[start : start + chunk_size]],
                dtype=torch.float32,
                device=device,
            )
            if side == "protein":
                encoded = model.encode_proteins(batch)
            elif side == "reaction":
                encoded = model.encode_reactions(batch)
            else:
                raise ValueError(side)
            pieces.append(encoded.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(pieces, axis=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    protein_matrix, protein_ids = load_protein_library(PROTEINS)
    protein_index = {value: row for row, value in enumerate(protein_ids)}

    first_schema = load_feature_schema(CENTER_ROOT / "cap_0p1/fold0")
    reaction_matrix, reaction_ids = load_registered_reaction_feature_library(
        REACTIONS,
        first_schema,
    )
    reaction_index = {value: row for row, value in enumerate(reaction_ids)}

    mapping = pd.read_csv(MAPPING, dtype=str).fillna("")
    mapping["success_bool"] = (
        mapping["success"].astype(str).str.lower().eq("true")
    )
    mapping["confidence_float"] = pd.to_numeric(
        mapping["confidence"],
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    available_by_reaction = dict(
        zip(
            mapping["reaction_id"].astype(str),
            mapping["success_bool"].astype(bool),
        )
    )
    quality_by_reaction = dict(
        zip(
            mapping["reaction_id"].astype(str),
            mapping["confidence_float"].astype(float),
        )
    )

    tables: list[pd.DataFrame] = []
    fold_audit: list[dict[str, object]] = []
    for fold in FOLDS:
        base_path = CENTER_ROOT / f"base/fold{fold}/dev_pair_scores.csv"
        frame = pd.read_csv(base_path, dtype={"reaction_id": str, "protein_id": str})
        frame["label"] = frame["label"].astype(int)
        frame["score"] = frame["score"].astype(float)
        missing_proteins = sorted(set(frame["protein_id"]) - set(protein_index))
        missing_reactions = sorted(set(frame["reaction_id"]) - set(reaction_index))
        if missing_proteins or missing_reactions:
            raise RuntimeError(
                f"fold {fold} falls outside registered assets: "
                f"proteins={missing_proteins[:3]} reactions={missing_reactions[:3]}"
            )

        model_dir = CENTER_ROOT / f"cap_0p1/fold{fold}/models"
        models = load_models(model_dir, "production", device)
        if len(models) != 1:
            raise RuntimeError(f"fold {fold}: expected one cap0p1 checkpoint")
        model = models[0]

        unique_proteins = sorted(frame["protein_id"].astype(str).unique())
        unique_reactions = sorted(frame["reaction_id"].astype(str).unique())
        protein_rows = np.asarray(
            [protein_index[value] for value in unique_proteins],
            dtype=np.int64,
        )
        reaction_rows = np.asarray(
            [reaction_index[value] for value in unique_reactions],
            dtype=np.int64,
        )
        protein_embedding = encode_rows(
            model,
            protein_matrix,
            protein_rows,
            side="protein",
            device=device,
        )
        reaction_embedding = encode_rows(
            model,
            reaction_matrix,
            reaction_rows,
            side="reaction",
            device=device,
        )
        p_local = {value: row for row, value in enumerate(unique_proteins)}
        r_local = {value: row for row, value in enumerate(unique_reactions)}
        p_rows = np.asarray(
            [p_local[value] for value in frame["protein_id"].astype(str)],
            dtype=np.int64,
        )
        r_rows = np.asarray(
            [r_local[value] for value in frame["reaction_id"].astype(str)],
            dtype=np.int64,
        )
        center_score = (
            reaction_embedding[r_rows] * protein_embedding[p_rows]
        ).sum(axis=1).astype(np.float64)
        base_score = frame["score"].to_numpy(np.float64)
        evidence = center_score - base_score

        available = frame["reaction_id"].map(available_by_reaction).fillna(False).astype(bool)
        quality = (
            frame["reaction_id"].map(quality_by_reaction).fillna(0.0).astype(float)
        )
        # Mapping failures have a deterministic zero fallback centre block. They
        # are explicitly marked unavailable rather than interpreted as weak evidence.
        evidence = np.where(available.to_numpy(), evidence, 0.0)

        local = pd.DataFrame(
            {
                "query_id": frame["reaction_id"].astype(str),
                "candidate_id": frame["protein_id"].astype(str),
                "core_score": base_score,
                "evidence_score": evidence,
                "available": available.to_numpy(bool),
                "quality": quality.to_numpy(float),
                "label": frame["label"].to_numpy(int),
                "fold": np.full(len(frame), fold, dtype=np.int64),
            }
        )
        tables.append(local)
        fold_audit.append(
            {
                "fold": fold,
                "rows": int(len(local)),
                "queries": int(local["query_id"].nunique()),
                "positive_rows": int((local["label"] == 1).sum()),
                "available_rows": int(local["available"].sum()),
                "available_queries": int(
                    local.loc[local["available"], "query_id"].nunique()
                ),
                "mean_mapping_confidence_available": float(
                    local.loc[local["available"], "quality"].mean()
                ),
                "mean_abs_center_delta_available": float(
                    np.abs(
                        local.loc[local["available"], "evidence_score"].to_numpy(float)
                    ).mean()
                ),
            }
        )
        print(json.dumps(fold_audit[-1]), flush=True)
        del model, models, protein_embedding, reaction_embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    table = pd.concat(tables, ignore_index=True)
    table.to_csv(OUT / "admission_table.csv", index=False)
    summary = {
        "schema": "fibre-reaction-center-evidence-export-v1",
        "source": "bounded identity-preserving reaction-centre residual, cap=0.1",
        "evidence_semantics": (
            "candidate score change induced only by the frozen-base "
            "reaction-centre residual branch"
        ),
        "availability": "RXNMapper success for the query reaction",
        "quality": "RXNMapper mapping confidence, used only as an optional reliability covariate",
        "folds": fold_audit,
        "rows": int(len(table)),
        "queries": int(table["query_id"].nunique()),
        "positive_rows": int((table["label"] == 1).sum()),
    }
    (OUT / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
