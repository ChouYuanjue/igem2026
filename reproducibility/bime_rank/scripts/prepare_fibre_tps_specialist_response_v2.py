from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.application.tps_adapted_coordinate import (
    TPSAdaptedCoordinateProjector,
)
from projects.active.bridge.model.assets import ROOT

OUT = ROOT / "results/fibre_tps_specialist_response_v2"
BROAD_PROTEIN = (
    ROOT / "data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy"
)
BROAD_ENTRIES = (
    ROOT / "data/catalyst_candidate_universes/general_merged/proteins/entries.csv"
)
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
REACTION_FEATURE_ROOT = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1"
V4 = ROOT / "results/fibre_dynamic_router_v4/prepared"
TPS_REFERENCE = ROOT / "results/fibre_application/tps_adapted_coordinate"
TPS_REFERENCE_META = ROOT / "results/fibre_application/full_data/inputs/reactions.csv"

TOPK = 1000
BROAD_BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"


def ordered_ids(path: Path, id_candidates=("Entry", "protein_id")) -> list[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if "row" in frame.columns:
        frame["row"] = pd.to_numeric(frame["row"]).astype(int)
        frame = frame.sort_values("row", kind="stable")
    column = next(c for c in id_candidates if c in frame.columns)
    return frame[column].astype(str).tolist()


def build_protein_cache(device: str, batch_size: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = np.load(BROAD_PROTEIN, mmap_mode="r")
    ids = ordered_ids(BROAD_ENTRIES)
    if len(raw) != len(ids):
        raise RuntimeError("Broad protein rows do not align")
    projector = TPSAdaptedCoordinateProjector(
        device=device,
        batch_size=batch_size,
    )
    z = projector.project_protein_features(raw)
    if z.shape != (len(ids), 768):
        raise RuntimeError(f"unexpected TPS protein coordinate: {z.shape}")
    path = OUT / "broad_protein_tps.npy"
    np.save(path, z.astype(np.float32))
    pd.DataFrame({"row": np.arange(len(ids)), "protein_id": ids}).to_csv(
        OUT / "broad_protein_entries.csv", index=False
    )
    result = {
        "schema": "fibre-tps-specialist-broad-protein-cache-v2",
        "status": "completed",
        "proteins": len(ids),
        "input_dim": int(raw.shape[1]),
        "output_dim": int(z.shape[1]),
        "model": "results/terpene_production_models/marts_adapted_drfp_pu",
        "checkpoint_count": len(projector.models),
        "aggregation": "per-seed L2-normalize -> concatenate -> L2-normalize",
        "labels_used": False,
    }
    (OUT / "protein_cache_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return result


def load_reference() -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(
        TPS_REFERENCE / "reaction_tps_adapted.csv",
        dtype={"reaction_id": str},
    ).fillna("")
    ids = frame.pop("reaction_id").astype(str)
    z = frame.to_numpy(np.float32)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-8)
    meta = pd.read_csv(TPS_REFERENCE_META, dtype=str).fillna("")
    meta = meta[meta["reaction_id"].astype(str).isin(set(ids))].copy()
    meta = meta.drop_duplicates("reaction_id")
    order = pd.DataFrame({"reaction_id": ids})
    meta = order.merge(meta, on="reaction_id", how="left")
    return z, meta


def domain_features(q: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    sim = reference @ q
    ordered = np.sort(sim)[::-1]
    top5 = ordered[: min(5, len(ordered))]
    top20 = ordered[: min(20, len(ordered))]
    # Similarity entropy only describes concentration inside the TPS reference set.
    scaled = (sim - sim.max()) / 0.05
    prob = np.exp(np.clip(scaled, -60.0, 0.0))
    prob /= max(prob.sum(), 1e-12)
    entropy = -(prob * np.log(np.clip(prob, 1e-12, None))).sum()
    return {
        "tps_ref_max_cosine": float(ordered[0]),
        "tps_ref_top5_mean": float(top5.mean()),
        "tps_ref_top20_mean": float(top20.mean()),
        "tps_ref_margin_1_2": float(
            ordered[0] - ordered[1] if len(ordered) > 1 else 0.0
        ),
        "tps_ref_softmax_entropy": float(entropy),
    }


def build_fold(fold: int, device: str, batch_size: int) -> dict:
    protein_path = OUT / "broad_protein_tps.npy"
    if not protein_path.exists():
        raise FileNotFoundError(
            "run stage=protein-cache before building TPS fold responses"
        )
    protein_z = np.load(protein_path, mmap_mode="r")
    cache = np.load(V4 / f"fold{fold}" / "cache.npy", allow_pickle=True)
    reaction_frame = pd.read_csv(REACTIONS, dtype=str).fillna("")
    smiles = dict(
        zip(
            reaction_frame["reaction_id"].astype(str),
            reaction_frame["reaction_smiles"].astype(str),
        )
    )
    rf_entries = pd.read_csv(REACTION_FEATURE_ROOT / "entries.csv", dtype=str).fillna("")
    rf_entries["row"] = pd.to_numeric(rf_entries["row"]).astype(int)
    rf_entries = rf_entries.sort_values("row", kind="stable")
    rf_index = dict(zip(rf_entries["reaction_id"].astype(str), rf_entries["row"].astype(int)))
    rf_matrix = np.load(REACTION_FEATURE_ROOT / "reaction_feature_matrix.npy", mmap_mode="r")
    if rf_matrix.ndim != 2 or rf_matrix.shape[1] != 2115:
        raise RuntimeError(f"unexpected Broad reaction feature shape: {rf_matrix.shape}")
    reference, reference_meta = load_reference()
    projector = TPSAdaptedCoordinateProjector(
        device=device,
        batch_size=batch_size,
    )
    query_ids = [str(record["query_id"]) for record in cache]
    missing = [q for q in query_ids if q not in rf_index]
    if missing:
        raise RuntimeError(f"missing cached TPS reaction inputs: {missing[:10]}")
    query_x = np.asarray(
        rf_matrix[[rf_index[q] for q in query_ids]], dtype=np.float32
    )
    query_z = projector.project_reaction_features(query_x).astype(np.float32)

    query_rows = []
    score_blocks = []
    failed = []
    for i, (record, q) in enumerate(zip(cache, query_z, strict=True)):
        query_id = str(record["query_id"])
        rxn = smiles.get(query_id, "")
        candidate_rows = np.asarray(record["candidate_rows"], dtype=np.int64)
        p = np.asarray(protein_z[candidate_rows], dtype=np.float32)
        score = (p @ q).astype(np.float32)
        features = domain_features(q, reference)
        query_rows.append(
            {
                "query_id": query_id,
                "reaction_smiles": rxn,
                **features,
                "candidate_count": int(len(candidate_rows)),
                "tps_score_mean": float(score.mean()),
                "tps_score_std": float(score.std()),
                "tps_score_top1_margin": float(
                    np.sort(score)[-1] - np.sort(score)[-2]
                    if len(score) > 1
                    else 0.0
                ),
                "tps_score_top20_mean": float(
                    np.sort(score)[-min(20, len(score)) :].mean()
                ),
            }
        )
        score_blocks.append(
            {
                "query_id": query_id,
                "candidate_rows": candidate_rows.astype(np.int32),
                "tps_score": score,
                "positive_rows": np.asarray(
                    record["positive_rows"], dtype=np.int32
                ),
            }
        )
        if (i + 1) % 100 == 0:
            print(
                f"fold={fold} queries={i+1}/{len(cache)}",
                flush=True,
            )

    out = OUT / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    qframe = pd.DataFrame(query_rows)
    qframe.to_csv(out / "query_features.csv", index=False)
    np.save(
        out / "score_cache.npy",
        np.asarray(score_blocks, dtype=object),
        allow_pickle=True,
    )
    pd.DataFrame(failed).to_csv(out / "failures.csv", index=False)
    result = {
        "schema": "fibre-tps-specialist-response-fold-v2",
        "fold": fold,
        "status": "completed",
        "queries_input": int(len(cache)),
        "queries_scored": int(len(qframe)),
        "failures": int(len(failed)),
        "candidate_depth": TOPK,
        "reference_reactions": int(len(reference)),
        "reference_seen_fraction": float(
            reference_meta["reaction_seen"].astype(str).str.lower().eq("true").mean()
            if "reaction_seen" in reference_meta
            else 0.0
        ),
        "reaction_input_source": "general_merged/drfp_categorical_v1 frozen 2115-D cache",
        "labels_used_for_response_features": False,
        "positive_rows_retained_for_later_train_val_utility_only": True,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result



def build_production(device: str, batch_size: int) -> dict:
    """Build label-free TPS query coordinates for every registered reaction.

    This is a serving asset only. It reuses the frozen TPS projector and
    reference set, but does not use benchmark labels, wet-lab outcomes, or
    gate fitting. The validation-frozen semantic threshold and authority
    regressor remain unchanged.
    """
    reaction_frame = pd.read_csv(REACTIONS, dtype=str).fillna("")
    query_ids = reaction_frame["reaction_id"].astype(str).tolist()
    if len(query_ids) != len(set(query_ids)):
        raise RuntimeError("registered reaction ids must be unique")

    rf_entries = pd.read_csv(REACTION_FEATURE_ROOT / "entries.csv", dtype=str).fillna("")
    rf_entries["row"] = pd.to_numeric(rf_entries["row"]).astype(int)
    rf_entries = rf_entries.sort_values("row", kind="stable")
    rf_index = dict(zip(rf_entries["reaction_id"].astype(str), rf_entries["row"].astype(int)))
    missing = [q for q in query_ids if q not in rf_index]
    if missing:
        raise RuntimeError(f"missing registered TPS reaction inputs: {missing[:10]}")

    rf_matrix = np.load(REACTION_FEATURE_ROOT / "reaction_feature_matrix.npy", mmap_mode="r")
    x = np.asarray(rf_matrix[[rf_index[q] for q in query_ids]], dtype=np.float32)
    projector = TPSAdaptedCoordinateProjector(device=device, batch_size=batch_size)
    z = projector.project_reaction_features(x).astype(np.float32)
    reference, _ = load_reference()
    rows = [
        {"query_id": qid, **domain_features(q, reference)}
        for qid, q in zip(query_ids, z, strict=True)
    ]

    out = OUT / "production"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "reaction_tps.npy", z)
    pd.DataFrame(rows).to_csv(out / "query_features.csv", index=False)
    result = {
        "schema": "fibre-tps-specialist-production-response-v1",
        "status": "completed",
        "queries": len(query_ids),
        "output_dim": int(z.shape[1]),
        "reference_reactions": int(len(reference)),
        "labels_used": False,
        "wetlab_outcomes_used": False,
        "gate_refit": False,
        "reaction_input_source": "general_merged/drfp_categorical_v1 frozen 2115-D cache",
        "note": "serving coverage for all registered reactions; gate calibration remains frozen",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def build_outer(device: str, batch_size: int) -> dict:
    query_ids = set()
    for d in sorted(BROAD_BENCH.iterdir()):
        path = d / "test_pairs.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["reaction_id"], dtype=str).fillna("")
        query_ids.update(frame["reaction_id"].astype(str))
    query_ids = sorted(query_ids)

    rf_entries = pd.read_csv(REACTION_FEATURE_ROOT / "entries.csv", dtype=str).fillna("")
    rf_entries["row"] = pd.to_numeric(rf_entries["row"]).astype(int)
    rf_entries = rf_entries.sort_values("row", kind="stable")
    rf_index = dict(zip(rf_entries["reaction_id"].astype(str), rf_entries["row"].astype(int)))
    missing = [q for q in query_ids if q not in rf_index]
    if missing:
        raise RuntimeError(f"missing cached TPS reaction inputs: {missing[:10]}")
    rf_matrix = np.load(REACTION_FEATURE_ROOT / "reaction_feature_matrix.npy", mmap_mode="r")
    x = np.asarray(rf_matrix[[rf_index[q] for q in query_ids]], dtype=np.float32)
    projector = TPSAdaptedCoordinateProjector(device=device, batch_size=batch_size)
    z = projector.project_reaction_features(x).astype(np.float32)
    reference, _ = load_reference()
    rows = []
    for qid, q in zip(query_ids, z, strict=True):
        rows.append({"query_id": qid, **domain_features(q, reference)})
    out = OUT / "full_outer"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "reaction_tps.npy", z)
    pd.DataFrame(rows).to_csv(out / "query_features.csv", index=False)
    result = {
        "schema": "fibre-tps-specialist-full-outer-response-v2",
        "status": "completed",
        "queries": len(query_ids),
        "output_dim": int(z.shape[1]),
        "reference_reactions": int(len(reference)),
        "labels_used": False,
        "pair_scores_materialized": False,
        "note": "pair scores are computed only for each query's frozen Broad Top-1000 during outer evaluation",
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=("protein-cache", "fold", "outer", "production"),
    )
    ap.add_argument("--fold", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=2048)
    args = ap.parse_args()

    if args.stage == "protein-cache":
        build_protein_cache(args.device, args.batch_size)
    elif args.stage == "outer":
        build_outer(args.device, args.batch_size)
    elif args.stage == "production":
        build_production(args.device, args.batch_size)
    else:
        if args.fold not in (0, 1, 2):
            raise ValueError("--fold must be 0, 1, or 2")
        build_fold(args.fold, args.device, args.batch_size)


if __name__ == "__main__":
    main()
