from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.runtime.cli import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.fibre.runtime.enzyme_ranker import (  # noqa: E402
    build_features,
    full_order,
    lexical_rank,
    load_ranker,
)
from projects.active.fibre.runtime.ranking_metrics import (  # noqa: E402
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from reproducibility.bime_rank.scripts.evaluate_broad_rhea_benchmark import (  # noqa: E402
    encode_chunks,
)


ASSET_ROOT = Path("/home/s241850073/igem2026")
OUT = ROOT / "results/fibre_anchored_evidence_r2e_strict_v1"
PROTOCOL = (
    ROOT
    / "reproducibility/bime_rank/records/FIBRE_ANCHORED_EVIDENCE_R2E_V1_PROTOCOL_LOCK.json"
)
PRIMARY = ASSET_ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
SECONDARY = ASSET_ROOT / "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1"
PRIMARY_PROTEIN = ASSET_ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
SECONDARY_PROTEIN = ASSET_ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
REACTION = (
    ASSET_ROOT
    / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"
)
BASE_RANKER = ASSET_ROOT / "results/r2e_lambdarank_fusion_v1/selected"
BASE_SHA = "86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e"
CLIP_PROTEIN = ASSET_ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
CLIP_REACTION = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
)
QUERY_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt"
)
PAIR_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv"
)
SUPPORT_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt"
)
DIFFICULTY = (
    ASSET_ROOT
    / "results/rhea128_to141_external_v2/posthoc_difficulty/"
    "rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv"
)
CURRENT_STRUCTURE_METRICS = (
    ASSET_ROOT
    / "results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/query_metrics.csv"
)


def clip_assets(candidate_ids: list[str], device: torch.device):
    pe = pd.read_csv(CLIP_PROTEIN / "entries.csv", dtype=str).fillna("")
    protein_matrix = np.load(CLIP_PROTEIN / "embeddings.npy", mmap_mode="r")
    candidate_rows = pe["candidate_row"].astype(int).to_numpy(np.int32)
    if [candidate_ids[int(row)] for row in candidate_rows] != pe["protein_id"].astype(str).tolist():
        raise RuntimeError("CLIPZyme protein asset alignment drifted")
    protein_tensor = torch.as_tensor(
        np.asarray(protein_matrix), dtype=torch.float32, device=device
    )

    re = pd.read_csv(CLIP_REACTION / "entries.csv", dtype=str).fillna("")
    reaction_matrix = np.load(CLIP_REACTION / "embeddings.npy", mmap_mode="r")
    supported = re[re["clipzyme_supported"].astype(str).str.lower().eq("true")]
    row_by_reaction = {
        str(reaction): int(row)
        for reaction, row in supported[["reaction_id", "row"]].itertuples(index=False)
    }
    lookup = np.full(len(candidate_ids), -1, dtype=np.int32)
    lookup[candidate_rows] = np.arange(len(candidate_rows), dtype=np.int32)
    return protein_tensor, candidate_rows, reaction_matrix, row_by_reaction, lookup


def query_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return ((values - values.mean()) / max(float(values.std()), 1e-6)).astype(np.float32)


def full_ranking_from_prefix(
    selected: np.ndarray,
    fallback_order: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    mask = np.zeros(candidate_count, dtype=bool)
    mask[selected] = True
    tail = fallback_order[~mask[fallback_order]]
    order = np.concatenate([selected, tail]).astype(np.int32, copy=False)
    if len(order) != candidate_count or len(np.unique(order)) != candidate_count:
        raise AssertionError("prefix+fallback did not construct a full permutation")
    return order


def main() -> None:
    lock = json.loads(PROTOCOL.read_text())
    strength = float(lock["evidence"]["final_nonnegative_strength"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    protein0, protein_ids = load_protein_library(PRIMARY_PROTEIN)
    protein1, protein_ids1 = load_protein_library(SECONDARY_PROTEIN)
    if protein_ids != protein_ids1:
        raise RuntimeError("primary/secondary candidate orders differ")
    schema0 = load_feature_schema(PRIMARY)
    schema1 = load_feature_schema(SECONDARY)
    reaction0, reaction_ids = load_registered_reaction_feature_library(REACTION, schema0)
    reaction1, reaction_ids1 = load_registered_reaction_feature_library(REACTION, schema1)
    if reaction_ids != reaction_ids1 or not np.array_equal(reaction0, reaction1):
        raise RuntimeError("primary/secondary reaction features differ")
    models0 = load_models(PRIMARY / "models", "production", device)
    models1 = load_models(SECONDARY / "models", "production", device)
    if len(models0) != 1 or len(models1) != 1:
        raise RuntimeError("strict confirmation expects one production checkpoint per source")

    print("encoding frozen core assets", flush=True)
    p0 = encode_chunks(models0[0], protein0, kind="protein", device=device, chunk_size=8192)
    p1 = encode_chunks(models1[0], protein1, kind="protein", device=device, chunk_size=8192)
    r0 = encode_chunks(models0[0], reaction0, kind="reaction", device=device, chunk_size=8192)
    r1 = encode_chunks(models1[0], reaction1, kind="reaction", device=device, chunk_size=8192)

    clip_p, clip_rows, clip_r, clip_rmap, clip_lookup = clip_assets(protein_ids, device)
    clip_lex = lexical_rank(protein_ids)[clip_rows]
    base_ranker, config = load_ranker(str(BASE_RANKER), BASE_SHA)
    if int(config["pool_k"]) != 100 or int(config["prefix_k"]) != 100:
        raise RuntimeError("unexpected frozen base ranker pool/prefix")

    query_ids = [x.strip() for x in QUERY_FILE.read_text().splitlines() if x.strip()]
    support = [x.strip() for x in SUPPORT_FILE.read_text().splitlines() if x.strip()]
    if len(query_ids) != 144 or len(support) != 166202:
        raise RuntimeError("strict protocol support drifted")
    pairs = pd.read_csv(PAIR_FILE, dtype=str).fillna("")
    positives = (
        pairs.groupby("reaction_id")["protein_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    difficulty = pd.read_csv(DIFFICULTY, dtype={"reaction_id": str})
    similarity = dict(
        zip(
            difficulty["reaction_id"].astype(str),
            difficulty["max_train_drfp_tanimoto"].astype(float),
        )
    )
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}
    full_index = {value: i for i, value in enumerate(protein_ids)}
    support_rows = np.asarray([full_index[x] for x in support], dtype=np.int32)
    support_mask = np.zeros(len(protein_ids), dtype=bool)
    support_mask[support_rows] = True
    support_index = {value: i for i, value in enumerate(support)}
    lex = lexical_rank(protein_ids)

    base_rows: list[dict[str, object]] = []
    fused_rows: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []

    for start in range(0, len(query_ids), 16):
        local_queries = query_ids[start : start + 16]
        qrows = torch.as_tensor(
            [reaction_index[q] for q in local_queries],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            primary_scores = (r0[qrows] @ p0.T).float().cpu().numpy()
            secondary_scores = (r1[qrows] @ p1.T).float().cpu().numpy()
            qclip = torch.as_tensor(
                np.stack(
                    [
                        np.asarray(clip_r[clip_rmap[q]], dtype=np.float32)
                        for q in local_queries
                    ]
                ),
                device=device,
            )
            clip_scores_batch = (qclip @ clip_p.T).float().cpu().numpy()

        for local, query in enumerate(local_queries):
            primary = primary_scores[local].astype(np.float32, copy=False)
            secondary = secondary_scores[local].astype(np.float32, copy=False)
            p_order, p_inv = full_order(primary, lex)
            s_order, s_inv = full_order(secondary, lex)
            fallback_secondary = float(similarity[query]) < 0.9
            fallback_order = s_order if fallback_secondary else p_order

            clip_scores = clip_scores_batch[local].astype(np.float32, copy=False)
            clip_order = np.lexsort((clip_lex, -clip_scores)).astype(np.int32)
            clip_inv = np.empty(len(clip_order), dtype=np.int32)
            clip_inv[clip_order] = np.arange(1, len(clip_order) + 1, dtype=np.int32)
            clip_top = clip_rows[clip_order[:100]]
            union = np.unique(
                np.concatenate([p_order[:100], s_order[:100], clip_top])
            ).astype(np.int32)

            base_features = build_features(
                primary,
                secondary,
                union,
                p_inv,
                s_inv,
                fallback_secondary,
                float(similarity[query]),
            )
            base_prediction = base_ranker.predict(xgb.DMatrix(base_features))
            core_z = query_zscore(base_prediction)

            local_clip = clip_lookup[union]
            clip_ok = local_clip >= 0
            clip_z = np.zeros(len(union), dtype=np.float32)
            clip_mean = float(clip_scores.mean())
            clip_std = max(float(clip_scores.std()), 1e-6)
            clip_z[clip_ok] = (
                clip_scores[local_clip[clip_ok]] - clip_mean
            ) / clip_std

            base_local_order = np.lexsort((lex[union], -core_z))
            fused_score = core_z + strength * clip_z
            fused_local_order = np.lexsort((lex[union], -fused_score))
            base_selected = union[base_local_order[: min(100, len(union))]]
            fused_selected = union[fused_local_order[: min(100, len(union))]]
            base_full = full_ranking_from_prefix(
                base_selected, fallback_order, len(protein_ids)
            )
            fused_full = full_ranking_from_prefix(
                fused_selected, fallback_order, len(protein_ids)
            )

            def metrics(full_order_values: np.ndarray) -> dict[str, object]:
                projected = full_order_values[support_mask[full_order_values]]
                inverse = np.empty(len(support), dtype=np.int32)
                for rank, full_row in enumerate(projected, 1):
                    inverse[support_index[protein_ids[int(full_row)]]] = rank
                positive_ranks = np.asarray(
                    [inverse[support_index[p]] for p in positives[query]],
                    dtype=np.int32,
                )
                return evaluate_full_candidate_ranks(positive_ranks, len(support))

            base_rows.append({"query_id": query, **metrics(base_full)})
            fused_rows.append({"query_id": query, **metrics(fused_full)})
            audit.append(
                {
                    "query_id": query,
                    "similarity": float(similarity[query]),
                    "fallback_secondary": bool(fallback_secondary),
                    "union_size": int(len(union)),
                    "structure_strength": strength,
                }
            )
        print("done", min(start + 16, len(query_ids)), "/", len(query_ids), flush=True)

    base_frame = pd.DataFrame(base_rows)
    fused_frame = pd.DataFrame(fused_rows)
    base_summary = summarize_query_metrics(base_frame)
    fused_summary = summarize_query_metrics(fused_frame)

    payload = {
        "schema": "fibre-anchored-scientific-evidence-r2e-v1-strict-result",
        "status": "single_frozen_confirmation_complete",
        "protocol": "Rhea release128->141 strict double-cold, 144 x 166202 common support",
        "structure_strength": strength,
        "external_metrics_used_for_selection": False,
        "core": {
            key: base_summary[key]
            for key in (
                "mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20",
                "hit_at_50", "median_best_positive_rank",
            )
        },
        "core_plus_structure": {
            key: fused_summary[key]
            for key in (
                "mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20",
                "hit_at_50", "median_best_positive_rank",
            )
        },
        "delta": {
            key: float(fused_summary[key]) - float(base_summary[key])
            for key in (
                "mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50",
            )
        },
    }
    if CURRENT_STRUCTURE_METRICS.exists():
        current = summarize_query_metrics(
            pd.read_csv(CURRENT_STRUCTURE_METRICS, dtype={"query_id": str})
        )
        payload["current_bime_structure_reference"] = {
            key: current[key]
            for key in (
                "mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20",
                "hit_at_50", "median_best_positive_rank",
            )
        }

    OUT.mkdir(parents=True, exist_ok=True)
    base_frame.to_csv(OUT / "core_query_metrics.csv", index=False)
    fused_frame.to_csv(OUT / "core_plus_structure_query_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "audit.csv", index=False)
    (OUT / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
