from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.runtime.cli import (
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.fibre.runtime.enzyme_ranker import (
    build_features,
    full_order,
    lexical_rank,
)
from projects.active.fibre.runtime.reaction_to_enzyme import _structural_features
from projects.active.fibre.runtime.ranking_metrics import (
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)


ASSET_ROOT = ROOT.parent / "igem2026"
PRIMARY = ASSET_ROOT / "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1"
SECONDARY = ASSET_ROOT / "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1"
PRIMARY_PROTEINS = ASSET_ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
SECONDARY_PROTEINS = ASSET_ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
REACTIONS = (
    ASSET_ROOT
    / "data/catalyst_candidate_universes/general_merged/reaction_features/"
    "drfp_categorical_rdkitplus_center_v1"
)
CLIP_PROTEINS = ASSET_ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
CLIP_REACTIONS = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/"
    "clipzyme_embeddings_gpu_v1"
)
QUERY_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/"
    "mutual_cold_query_ids.txt"
)
PAIR_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/"
    "mutual_cold_test_pairs.csv"
)
SUPPORT_FILE = (
    ASSET_ROOT
    / "results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt"
)
DIFFICULTY_FILE = (
    ASSET_ROOT
    / "results/rhea128_to141_external_v2/posthoc_difficulty/"
    "rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv"
)
STRUCTURAL_BIME_METRICS = (
    ASSET_ROOT
    / "results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/query_metrics.csv"
)
INTERNAL_RESULT = ROOT / "results/fibre_evidence_fusion_structure_v1/result.json"
PROTOCOL_LOCK = (
    ROOT
    / "reproducibility/bime_rank/records/"
    "FIBRE_ANCHORED_STRUCTURE_EVIDENCE_V1_PROTOCOL_LOCK.json"
)
OUT = ROOT / "results/fibre_evidence_fusion_structure_v1_strict_temporal"

STRUCTURE_STRENGTH = 0.4919222801584656
ROUTER_THRESHOLD = 0.9
POOL_K = 100
PREFIX_K = 100
BASE_FEATURE_COUNT = 24
CLIP_Z_COLUMN = BASE_FEATURE_COUNT + 1
CLIP_SUPPORTED_COLUMN = BASE_FEATURE_COUNT + 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encode_chunks(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    side: str,
    device: torch.device,
    chunk_size: int = 8192,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(values), chunk_size):
            batch = torch.as_tensor(
                values[start : start + chunk_size],
                dtype=torch.float32,
                device=device,
            )
            if side == "protein":
                encoded = model.encode_proteins(batch)
            elif side == "reaction":
                encoded = model.encode_reactions(batch)
            else:
                raise ValueError(side)
            pieces.append(encoded.detach())
    return torch.cat(pieces, dim=0)


def projected_positive_ranks(
    order: np.ndarray,
    *,
    protein_ids: list[str],
    support_mask: np.ndarray,
    support_index: dict[str, int],
    positives: set[str],
) -> np.ndarray:
    projected = order[support_mask[order]]
    inverse = np.empty(len(support_index), dtype=np.int32)
    for rank, full_row in enumerate(projected, 1):
        inverse[support_index[protein_ids[int(full_row)]]] = rank
    return np.asarray(
        [inverse[support_index[value]] for value in positives],
        dtype=np.int32,
    )


def main() -> None:
    if not PROTOCOL_LOCK.exists():
        raise RuntimeError("protocol lock must exist before strict-temporal evaluation")
    lock = json.loads(PROTOCOL_LOCK.read_text())
    if float(lock["frozen_structure_strength"]) != STRUCTURE_STRENGTH:
        raise RuntimeError("structure coefficient differs from protocol lock")
    if sha256(INTERNAL_RESULT) != lock["strength_source"]["sha256"]:
        raise RuntimeError("internal OOF result drifted after protocol lock")
    if sha256(Path(__file__)) != lock["implementation"]["evaluator_sha256"]:
        raise RuntimeError("strict evaluator drifted after protocol lock")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_features, protein_ids = load_protein_library(PRIMARY_PROTEINS)
    secondary_features, secondary_ids = load_protein_library(SECONDARY_PROTEINS)
    if protein_ids != secondary_ids:
        raise RuntimeError("primary/secondary candidate order differs")
    primary_schema = load_feature_schema(PRIMARY)
    secondary_schema = load_feature_schema(SECONDARY)
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        REACTIONS,
        primary_schema,
    )
    reaction_features_2, reaction_ids_2 = load_registered_reaction_feature_library(
        REACTIONS,
        secondary_schema,
    )
    if reaction_ids != reaction_ids_2 or not np.array_equal(
        reaction_features,
        reaction_features_2,
    ):
        raise RuntimeError("primary/secondary reaction feature libraries differ")

    primary_models = load_models(PRIMARY / "models", "production", device)
    secondary_models = load_models(SECONDARY / "models", "production", device)
    if len(primary_models) != 1 or len(secondary_models) != 1:
        raise RuntimeError("expected one frozen production model per source")
    primary_model = primary_models[0]
    secondary_model = secondary_models[0]

    print("encoding frozen candidate/query towers", flush=True)
    primary_protein = encode_chunks(
        primary_model,
        primary_features,
        side="protein",
        device=device,
    )
    secondary_protein = encode_chunks(
        secondary_model,
        secondary_features,
        side="protein",
        device=device,
    )
    primary_reaction = encode_chunks(
        primary_model,
        reaction_features,
        side="reaction",
        device=device,
    )
    secondary_reaction = encode_chunks(
        secondary_model,
        reaction_features,
        side="reaction",
        device=device,
    )

    clip_entries = pd.read_csv(CLIP_PROTEINS / "entries.csv", dtype=str).fillna("")
    clip_rows = clip_entries["candidate_row"].astype(int).to_numpy(np.int32)
    clip_ids = clip_entries["protein_id"].astype(str).tolist()
    if [protein_ids[int(row)] for row in clip_rows] != clip_ids:
        raise RuntimeError("CLIPZyme protein asset order drifted")
    clip_protein = torch.as_tensor(
        np.array(
            np.load(CLIP_PROTEINS / "embeddings.npy", mmap_mode="r"),
            dtype=np.float32,
            copy=True,
        ),
        device=device,
    )
    clip_lookup = np.full(len(protein_ids), -1, dtype=np.int32)
    clip_lookup[clip_rows] = np.arange(len(clip_rows), dtype=np.int32)

    clip_reaction_entries = pd.read_csv(
        CLIP_REACTIONS / "entries.csv", dtype=str
    ).fillna("")
    clip_reaction_map = {
        str(reaction_id): int(row)
        for reaction_id, row, supported in clip_reaction_entries[
            ["reaction_id", "row", "clipzyme_supported"]
        ].itertuples(index=False)
        if str(supported).lower() == "true"
    }
    clip_reaction = np.load(CLIP_REACTIONS / "embeddings.npy", mmap_mode="r")

    query_ids = [line.strip() for line in QUERY_FILE.read_text().splitlines() if line.strip()]
    support = [line.strip() for line in SUPPORT_FILE.read_text().splitlines() if line.strip()]
    pairs = pd.read_csv(PAIR_FILE, dtype=str).fillna("")
    positives = (
        pairs.groupby("reaction_id")["protein_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    difficulty = pd.read_csv(DIFFICULTY_FILE, dtype={"reaction_id": str})
    similarity = dict(
        zip(
            difficulty["reaction_id"].astype(str),
            difficulty["max_train_drfp_tanimoto"].astype(float),
        )
    )
    reaction_index = {value: row for row, value in enumerate(reaction_ids)}
    protein_index = {value: row for row, value in enumerate(protein_ids)}
    support_rows = np.asarray([protein_index[value] for value in support], dtype=np.int32)
    support_mask = np.zeros(len(protein_ids), dtype=bool)
    support_mask[support_rows] = True
    support_index = {value: row for row, value in enumerate(support)}
    lex = lexical_rank(protein_ids)
    clip_lex = lex[clip_rows]

    baseline_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for start in range(0, len(query_ids), 16):
        local_queries = query_ids[start : start + 16]
        qrows = torch.as_tensor(
            [reaction_index[value] for value in local_queries],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            primary_score_batch = (
                primary_reaction[qrows] @ primary_protein.T
            ).float().cpu().numpy()
            secondary_score_batch = (
                secondary_reaction[qrows] @ secondary_protein.T
            ).float().cpu().numpy()

        for offset, query in enumerate(local_queries):
            if query not in clip_reaction_map:
                raise RuntimeError(f"strict query lacks CLIPZyme reaction support: {query}")
            primary_scores = primary_score_batch[offset].astype(np.float32, copy=False)
            secondary_scores = secondary_score_batch[offset].astype(np.float32, copy=False)
            primary_order, primary_inv = full_order(primary_scores, lex)
            secondary_order, secondary_inv = full_order(secondary_scores, lex)
            use_secondary = float(similarity[query]) < ROUTER_THRESHOLD
            fallback_order = secondary_order if use_secondary else primary_order

            clip_query = torch.as_tensor(
                np.array(
                    clip_reaction[clip_reaction_map[query]],
                    dtype=np.float32,
                    copy=True,
                ),
                device=device,
            )
            with torch.no_grad():
                clip_scores = (clip_protein @ clip_query).float().cpu().numpy()
            clip_order = np.lexsort((clip_lex, -clip_scores)).astype(np.int32)
            clip_inverse = np.empty(len(clip_order), dtype=np.int32)
            clip_inverse[clip_order] = np.arange(1, len(clip_order) + 1, dtype=np.int32)

            union = np.unique(
                np.concatenate(
                    [
                        primary_order[:POOL_K],
                        secondary_order[:POOL_K],
                        clip_rows[clip_order[:POOL_K]],
                    ]
                )
            ).astype(np.int32)
            base_x = build_features(
                primary_scores,
                secondary_scores,
                union,
                primary_inv,
                secondary_inv,
                use_secondary,
                float(similarity[query]),
            )
            structure_x = _structural_features(
                base_x,
                union,
                clip_scores,
                clip_inverse,
                clip_lookup,
                len(protein_ids),
            )
            core_z = base_x[:, 14].astype(np.float64, copy=False)
            clip_z = structure_x[:, CLIP_Z_COLUMN].astype(np.float64, copy=False)
            clip_supported = structure_x[:, CLIP_SUPPORTED_COLUMN] > 0.5
            fused = core_z + STRUCTURE_STRENGTH * np.where(
                clip_supported,
                clip_z,
                0.0,
            )
            local_order = np.lexsort((lex[union], -fused))
            promoted = union[local_order[: min(PREFIX_K, len(local_order))]]
            promoted_mask = np.zeros(len(protein_ids), dtype=bool)
            promoted_mask[promoted] = True
            candidate_order = np.concatenate(
                [promoted, fallback_order[~promoted_mask[fallback_order]]]
            ).astype(np.int32, copy=False)

            baseline_positive_ranks = projected_positive_ranks(
                fallback_order,
                protein_ids=protein_ids,
                support_mask=support_mask,
                support_index=support_index,
                positives=positives[query],
            )
            candidate_positive_ranks = projected_positive_ranks(
                candidate_order,
                protein_ids=protein_ids,
                support_mask=support_mask,
                support_index=support_index,
                positives=positives[query],
            )
            baseline_rows.append(
                {
                    "query_id": query,
                    **evaluate_full_candidate_ranks(
                        baseline_positive_ranks,
                        len(support),
                    ),
                }
            )
            candidate_rows.append(
                {
                    "query_id": query,
                    **evaluate_full_candidate_ranks(
                        candidate_positive_ranks,
                        len(support),
                    ),
                }
            )
            audit_rows.append(
                {
                    "query_id": query,
                    "fallback_secondary": bool(use_secondary),
                    "similarity": float(similarity[query]),
                    "union_size": int(len(union)),
                    "clip_supported_in_union": int(clip_supported.sum()),
                }
            )
        print(
            "strict structure evidence",
            min(start + 16, len(query_ids)),
            "/",
            len(query_ids),
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    baseline_frame = pd.DataFrame(baseline_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    baseline_frame.to_csv(OUT / "baseline_query_metrics.csv", index=False)
    candidate_frame.to_csv(OUT / "candidate_query_metrics.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(OUT / "audit.csv", index=False)

    baseline_summary = summarize_query_metrics(baseline_frame)
    candidate_summary = summarize_query_metrics(candidate_frame)
    keys = ("mrr", "map", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
    delta = {
        key: float(candidate_summary[key] - baseline_summary[key])
        for key in keys
    }

    structural_bime = None
    if STRUCTURAL_BIME_METRICS.exists():
        previous = pd.read_csv(STRUCTURAL_BIME_METRICS, dtype={"query_id": str})
        structural_bime = summarize_query_metrics(previous)

    result = {
        "schema": "fibre-anchored-structure-evidence-strict-temporal-v1",
        "status": "frozen_external_evaluation_once",
        "protocol": (
            "Rhea128->141 strict mutual double-cold R2E, 144 queries x "
            "166202 common candidates; coefficient fixed from internal OOF before reveal"
        ),
        "structure_strength": STRUCTURE_STRENGTH,
        "queries": len(query_ids),
        "candidate_count": len(support),
        "positive_pairs": len(pairs),
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "delta": delta,
        "previous_structural_bime": structural_bime,
        "protocol_lock": str(PROTOCOL_LOCK.relative_to(ROOT)),
        "protocol_lock_sha256": sha256(PROTOCOL_LOCK),
    }
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
