from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reproducibility/research_release_manifest.json"
GITHUB_BLOB_LIMIT = 100_000_000

MODEL_REPRODUCTION_SUPPORT_ROOTS = [
    "results/terpene_horizyn_reaction_feature_distillation",
    # Project-owned ancestor used to initialize the current MARTS fallback deployments.
    "results/terpene_production_models/drfp_categorical",
]

CURRENT_MODEL_ROOTS = [
    "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1",
    "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1",
    "results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1",
    "results/catalyst_clean_mainline_v1/e2r_anchored_lambdamart_v3",
    "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected",
    "results/bime_rank_unified_v1/r2e_seed_context_v1/selected",
    "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected",
    "results/bime_rank_unified_v1/e2r_seed_context_v1/selected",
]

# These are model-training caches / derived feature matrices, not learned weights.
CURRENT_MODEL_EXCLUDES = {
    "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1/reaction_feature_matrix.npy",
    "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1/reaction_feature_matrix.npy",
}

CURRENT_RUNTIME_SUPPORT_FILES = [
    # rank_open_world.py reads this by default for the production TPS Top-20 CAGE rescue.
    # Omitting it from a clean clone silently changes the live ranking path.
    "results/terpene_cage_screen/all_rhea_gate/all_pair_scores.csv",
]

CURRENT_MODEL_TRAINING_INPUT_FILES = [
    # Default split/group authorities consumed by train_marts_adapted_production.py.
    "data/terpene_sequence_clusters/clusters_id50.csv",
    "data/terpene_cold_splits/reaction_cluster_folds.csv",
]

# Files explicitly referenced by the current production route as evidence. They are
# direct release inputs even when they live under deny-by-default results/.
PRODUCTION_ROUTE_EVIDENCE_FILES = [
    "projects/active/terpene_screening/CATALYST_FAST_R2E_SIMILARITY_ROUTER_V1.json",
    "projects/active/terpene_screening/CATALYST_R2E_LAMBDARANK_FUSION_V1_CONFIRMATION_RESULT.json",
    "projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION_RESULT.json",
    "results/bime_rank_unified_v1/r2e_seed_context_v1/development_result.json",
    "results/bime_rank_unified_v1/e2r_seed_context_v1/development_result.json",
]

# The legacy terpene runtime manifest remains a full-server compatibility/provenance
# contract. Assets explicitly demoted by the audited Git-index-only policy remain
# hash-verifiable on a provisioned development server but are not vendored in the
# current scientific release. Keep the exclusion source machine-readable rather than
# duplicating path lists in code.
RUNTIME_DEMOTIONS = ROOT / "reproducibility/bime_rank/historical_runtime_asset_demotions.json"


def historical_runtime_direct_excludes(runtime_files: set[str]) -> set[str]:
    if not RUNTIME_DEMOTIONS.is_file():
        raise FileNotFoundError(RUNTIME_DEMOTIONS.relative_to(ROOT))
    payload = json.loads(RUNTIME_DEMOTIONS.read_text(encoding="utf-8"))
    if payload.get("category") != "historical_runtime_asset_demotions":
        raise RuntimeError("historical runtime asset demotion category drift")
    records = payload.get("records", [])
    if int(payload.get("count", -1)) != len(records):
        raise RuntimeError("historical runtime asset demotion count mismatch")
    paths = [str(record.get("path", "")) for record in records]
    if any(not path for path in paths):
        raise RuntimeError("historical runtime asset demotion record missing path")
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate historical runtime asset demotion path")
    missing = sorted(set(paths) - runtime_files)
    if missing:
        raise RuntimeError(f"runtime-demoted assets absent from legacy runtime manifest: {missing[:5]}")
    return set(paths)

DATABASE_RELEASE_FILES = [
    "data/catalyst_candidate_universes/general_merged/manifest.json",
    "data/catalyst_candidate_universes/general_merged/summary.json",
    "data/catalyst_candidate_universes/general_merged/sequence_version_conflicts.csv",
    "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv",
    "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
    "data/catalyst_candidate_universes/general_merged/associations.csv",
    "data/catalyst_candidate_universes/general_merged/reactions.csv",
    "data/catalyst_candidate_universes/general_merged/proteins/entries.csv",
    "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv",
    "reproducibility/bime_rank/rxnmapper_general_merged_v1.json",
    "data/terpene_embeddings/uniprot_tps_domain_rescue_esmc600m/entries.csv",
    "data/terpene_uniprot_expansion/uniprot_tps_domain_only_rescue_candidates.tsv",
    "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv",
    "data/external/enzgfm_current/general_merged_650m_mean_v1/entries.csv",
    "data/external/enzgfm_current/general_merged_650m_mean_v1/manifest.json",
    "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/entries.csv",
    "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/manifest.json",
    "results/clipzyme_native_extension_v1/r2e_strict650_candidate_embeddings_v1/new_embeddings.npy",
    "results/clipzyme_native_extension_v1/r2e_strict650_candidate_embeddings_v1/new_entries.csv",
    "results/clipzyme_native_extension_v1/r2e_strict650_candidate_embeddings_v1/manifest.json",
    "results/clipzyme_native_extension_v1/r2e_strict650_candidate_embeddings_v1/support_manifest.json",
    "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/embeddings.npy",
    "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/entries.csv",
    "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/unsupported_reactions.csv",
    "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/manifest.json",
    "results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1/entries.csv",
    "results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1/manifest.json",
    "results/bime_rank_unified_v1/clipzyme_e2r_pdb_extension_v1/embeddings.npy",
    "results/bime_rank_unified_v1/clipzyme_e2r_pdb_extension_v1/entries.csv",
    "results/bime_rank_unified_v1/clipzyme_e2r_pdb_extension_v1/manifest.json",
]

EVALUATION_SUPPORT_ASSETS = [
    {"path": "results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv", "role": "frozen_upstream_rhea_v2_positive_support"},
    {"path": "results/rhea128_to141_external_v2/posthoc_difficulty/rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv", "role": "label_independent_r2e_router_similarity_support"},
    {"path": "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt", "role": "clipzyme_r2e_fair_query_ids"},
    {"path": "results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv", "role": "clipzyme_r2e_fair_positive_pairs"},
    {"path": "results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt", "role": "clipzyme_r2e_common_candidate_support"},
    {"path": "results/clipzyme_native_extension_v1/r2e_strict650_current_system_fair_v1/query_metrics.csv", "role": "clipzyme_r2e_previous_bime_per_query_comparator"},
    {"path": "results/clipzyme_native_extension_v1/r2e_strict650_clipzyme_fair_v1/clipzyme_query_metrics.csv", "role": "clipzyme_r2e_official_per_query_comparator"},
    {"path": "results/clipzyme_native_extension_v1/e2r_strict650_current_system_vs_clipzyme_v1/catalyst_current_query_metrics.csv", "role": "clipzyme_e2r_previous_bime_per_query_comparator"},
    {"path": "results/clipzyme_native_extension_v1/e2r_strict650_mutual_cold_10131_v2_lexical/official_clipzyme_query_metrics.csv", "role": "clipzyme_e2r_official_per_query_comparator"},
    {"path": "results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1/embeddings.npy", "role": "clipzyme_e2r_fixed_query_embeddings"},
    {"path": "results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1/entries.csv", "role": "clipzyme_e2r_query_embedding_index"},
    {"path": "results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1/inputs.csv", "role": "clipzyme_e2r_query_embedding_inputs"},
    {"path": "results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1/manifest.json", "role": "clipzyme_e2r_query_embedding_manifest"},
    {"path": "results/clipzyme_native_extension_v1/strict650_e2r_query_embeddings_v1/structure_manifest.csv", "role": "clipzyme_e2r_query_structure_manifest"},
]

AGGREGATE_SUPPORT_ASSETS = [
    {"path": "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json", "role": "expert_admission_r2e_clipzyme_internal"},
    {"path": "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selection_result.json", "role": "expert_admission_e2r_clipzyme_internal"},
    {"path": "results/bime_rank_unified_v1/r2e_homology_context_v1/development_result.json", "role": "expert_admission_homology_internal"},
    {"path": "results/bime_rank_unified_v1/r2e_homology_context_retention_v1/summary.json", "role": "expert_admission_homology_external_retention"},
    {"path": "results/bime_rank_unified_v1/r2e_reciprocal_consistency_v1/development_result.json", "role": "expert_admission_reciprocal_internal"},
    {"path": "results/bime_rank_unified_v1/r2e_reciprocal_external_confirmation_v1/summary.json", "role": "expert_admission_reciprocal_external_retention"},
    {"path": "results/bime_rank_unified_v1/tps_cage_top20_expert_v1/development_result.json", "role": "expert_admission_enzymecage_internal_oof"},
    {"path": "results/bime_rank_unified_v1/tps_cage_top20_expert_v1/prepare_summary.json", "role": "expert_admission_enzymecage_support_counts"},
    {"path": "results/bime_rank_unified_v1/e2r_runtime_smoke_v2/summary.json", "role": "expert_admission_e2r_runtime_smoke"},
    {"path": "results/bime_rank_unified_v1/r2e_runtime_smoke_v2/summary.json", "role": "expert_admission_r2e_runtime_smoke"},
    {"path": "results/bime_rank_unified_v1/promotion_audit_v1/manifest_diff.json", "role": "expert_admission_first_promotion_manifest_audit"},
    {"path": "results/bime_rank_unified_v1/promotion_audit_v1/e2r_runtime_retention.json", "role": "expert_admission_first_promotion_runtime_retention"},
    {"path": "results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_manifest_equivalence.json", "role": "expert_admission_first_promotion_manifest_equivalence"},
    {"path": "results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_smoke/summary.json", "role": "expert_admission_first_promotion_smoke"},
    {"path": "results/bime_rank_unified_v1/promotion_audit_v1/promotion_gate.json", "role": "expert_admission_first_promotion_gate"},
    {"path": "results/bime_rank_unified_v1/cost_aware_shortlist_retention_v1/summary.json", "role": "cost_aware_internal_shortlist_retention"},
    {"path": "results/requested_r2e20_bime_v2_20260906/summary.json", "role": "cost_aware_wetlab_candidate_universe_summary"},
    {"path": "results/requested_r2e20_bime_v2_20260906/stage2_summary.json", "role": "cost_aware_wetlab_stage2_summary"},
    {"path": "results/requested_r2e20_bime_v2_20260906/enzgfm_stage2_530/manifest.json", "role": "cost_aware_wetlab_enzgfm_stage2_manifest"},
    {"path": "reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json", "role": "cost_aware_verified_hardware_specific_timing"},
]

REACTION_FEATURE_METADATA_DIRS = [
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1",
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1",
    "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1",
]

REBUILDABLE_ASSETS = [
    {
        "path": "data/terpene_horizyn_adapter_v2/train_standardized_reactions.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 6315222,
        "expected_sha256": "6b9b52ddaddf48294dd691433a69470732e80b7e9cba38ae045c5e2d0fb360dc",
        "builder": "projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "inputs": ["external/horizyn/data/sota/train_rxns.csv", "external/horizyn/configs/sota.yaml"],
    },
    {
        "path": "data/terpene_horizyn_adapter_v2/test_standardized_reactions.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 530754,
        "expected_sha256": "b8432651978dccceab85ed4e4bf66a135e62a2dd3a54ab6e1273d67882c01cad",
        "builder": "projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "inputs": ["external/horizyn/data/sota/test_rxns.csv", "external/horizyn/configs/sota.yaml"],
    },
    {
        "path": "data/terpene_horizyn_adapter_v2/marts_standardized_reactions.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 180518,
        "expected_sha256": "3f14e626e87e5a9f29ad1c912d1428bcea1f66f109a19903008822e93f877a3c",
        "builder": "projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "inputs": ["data/terpene_marts_adaptation/reaction_entities.csv", "external/horizyn/configs/sota.yaml"],
    },
    {
        "path": "results/terpene_horizyn_reaction_overlap.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 187167,
        "expected_sha256": "316e2e1022709bf134910e09b4e377ead33f901e68e56417bef86ce3e6932b09",
        "builder": "projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_horizyn_distillation_preprocessing.py",
        "inputs": [
            "data/terpene_horizyn_adapter_v2/train_standardized_reactions.csv",
            "data/terpene_horizyn_adapter_v2/test_standardized_reactions.csv",
            "data/terpene_horizyn_adapter_v2/marts_standardized_reactions.csv"
        ],
    },
    {
        "path": "results/terpene_reactzyme_transfer_audit_v1/reactzyme_expanded_pairs.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 118375762,
        "expected_sha256": "3eac711124e05fad84cd946734818e45b28cb23ca445d98984eff1b6e1d69cef",
        "builder": "projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "inputs": ["data/external/reactzyme/cleaned_uniprot_rhea.tsv"],
    },
    {
        "path": "data/external/reactzyme_transfer/unique_sequences.tsv",
        "kind": "derived_intermediate",
        "expected_bytes": 77395594,
        "expected_sha256": "f934bafbf6ad451ff4cd480b512eb1543790cc9ffeb188196eef87b506483c99",
        "builder": "projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "inputs": ["data/external/reactzyme/enzyme_smi_split.zip"],
    },
    {
        "path": "data/external/reactzyme_transfer/esmc600m_mean/entries.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 5952018,
        "expected_sha256": "ff65c94912bdc0fc93376189d0330e6dd7a0a391e211a40b2324840cfd9d4e80",
        "builder": "projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "inputs": ["data/external/reactzyme/enzyme_smi_split.zip"],
    },
    {
        "path": "data/external/reactzyme_transfer/global_clean_v2/reaction_overlap_audit.csv",
        "kind": "derived_intermediate",
        "expected_bytes": 2926793,
        "expected_sha256": "78f2a3a8a8bb64b4d7277f996de2c2faf22dc1619bf78aaa95547f0382751ce1",
        "builder": "projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_reactzyme_transfer_assets.py",
        "inputs": [
            "data/external/reactzyme/rhea_molecules.tsv",
            "results/terpene_reactzyme_transfer_audit_v1/reactzyme_expanded_pairs.csv",
            "data/terpene/enzyme_terpene_synthase.tsv"
        ],
    },
    {
        "path": "data/external/reactzyme_transfer/esmc600m_mean/embeddings.npy",
        "kind": "derived_feature_matrix",
        "expected_shape": [178327, 1152],
        "expected_bytes": 821730944,
        "expected_sha256": "d7ecb9c4de7e4b0d31517656fdc122965372188d44c2fd783e0484b5521e4c15",
        "builder": "projects/active/terpene_screening/extract_esmc_embeddings.py",
        "command": ".venv/bin/python projects/active/terpene_screening/extract_esmc_embeddings.py --input data/external/reactzyme_transfer/unique_sequences.tsv --input-sep '\\t' --entry-column Entry --sequence-column Sequence --output-dir data/external/reactzyme_transfer/esmc600m_mean --model esmc_600m --max-batch-tokens 8192 --max-batch-size 32",
        "inputs": ["data/external/reactzyme_transfer/unique_sequences.tsv"],
        "model": "EvolutionaryScale/esmc-600m-2024-12",
        "observed_huggingface_revision": "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c",
        "package": "esm==3.1.1",
    },
    {
        "path": "data/catalyst_candidate_universes/general_merged/proteins/embeddings.npy",
        "kind": "derived_feature_matrix",
        "expected_shape": [185918, 1152],
        "builder": "projects/active/terpene_screening/extract_esmc_embeddings.py",
        "command": ".venv/bin/python projects/active/terpene_screening/extract_esmc_embeddings.py --input data/catalyst_candidate_universes/general_merged/protein_sequences.tsv --input-sep '\\t' --entry-column protein_id --sequence-column sequence --output-dir data/catalyst_candidate_universes/general_merged/proteins --model esmc_600m",
        "model": "EvolutionaryScale/esmc-600m-2024-12",
        "observed_huggingface_revision": "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c",
        "package": "esm==3.1.1",
    },
    {
        "path": "data/external/enzgfm_current/general_merged_650m_mean_v1/embeddings.npy",
        "kind": "derived_feature_matrix",
        "expected_shape": [185918, 2048],
        "builder": "projects/active/terpene_screening/build_enzgfm_protein_features.py",
        "merge_builder": "projects/active/terpene_screening/merge_protein_feature_libraries.py",
        "command": ".venv/bin/python projects/active/terpene_screening/build_enzgfm_protein_features.py --sequences data/catalyst_candidate_universes/general_merged/protein_sequences.tsv --all-proteins --model-dir external_models/enzgfm/EnzGFM_650M --reference-root external/enzgfm_reference --output-dir data/external/enzgfm_current/general_merged_650m_mean_v1",
        "historical_exact_merge_contract": "data/external/enzgfm_current/general_merged_650m_mean_v1/manifest.json",
    },
    {
        "path": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1/reaction_feature_matrix.npy",
        "kind": "derived_feature_matrix",
        "expected_shape": [11081, 2115],
        "builder": "projects/active/terpene_screening/build_general_reaction_features.py",
        "command": ".venv/bin/python projects/active/terpene_screening/build_general_reaction_features.py",
    },
    {
        "path": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1/reaction_feature_matrix.npy",
        "kind": "derived_feature_matrix",
        "builder": "projects/active/terpene_screening/build_rdkitplus_augmented_reaction_features.py",
        "command": ".venv/bin/python projects/active/terpene_screening/build_rdkitplus_augmented_reaction_features.py",
        "metadata_manifest": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1/manifest.json",
    },
    {
        "path": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy",
        "kind": "derived_feature_matrix",
        "builder": "projects/active/terpene_screening/build_reaction_center_augmented_features.py",
        "command": ".venv/bin/python projects/active/terpene_screening/build_reaction_center_augmented_features.py",
        "metadata_manifest": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/manifest.json",
        "precomputed_inputs": ["data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"],
        "precompute_provenance": "reproducibility/bime_rank/rxnmapper_general_merged_v1.json",
    },
    {
        "path": "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/embeddings.npy",
        "kind": "derived_feature_matrix",
        "expected_shape": [166207, 1280],
        "builder": "reproducibility/bime_rank/source_snapshots/build_clipzyme_r2e_candidate_asset.py",
        "command": ".venv/bin/python reproducibility/bime_rank/source_snapshots/build_clipzyme_r2e_candidate_asset.py",
        "metadata_manifest": "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1/manifest.json",
        "note": "Combines the official released CLIPZyme screening asset with the tracked native-extension embeddings in frozen general-candidate order.",
    },
    {
        "path": "results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1/reaction_feature_matrix.npy",
        "kind": "training_cache",
        "replacement": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy",
        "required_for_inference": False,
    },
    {
        "path": "results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1/reaction_feature_matrix.npy",
        "kind": "training_cache",
        "replacement": "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/reaction_feature_matrix.npy",
        "required_for_inference": False,
    },
]

EXTERNAL_ASSETS = [
    {
        "name": "Horizyn SOTA train reactions",
        "target": "external/horizyn/data/sota/train_rxns.csv",
        "bytes": 3411244,
        "md5": "7b0335ac694e4afee87e7a0a970f56e4",
        "sha256": "8ffdb54bf6847c8c6e7e97545a4e455eab0a2ff8f427d01de2a50e65015b96d0",
        "repository": "https://github.com/dayhofflabs/horizyn.git",
        "commit": "e6655e732f574c8bfa0488b9bc5068b67e382745",
        "zenodo_doi": "10.5281/zenodo.17957034",
        "zenodo_record": "17957034",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group horizyn",
    },
    {
        "name": "Horizyn SOTA test reactions",
        "target": "external/horizyn/data/sota/test_rxns.csv",
        "bytes": 287620,
        "md5": "a45305ba22d4077d7a3f07d5f5d93ff5",
        "sha256": "9596d2bb4c2d4033c5fdada6c563a186e235fbbdb13e06c0d59e569fe077f514",
        "repository": "https://github.com/dayhofflabs/horizyn.git",
        "commit": "e6655e732f574c8bfa0488b9bc5068b67e382745",
        "zenodo_doi": "10.5281/zenodo.17957034",
        "zenodo_record": "17957034",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group horizyn",
    },
    {
        "name": "Horizyn SOTA configuration",
        "target": "external/horizyn/configs/sota.yaml",
        "bytes": 2381,
        "sha256": "582545d9e008d218e424f037a3927546fdc7b16a25ff37c3141b0888f2006177",
        "repository": "https://github.com/dayhofflabs/horizyn.git",
        "commit": "e6655e732f574c8bfa0488b9bc5068b67e382745",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group horizyn",
        "role": "pinned_repo_member",
    },
    {
        "name": "Horizyn v1.0 development checkpoint (training source)",
        "target": "external/horizyn/checkpoints/horizyn_v1_0_dev.ckpt",
        "bytes": 201401250,
        "md5": "5b1f938f8b0a82fbe91892a3b4e2bf2c",
        "sha256": "31bb9b6d73241b7807050377799de8b4bfb17f42a6cd652c8b17b65faf754c25",
        "repository": "https://github.com/dayhofflabs/horizyn.git",
        "commit": "e6655e732f574c8bfa0488b9bc5068b67e382745",
        "zenodo_doi": "10.5281/zenodo.20348783",
        "zenodo_record": "20348783",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group horizyn",
        "role": "independent_training_input_for_exact_residual",
    },
    {
        "name": "ReactZyme UniProt-Rhea source snapshot",
        "target": "data/external/reactzyme/cleaned_uniprot_rhea.tsv",
        "bytes": 87396123,
        "md5": "669bdd627c946114e87f06bffb4f33d9",
        "sha256": "c2d1807562e1e296796820499733bdacab8d4e18ce753e499be558512a927359",
        "repository": "https://github.com/WillHua127/ReactZyme",
        "zenodo_doi": "10.5281/zenodo.11494913",
        "zenodo_record": "11494913",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group reactzyme",
    },
    {
        "name": "ReactZyme Rhea molecule snapshot",
        "target": "data/external/reactzyme/rhea_molecules.tsv",
        "bytes": 3497386,
        "md5": "cb5a575a08954f6d28311b9a4bef52fe",
        "sha256": "98147c030b4f814da059cc9112b126f42f0239be93cf030f62103c35e122714e",
        "repository": "https://github.com/WillHua127/ReactZyme",
        "zenodo_doi": "10.5281/zenodo.11494913",
        "zenodo_record": "11494913",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group reactzyme",
    },
    {
        "name": "ReactZyme enzyme split archive",
        "target": "data/external/reactzyme/enzyme_smi_split.zip",
        "bytes": 47503969,
        "md5": "e351fdb85830968fc9abe933c39f9eda",
        "sha256": "80e00f03c8d934af20cab872c9b1df00f093c47524a00b1af8d7743d91841312",
        "repository": "https://github.com/WillHua127/ReactZyme",
        "zenodo_doi": "10.5281/zenodo.11494913",
        "zenodo_record": "11494913",
        "restore": ".venv/bin/python scripts/maintenance/restore_reproduction_assets.py --group reactzyme",
    },
    {
        "name": "Horizyn v1.0 development checkpoint",
        "target": "results/terpene_production_models/marts_adapted_drfp_pu_r2e_exact_residual/horizyn_v1_0_dev.ckpt",
        "bytes": 201401250,
        "sha256": "31bb9b6d73241b7807050377799de8b4bfb17f42a6cd652c8b17b65faf754c25",
        "repository": "https://github.com/dayhofflabs/horizyn.git",
        "commit": "e6655e732f574c8bfa0488b9bc5068b67e382745",
        "zenodo_doi": "10.5281/zenodo.20348783",
        "restore": "bash scripts/bootstrap_terpene_runtime.sh",
    },
    {
        "name": "CLIPZyme released checkpoint",
        "target": "external_models/clipzyme_checkpoint/clipzyme_model.ckpt",
        "bytes": 2706606717,
        "sha256": "536257d84126342105bd96046d98f68f58de7ceaa063331bb5b240e72c29bc98",
        "repository": "https://github.com/pgmikhael/clipzyme.git",
        "commit": "6e48ae05e2cf705af16368afc579246d80767326",
        "zenodo_doi": "10.5281/zenodo.11187747",
        "upstream_checkpoint_url": "https://zenodo.org/records/11187895/files/clipzyme_model.zip",
    },
    {
        "name": "CLIPZyme released screening data",
        "target": "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p",
        "sha256": "2b2f5072aed2adb13e06341ade74eac51790489daf76434c63fe5c517ac8c9bd",
        "repository": "https://github.com/pgmikhael/clipzyme.git",
        "commit": "6e48ae05e2cf705af16368afc579246d80767326",
        "zenodo_doi": "10.5281/zenodo.11187747",
        "archive_md5": "1ebd955e83fa480aea198c20c1a66381",
    },
    {
        "name": "EnzGFM-650M encoder",
        "target": "external_models/enzgfm/EnzGFM_650M/model.safetensors",
        "bytes": 2625542592,
        "sha256": "6fe6e90e4ef83789d3d117da0fb20246d46bcab462473536b85cd78ea6a7cb9d",
        "repository": "https://github.com/DeepBxM/EnzGFM.git",
        "commit": "0631099828e1b2aadbfe56751eec5f67a7bfdf0c",
        "zenodo_record": "22042585",
        "archive_md5": "4fc54568ac2c913722895415fc64097e",
    },    {
        "name": "Rhea release128 Swiss-Prot association snapshot",
        "target": "data/external/rhea_snapshot_external_v1/release128_rhea2uniprot_sprot.tsv",
        "bytes": 7742487,
        "sha256": "06a88c1fb29a3170bc533e9557e8da7b278c6c0e2583159492c08aaea958abe0",
        "upstream_archive_url": "https://ftp.expasy.org/databases/rhea/old_releases/128.tar.bz2",
        "archive_member_name": "rhea2uniprot_sprot.tsv",
        "rhea_release": 128,
        "release_date": "2023-07-12",
    },
    {
        "name": "Rhea release141 Swiss-Prot association snapshot",
        "target": "data/external/rhea_snapshot_external_v1/release141_rhea2uniprot_sprot.tsv",
        "bytes": 8757953,
        "sha256": "0dcfdb4fb8cdc126004f9fee9fd519e605ec09a6ad5ecf495baf0a29063498d7",
        "upstream_archive_url": "https://ftp.expasy.org/databases/rhea/old_releases/141.tar.bz2",
        "archive_member_name": "rhea2uniprot_sprot.tsv",
        "rhea_release": 141,
        "release_date": "2026-06-10",
    },
]

EVALUATION_SUPPORT_REBUILDS = [
    {
        "asset": "results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv",
        "builder": "projects/active/terpene_screening/rebuild_rhea128_to141_strict_support_v2.py",
        "command": ".venv/bin/python projects/active/terpene_screening/rebuild_rhea128_to141_strict_support_v2.py --release128-sprot data/external/rhea_snapshot_external_v1/release128_rhea2uniprot_sprot.tsv --release141-sprot data/external/rhea_snapshot_external_v1/release141_rhea2uniprot_sprot.tsv --output-root <output-dir>",
        "expected_sha256": "9a53a465e6327e2c04a4fdd6171abd7d076aec2a3441a34955bf0f4526bc3334",
        "model_scoring": False,
        "protocol_modified": False,
    },
]

PRIVATE_ROOTS = [
    "local_candidate_libraries/",
    "downloads/",
    "archive/experiments/",
    "reports/",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(relative: str) -> Iterable[str]:
    root = ROOT / relative
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield str(path.relative_to(ROOT))


def add_existing(target: set[str], paths: Iterable[str]) -> None:
    for relative in paths:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
        target.add(relative)


def main() -> None:
    runtime = json.loads((ROOT / "reproducibility/terpene_runtime_manifest.json").read_text())
    canonical = json.loads((ROOT / "reproducibility/bime_rank/canonical.json").read_text())

    runtime_files = set(runtime["files"])
    direct: set[str] = runtime_files - historical_runtime_direct_excludes(runtime_files)
    direct.update(claim["primary"] for claim in canonical["claims"].values())
    add_existing(direct, PRODUCTION_ROUTE_EVIDENCE_FILES)
    add_existing(direct, CURRENT_RUNTIME_SUPPORT_FILES)
    add_existing(direct, CURRENT_MODEL_TRAINING_INPUT_FILES)
    add_existing(direct, DATABASE_RELEASE_FILES)
    add_existing(direct, [item["path"] for item in EVALUATION_SUPPORT_ASSETS])
    add_existing(direct, [item["path"] for item in AGGREGATE_SUPPORT_ASSETS])

    for root in MODEL_REPRODUCTION_SUPPORT_ROOTS:
        for relative in files_under(root):
            direct.add(relative)

    for root in CURRENT_MODEL_ROOTS:
        for relative in files_under(root):
            if relative not in CURRENT_MODEL_EXCLUDES:
                direct.add(relative)

    for root in REACTION_FEATURE_METADATA_DIRS:
        for relative in files_under(root):
            if not relative.endswith("/reaction_feature_matrix.npy"):
                direct.add(relative)

    records = []
    for relative in sorted(direct):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        size = path.stat().st_size
        if size >= GITHUB_BLOB_LIMIT:
            raise RuntimeError(f"direct release asset exceeds GitHub blob limit: {relative} ({size})")
        if any(relative.startswith(root) for root in PRIVATE_ROOTS):
            raise RuntimeError(f"private/local root cannot be a release asset: {relative}")
        records.append({"path": relative, "bytes": size, "sha256": sha256(path)})

    direct_by_path = {record["path"]: record for record in records}
    evaluation_support_records = []
    for item in EVALUATION_SUPPORT_ASSETS:
        relative = str(item["path"])
        direct_record = direct_by_path[relative]
        evaluation_support_records.append({
            **item,
            "bytes": int(direct_record["bytes"]),
            "sha256": str(direct_record["sha256"]),
        })

    aggregate_support_records = []
    for item in AGGREGATE_SUPPORT_ASSETS:
        relative = str(item["path"])
        direct_record = direct_by_path[relative]
        aggregate_support_records.append({
            **item,
            "bytes": int(direct_record["bytes"]),
            "sha256": str(direct_record["sha256"]),
        })

    payload = {
        "schema_version": 1,
        "release_date": "2026-09-07",
        "release_branch": "master",
        "policy": {
            "git_scope": "explicit whitelist; data/ and results/ remain ignored by default and only listed release assets are force-tracked",
            "self_trained_weights": "commit directly when below the GitHub single-blob limit",
            "large_derived_databases": "commit canonical tables/manifests/builders; rebuild matrices deterministically or from fixed model inputs",
            "large_third_party_weights": "do not vendor; pin upstream source/version/checksum and restore separately",
            "private_local_assets": "never commit",
            "scientific_model_or_benchmark_experiments_rerun_for_release": False,
            "deterministic_reproducibility_reruns_for_release": [
                "reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json"
            ],
        },
        "direct_git_assets": records,
        "direct_git_asset_count": len(records),
        "direct_git_bytes": sum(r["bytes"] for r in records),
        "canonical_claim_primaries": {
            claim_id: claim["primary"] for claim_id, claim in canonical["claims"].items()
        },
        "rebuildable_assets": REBUILDABLE_ASSETS,
        "evaluation_support_assets": evaluation_support_records,
        "aggregate_support_assets": aggregate_support_records,
        "evaluation_support_rebuilds": EVALUATION_SUPPORT_REBUILDS,
        "external_assets": EXTERNAL_ASSETS,
        "private_roots": PRIVATE_ROOTS,
        "publication_metadata": {
            "citation_file": "CITATION.cff",
            "third_party_notices": "THIRD_PARTY_NOTICES.md",
            "project_license_status": "not_declared",
            "project_license_file": None,
        },
        "validation": {
            "validator": "scripts/maintenance/validate_research_release.py",
            "runtime_validator": "scripts/verify_terpene_runtime.py --portable-only",
            "judge_validator": "scripts/maintenance/validate_bime_judge_report.py",
            "source_role_manifest": "reproducibility/bime_rank/source_roles.json",
            "source_role_builder": "scripts/maintenance/build_bime_source_roles.py",
            "canonical_source_provenance": "reproducibility/bime_rank/canonical_source_provenance.json",
            "model_asset_index": "reproducibility/bime_rank/model_assets.json",
            "model_asset_builder": "scripts/maintenance/build_bime_model_asset_index.py",
            "database_asset_index": "reproducibility/bime_rank/database_assets.json",
            "database_asset_builder": "scripts/maintenance/build_bime_database_asset_index.py",
            "historical_source_demotions": "reproducibility/bime_rank/historical_source_demotions.json",
            "historical_research_source_demotions": "reproducibility/bime_rank/historical_research_source_demotions.json",
            "historical_artifact_demotions": "reproducibility/bime_rank/historical_artifact_demotions.json",
            "historical_runtime_asset_demotions": "reproducibility/bime_rank/historical_runtime_asset_demotions.json",
            "project_test_runner": "scripts/maintenance/run_bime_project_tests.py",
            "release_regression_suite": [
                "scripts/maintenance/tests/test_bime_asset_resolver.py",
                "projects/active/terpene_screening/tests/test_bime_context_experts.py",
                "projects/active/terpene_screening/tests/test_bime_rank_candidate_contract.py",
                "projects/active/terpene_screening/tests/test_clipzyme_directed_fallback_contract_v1.py",
                "projects/active/terpene_screening/tests/test_hierarchical_expert_routing.py",
                "projects/active/terpene_screening/tests/test_production_core.py",
                "projects/active/terpene_screening/tests/test_release_identity_and_legacy_redirects.py",
            ],
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUT.relative_to(ROOT)),
        "direct_git_asset_count": len(records),
        "direct_git_bytes": payload["direct_git_bytes"],
        "rebuildable_assets": len(REBUILDABLE_ASSETS),
        "external_assets": len(EXTERNAL_ASSETS),
    }, indent=2))


if __name__ == "__main__":
    main()
