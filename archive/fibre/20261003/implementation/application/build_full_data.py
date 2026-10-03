from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from projects.active.fibre.application.tps_adapted_coordinate import (
    DEFAULT_MODEL as TPS_SOURCE_MODEL,
    DEFAULT_PROTEIN_FEATURES as TPS_CANONICAL_PROTEIN_FEATURES,
    DEFAULT_REACTION_FEATURES as TPS_CANONICAL_REACTION_FEATURES,
    build_tps_adapted_coordinate,
)
from projects.active.fibre.portable.core_bundle import build_bundle


ROOT=Path(__file__).resolve().parents[4]
CANONICAL=ROOT/"data/terpene_marts_adaptation"
PROTEIN_GEOMETRY=ROOT/"data/terpene_multiresolution_protein_geometry_v4"
REACTION_GEOMETRY=ROOT/"data/terpene_multiresolution_reaction_geometry_v1"
DEFAULT_OUTPUT=ROOT/"results/fibre_application/full_data"
OBSERVATION_INDEX=ROOT/"results/fibre_observation_index_v1"
UNIPROT_STATE=ROOT/"results/fibre_uniprot_state_v1"
PUBLICATION_CONTEXT=ROOT/"results/fibre_publication_context_deepseek_v1"
RHEA_MAPPING=ROOT/"results/fibre_rhea_mapping_v1"
ASSAY_CONTEXT=ROOT/"results/fibre_assay_context_v1"
CATALYTIC_STATE=ROOT/"results/fibre_catalytic_state_v1"
CROSS_SOURCE_EVIDENCE=ROOT/"results/fibre_cross_source_catalytic_evidence_v1"


def portable_path(path: str | Path) -> str:
    p=Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda:fh.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: str | Path) -> str:
    base=Path(root)
    h=hashlib.sha256()
    for path in sorted((p for p in base.rglob("*") if p.is_file()),key=lambda p:p.relative_to(base).as_posix()):
        rel=path.relative_to(base).as_posix().encode("utf-8")
        h.update(rel); h.update(b"\0")
        with path.open("rb") as fh:
            for block in iter(lambda:fh.read(1<<20),b""):
                h.update(block)
        h.update(b"\0")
    return h.hexdigest()


def _rewrite_bundle_sources(manifest: dict, output_dir: Path) -> None:
    geometry=manifest.get("factor_geometry") or {}
    for entry in geometry.values():
        if not isinstance(entry,dict):
            continue
        source=entry.get("source")
        if source:
            entry["source"]=portable_path(source)
    (output_dir/"manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
    )


def prepare_application_tables(output: Path) -> dict[str,Path]:
    output.mkdir(parents=True,exist_ok=True)
    proteins=pd.read_csv(CANONICAL/"protein_entities.csv",dtype=str).fillna("")
    reactions=pd.read_csv(CANONICAL/"reaction_entities.csv",dtype=str).fillna("")
    pairs=pd.read_csv(CANONICAL/"marts_pair_folds.csv",dtype=str).fillna("")
    positive=(
        pairs[["Entry","rhea_id"]]
        .drop_duplicates()
        .rename(columns={"Entry":"protein_id","rhea_id":"reaction_id"})
        .sort_values(["reaction_id","protein_id"],kind="stable")
        .reset_index(drop=True)
    )
    protein_ids=set(proteins.protein_id.astype(str))
    reaction_ids=set(reactions.reaction_id.astype(str))
    if not set(positive.protein_id)<=protein_ids:
        raise ValueError("full-data positive support contains protein outside canonical universe")
    if not set(positive.reaction_id)<=reaction_ids:
        raise ValueError("full-data positive support contains reaction outside canonical universe")
    p=output/"proteins.csv"; r=output/"reactions.csv"; o=output/"positive_pairs.csv"
    proteins.to_csv(p,index=False); reactions.to_csv(r,index=False); positive.to_csv(o,index=False)
    return {"proteins":p,"reactions":r,"pairs":o}


def build_full_data_application(
    *,
    output_dir: str | Path=DEFAULT_OUTPUT,
    device: str="cpu",
    graph_backend: str="dense_exact",
) -> dict:
    """Build isolated full-data application references without benchmark semantics.

    This builder is retained for the full-data correspondence application bundle
    and provenance assets. It is not the current FIBRE method definition. Its
    outputs remain useful for the deployed compatibility path and as candidate
    assets for future interaction-atlas integration.
    """
    output=Path(output_dir).resolve()
    tables=prepare_application_tables(output/"inputs")
    tps_dir=output/"tps_adapted_coordinate"
    tps_manifest=build_tps_adapted_coordinate(
        output_dir=tps_dir,device=device
    )

    global_dir=output/"primary_global_reference"
    global_manifest=build_bundle(
        proteins_path=tables["proteins"],
        reactions_path=tables["reactions"],
        pairs_path=tables["pairs"],
        output_dir=global_dir,
        protein_affinity_path=PROTEIN_GEOMETRY/"partial_pullback_affinity.npz",
        reaction_affinity_path=REACTION_GEOMETRY/"partial_pullback_affinity.npz",
    )
    _rewrite_bundle_sources(global_manifest,global_dir)

    tps_bundle_dir=output/"tps_domain_reference"
    tps_bundle_manifest=build_bundle(
        proteins_path=tables["proteins"],
        reactions_path=tables["reactions"],
        pairs_path=tables["pairs"],
        output_dir=tps_bundle_dir,
        protein_features_path=tps_dir/"protein_tps_adapted.csv",
        reaction_features_path=tps_dir/"reaction_tps_adapted.csv",
        graph_backend=graph_backend,
    )
    _rewrite_bundle_sources(tps_bundle_manifest,tps_bundle_dir)

    evidence_sources={
        "observation_index":OBSERVATION_INDEX,
        "uniprot_state":UNIPROT_STATE,
        "publication_context":PUBLICATION_CONTEXT,
        "rhea_mapping":RHEA_MAPPING,
    }
    evidence_generated={
        "assay_context":ASSAY_CONTEXT,
        "catalytic_state":CATALYTIC_STATE,
        "cross_source_catalytic_evidence":CROSS_SOURCE_EVIDENCE,
    }
    manifest={
        "schema":"starase-full-data-application-v1",
        "release_profile":"starase-application",
        "benchmark_claims_allowed":False,
        "purpose":"best-information current TPS application; no train/validation/test partition",
        "canonical_universe":{
            "protein_states":int(global_manifest["protein_count"]),
            "reaction_states":int(global_manifest["reaction_count"]),
            "accepted_positive_pairs":int(global_manifest["positive_pair_count"]),
            "pair_source":"all unique rows of data/terpene_marts_adaptation/marts_pair_folds.csv; fold labels intentionally ignored for application",
        },
        "resolutions":{
            "primary_global":{
                "role":"promoted canonical FIBRE application correspondence",
                "bundle":portable_path(global_dir),
                "protein_geometry":"data/terpene_multiresolution_protein_geometry_v4/partial_pullback_affinity.npz",
                "reaction_geometry":"data/terpene_multiresolution_reaction_geometry_v1/partial_pullback_affinity.npz",
                "pair_supervised_factor_geometry":False,
            },
            "tps_domain":{
                "role":"application-only domain-adapted FIBRE correspondence resolution",
                "bundle":portable_path(tps_bundle_dir),
                "coordinate_manifest":portable_path(tps_dir/"manifest.json"),
                "pair_supervised_factor_geometry":True,
                "legacy_cross_factor_score_used":False,
                "combination_with_primary":(
                    "secondary domain resolution / relation context; not a scalar score "
                    "mixture and not benchmark evidence"
                ),
            },
            "catalytic_pocket":{
                "role":"local catalytic resolution, not forced into primary total ranking",
                "assets":[
                    "data/terpene_catalytic_consensus_geometry_v1",
                    "data/terpene_pocket_local_aligned_v2",
                    "data/terpene_structural_observations_v1",
                ],
            },
            "reaction_center":{
                "role":"catalytic-local reaction resolution; retained outside primary global reaction geodesic",
                "assets":[
                    "data/terpene_multiresolution_reaction_geometry_v2",
                    "data/terpene_reaction_center_observations_v1",
                ],
            },
            "mechanistic":{
                "role":"family-conditional mechanistic coordinates and annotations",
                "assets":[
                    "data/terpene_family_aware_motif_coordinates_v1",
                    "data/terpene_marts_adaptation/protein_architecture_annotations.csv",
                    "data/terpene_marts/marts_mechanism_steps.tsv",
                ],
            },
            "evidence":{
                "role":"provenance-bound interpretation; never an untracked score bonus",
                "assets":[
                    "results/fibre_observation_index_v1",
                    "results/fibre_uniprot_state_v1",
                    "projects/active/fibre/evidence",
                ],
            },
        },
        "input_sha256":{
            "canonical_proteins":sha256_file(tables["proteins"]),
            "canonical_reactions":sha256_file(tables["reactions"]),
            "all_positive_pairs":sha256_file(tables["pairs"]),
            "primary_protein_affinity":sha256_file(
                PROTEIN_GEOMETRY/"partial_pullback_affinity.npz"
            ),
            "primary_reaction_affinity":sha256_file(
                REACTION_GEOMETRY/"partial_pullback_affinity.npz"
            ),
            "tps_coordinate_manifest":sha256_file(tps_dir/"manifest.json"),
        },
        "source_input_sha256":{
            "canonical_protein_entities":sha256_file(CANONICAL/"protein_entities.csv"),
            "canonical_reaction_entities":sha256_file(CANONICAL/"reaction_entities.csv"),
            "accepted_pair_source":sha256_file(CANONICAL/"marts_pair_folds.csv"),
            "primary_protein_affinity":sha256_file(
                PROTEIN_GEOMETRY/"partial_pullback_affinity.npz"
            ),
            "primary_reaction_affinity":sha256_file(
                REACTION_GEOMETRY/"partial_pullback_affinity.npz"
            ),
            "tps_canonical_protein_features":sha256_file(
                TPS_CANONICAL_PROTEIN_FEATURES
            ),
            "tps_canonical_reaction_features":sha256_file(
                TPS_CANONICAL_REACTION_FEATURES
            ),
            "tps_training_pairs":sha256_file(TPS_SOURCE_MODEL/"training_pairs.csv"),
            "tps_feature_schema":sha256_file(TPS_SOURCE_MODEL/"feature_schema.json"),
        },
        "tps_source_checkpoints":list(tps_manifest.get("checkpoints") or []),
        "generated_tree_sha256":{
            "primary_global_reference":tree_sha256(global_dir),
            "tps_adapted_coordinate":tree_sha256(tps_dir),
            "tps_domain_reference":tree_sha256(tps_bundle_dir),
        },
        "tps_source_training":tps_manifest["source_training"],
        "enzymology_evidence":{
            "sources":{
                key:{
                    "path":portable_path(path),
                    "tree_sha256":tree_sha256(path),
                }
                for key,path in evidence_sources.items()
                if path.is_dir()
            },
            "generated":{
                key:{
                    "path":portable_path(path),
                    "tree_sha256":tree_sha256(path),
                }
                for key,path in evidence_generated.items()
                if path.is_dir()
            },
            "required_generated_for_full_information_runtime":[
                "assay_context",
                "catalytic_state",
                "cross_source_catalytic_evidence",
            ],
            "ranking_effect":"none_by_itself",
            "scope_policy":"protein/reaction/pair evidence scopes preserved",
        },
        "important_semantics":{
            "all_available_information_is_assigned_a_role":True,
            "all_information_is_not_forced_into_one_scalar_metric":True,
            "missing_observation_is_negative":False,
            "application_metrics_are_benchmark_claims":False,
        },
    }
    (output/"manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8"
    )
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Build the isolated all-data Starase TPS application references."
    )
    ap.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument("--device",default="cpu")
    ap.add_argument(
        "--tps-graph-backend",
        choices=["dense_exact","blockwise_exact"],
        default="dense_exact",
    )
    args=ap.parse_args()
    print(json.dumps(build_full_data_application(
        output_dir=args.output_dir,
        device=args.device,
        graph_backend=args.tps_graph_backend,
    ),indent=2))


if __name__=="__main__":
    main()
