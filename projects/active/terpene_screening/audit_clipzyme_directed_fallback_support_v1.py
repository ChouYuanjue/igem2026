from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from rdkit import RDLogger

from projects.active.terpene_screening.audit_clipzyme_atommap_compatibility import compatibility
from projects.active.terpene_screening.audit_clipzyme_outer_overlap import clipzyme_train_sets, reaction_keys_from_smiles
from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
DEFAULT_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
DEFAULT_CLIPZYME_CACHE = ROOT / "external_models/clipzyme_audit/clipzyme_data/cached_enzymemap.p"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_directed_fallback_support_v1"
PRIORITY_CELLS = (
    "reactzyme_reaction_projected_double_cold",
    "temporal_post2020_double_cold",
    "broad_reaction_hash_cold_protein_seen",
)
MIN_REACTION_QUERIES = 50


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Performance-blind reaction-side native-support audit for the frozen directed CLIPZyme fallback cells. "
            "This script never reads retrieval scores and never finalizes a cell before protein-side author support is audited."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--reactions", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--clipzyme-cache", type=Path, default=DEFAULT_CLIPZYME_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-reaction-queries", type=int, default=MIN_REACTION_QUERIES)
    args = parser.parse_args()
    if args.minimum_reaction_queries < 1:
        raise ValueError("minimum-reaction-queries must be positive")

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    reactions = pd.read_csv(args.reactions.resolve(), dtype=str).fillna("")
    if not {"reaction_id", "reaction_smiles", "source_layer"} <= set(reactions):
        raise ValueError("reaction registry is missing reaction_id/reaction_smiles/source_layer")
    reactions = reactions[["reaction_id", "reaction_smiles", "source_layer"]].drop_duplicates("reaction_id")
    keys = reactions["reaction_smiles"].map(reaction_keys_from_smiles)
    reactions["oriented_key"] = [value[0] for value in keys]
    reactions["undirected_key"] = [value[1] for value in keys]
    reaction_lookup = reactions.set_index("reaction_id", drop=False)

    mapping = pd.read_csv(args.mapping.resolve(), dtype=str).fillna("")
    if not {"reaction_id", "mapped_rxn", "success", "confidence"} <= set(mapping):
        raise ValueError("RXNMapper registry is missing required columns")
    mapping = mapping.drop_duplicates("reaction_id").set_index("reaction_id", drop=False)

    samples = pickle.load(args.clipzyme_cache.resolve().open("rb"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("CLIPZyme cache must contain a nonempty sample list")
    clip_train = clipzyme_train_sets(samples)
    clip_train_undirected = set(clip_train["reaction_undirected"])

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    first_reaction_side_candidate: str | None = None

    for priority, cell in enumerate(PRIORITY_CELLS, start=1):
        pair_path = args.benchmark_root.resolve() / cell / "test_pairs.csv"
        if not pair_path.is_file():
            summaries.append({
                "priority": priority,
                "cell": cell,
                "status": "missing_frozen_cell",
                "reaction_novel_native_supported_queries": 0,
                "meets_reaction_side_minimum": False,
            })
            continue
        pairs = pd.read_csv(pair_path, dtype=str).fillna("")
        if "reaction_id" not in pairs:
            raise ValueError(f"{pair_path} lacks reaction_id")
        reaction_ids = sorted(set(pairs["reaction_id"].astype(str)))
        rows: list[dict[str, object]] = []
        for reaction_id in reaction_ids:
            registry_present = reaction_id in reaction_lookup.index
            reaction_smiles = ""
            source_layer = ""
            oriented_key = None
            undirected_key = None
            if registry_present:
                r = reaction_lookup.loc[reaction_id]
                reaction_smiles = str(r["reaction_smiles"])
                source_layer = str(r["source_layer"])
                oriented_key = r["oriented_key"] if pd.notna(r["oriented_key"]) else None
                undirected_key = r["undirected_key"] if pd.notna(r["undirected_key"]) else None

            mapping_present = reaction_id in mapping.index
            map_success = False
            map_confidence = None
            compat = compatibility("", False)
            if mapping_present:
                m = mapping.loc[reaction_id]
                map_success = str(m["success"]).lower() == "true"
                map_confidence = pd.to_numeric(pd.Series([m["confidence"]]), errors="coerce").iloc[0]
                compat = compatibility(str(m["mapped_rxn"]), map_success)

            clip_train_seen = bool(undirected_key and undirected_key in clip_train_undirected)
            native_supported = bool(
                registry_present
                and oriented_key
                and undirected_key
                and compat["clipzyme_graph_prereq"]
                and not clip_train_seen
            )
            rows.append({
                "reaction_id": reaction_id,
                "source_layer": source_layer,
                "directed_reaction_smiles": reaction_smiles,
                "independent_directed_registry_present": registry_present,
                "directed_reaction_parseable": bool(oriented_key and undirected_key),
                "rxnmapper_record_present": mapping_present,
                "rxnmapper_success": map_success,
                "mapping_confidence": map_confidence,
                "clipzyme_graph_prereq": bool(compat["clipzyme_graph_prereq"]),
                "compatibility_reason": str(compat["compatibility_reason"]),
                "clipzyme_train_reaction_undirected_seen": clip_train_seen,
                "reaction_novel_to_clipzyme_train": bool(undirected_key and not clip_train_seen),
                "native_reaction_supported": native_supported,
            })
        detail = pd.DataFrame(rows)
        detail.to_csv(out / f"{priority:02d}_{cell}_reaction_support.csv", index=False)
        supported = int(detail["native_reaction_supported"].sum())
        meets = supported >= args.minimum_reaction_queries
        if meets and first_reaction_side_candidate is None:
            first_reaction_side_candidate = cell
        summaries.append({
            "priority": priority,
            "cell": cell,
            "status": "reaction_side_audited_protein_side_pending",
            "unique_reaction_queries": int(len(detail)),
            "directed_registry_queries": int(detail["independent_directed_registry_present"].sum()),
            "directed_parseable_queries": int(detail["directed_reaction_parseable"].sum()),
            "rxnmapper_success_queries": int(detail["rxnmapper_success"].sum()),
            "clipzyme_graph_prereq_queries": int(detail["clipzyme_graph_prereq"].sum()),
            "clipzyme_train_seen_queries": int(detail["clipzyme_train_reaction_undirected_seen"].sum()),
            "reaction_novel_native_supported_queries": supported,
            "minimum_reaction_queries": int(args.minimum_reaction_queries),
            "meets_reaction_side_minimum": meets,
        })

    summary = {
        "status": "performance_blind_reaction_side_fallback_audit_no_model_scores",
        "priority_order": list(PRIORITY_CELLS),
        "selection_uses_model_scores": False,
        "selection_uses_test_performance": False,
        "reaction_side_priority_candidate": first_reaction_side_candidate,
        "final_fallback_cell_selected": None,
        "final_selection_blocker": "author CLIPZyme protein-side support must be intersected before final cell selection",
        "cells": summaries,
        "source_sha256": {
            "reaction_registry": sha256_file(args.reactions.resolve()),
            "rxnmapper_registry": sha256_file(args.mapping.resolve()),
            "clipzyme_cache": sha256_file(args.clipzyme_cache.resolve()),
        },
        "fairness_boundary": (
            "Priority is frozen before scores. A reaction query is reaction-side eligible only when the frozen benchmark supplies "
            "an independently recorded directed reaction, RXNMapper satisfies the official CLIPZyme graph prerequisite, and the "
            "reaction is absent from the reconstructed CLIPZyme author training split. No retrieval score is loaded by this audit."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(summaries).to_csv(out / "cell_summary.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
