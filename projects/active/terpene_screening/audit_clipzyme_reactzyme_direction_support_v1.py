from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem, RDLogger

from projects.active.terpene_screening.audit_clipzyme_atommap_compatibility import compatibility
from projects.active.terpene_screening.audit_clipzyme_outer_overlap import clipzyme_train_sets, reaction_keys_from_smiles
from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import normalize_reaction_bag, sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TEST = ROOT / "data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt"
DEFAULT_REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
DEFAULT_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv"
DEFAULT_CLIPZYME_CACHE = ROOT / "external_models/clipzyme_audit/clipzyme_data/cached_enzymemap.p"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_reactzyme_direction_support_v1"
MIN_DIRECT_QUERIES = 50


def canonical_molecule(smiles: str) -> str | None:
    # Match ReactZyme author support semantics for wildcard atoms before RDKit canonicalization.
    mol = Chem.MolFromSmiles(str(smiles).replace("*", "C"))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_bag(value: str) -> str | None:
    parts = [part for part in normalize_reaction_bag(value).split(".") if part]
    canonical = [canonical_molecule(part) for part in parts]
    if not parts or any(item is None for item in canonical):
        return None
    return ".".join(sorted(str(item) for item in canonical))


def sideblind_key_from_directed(smiles: str) -> str | None:
    if ">>" not in str(smiles):
        return None
    left, right = str(smiles).split(">>", 1)
    pieces = [part for part in left.split(".") if part] + [part for part in right.split(".") if part]
    canonical = [canonical_molecule(part) for part in pieces]
    if not pieces or any(item is None for item in canonical):
        return None
    return ".".join(sorted(str(item) for item in canonical))


def load_reactzyme_test(path: Path) -> pd.DataFrame:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for source_key, value in payload.items():
        if len(value) < 2:
            raise ValueError(f"unexpected ReactZyme tuple at {source_key}")
        bag = normalize_reaction_bag(value[0])
        rows.append({"source_key": str(source_key), "reaction_bag": bag, "sideblind_key": canonical_bag(bag)})
    frame = pd.DataFrame(rows)
    if len(frame) != 14692:
        raise AssertionError(f"ReactZyme reaction_smi test row drift: {len(frame)} != 14692")
    if frame["reaction_bag"].nunique() != 386:
        raise AssertionError("ReactZyme reaction_smi unique reaction count drift")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Performance-blind audit for whether unordered ReactZyme reaction_smi queries can be mapped to "
            "independently recorded directed reactions that are valid native CLIPZyme inputs. No model score is read."
        )
    )
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--reactions", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--clipzyme-cache", type=Path, default=DEFAULT_CLIPZYME_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-direct-queries", type=int, default=MIN_DIRECT_QUERIES)
    args = parser.parse_args()
    if args.minimum_direct_queries < 1:
        raise ValueError("minimum-direct-queries must be positive")

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    test = load_reactzyme_test(args.test.resolve())

    reactions = pd.read_csv(args.reactions.resolve(), dtype=str).fillna("")
    required = {"reaction_id", "reaction_smiles", "source_layer"}
    if not required <= set(reactions):
        raise ValueError(f"reaction registry missing columns: {sorted(required - set(reactions))}")
    reactions = reactions[["reaction_id", "reaction_smiles", "source_layer"]].drop_duplicates("reaction_id")
    reactions["sideblind_key"] = reactions["reaction_smiles"].map(sideblind_key_from_directed)
    oriented_and_undirected = reactions["reaction_smiles"].map(reaction_keys_from_smiles)
    reactions["oriented_key"] = [value[0] for value in oriented_and_undirected]
    reactions["undirected_key"] = [value[1] for value in oriented_and_undirected]

    by_bag: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in reactions.itertuples(index=False):
        if row.sideblind_key and row.oriented_key and row.undirected_key:
            by_bag[str(row.sideblind_key)].append(
                {
                    "reaction_id": str(row.reaction_id),
                    "reaction_smiles": str(row.reaction_smiles),
                    "source_layer": str(row.source_layer),
                    "oriented_key": str(row.oriented_key),
                    "undirected_key": str(row.undirected_key),
                }
            )

    mapping = pd.read_csv(args.mapping.resolve(), dtype=str).fillna("")
    if not {"reaction_id", "mapped_rxn", "success", "confidence"} <= set(mapping):
        raise ValueError("RXNMapper registry is missing required columns")
    mapping = mapping.drop_duplicates("reaction_id").set_index("reaction_id", drop=False)

    samples = pickle.load(args.clipzyme_cache.resolve().open("rb"))
    if not isinstance(samples, list) or not samples:
        raise ValueError("CLIPZyme cache must contain a nonempty sample list")
    clip_train = clipzyme_train_sets(samples)
    clip_train_undirected = set(clip_train["reaction_undirected"])

    unique_test = test[["reaction_bag", "sideblind_key"]].drop_duplicates().sort_values("reaction_bag")
    audit_rows = []
    for row in unique_test.itertuples(index=False):
        candidates = by_bag.get(str(row.sideblind_key), []) if row.sideblind_key else []
        directed_keys = sorted({item["oriented_key"] for item in candidates})
        unique_direction = len(directed_keys) == 1
        selected = None
        if unique_direction:
            # Multiple registry IDs can encode the same chemistry. Selection is lexical and fixed before
            # mapping/support inspection, never chosen for RXNMapper or retrieval performance.
            selected = sorted(
                (item for item in candidates if item["oriented_key"] == directed_keys[0]),
                key=lambda item: (item["reaction_id"], item["source_layer"]),
            )[0]

        map_success = False
        map_confidence = None
        mapped_rxn = ""
        compat = compatibility("", False)
        if selected is not None and selected["reaction_id"] in mapping.index:
            m = mapping.loc[selected["reaction_id"]]
            map_success = str(m["success"]).lower() == "true"
            map_confidence = pd.to_numeric(pd.Series([m["confidence"]]), errors="coerce").iloc[0]
            mapped_rxn = str(m["mapped_rxn"])
            compat = compatibility(mapped_rxn, map_success)

        clip_train_seen = bool(selected and selected["undirected_key"] in clip_train_undirected)
        native_reaction_supported = bool(
            selected is not None
            and unique_direction
            and compat["clipzyme_graph_prereq"]
            and not clip_train_seen
        )
        audit_rows.append(
            {
                "reaction_bag": str(row.reaction_bag),
                "sideblind_key": str(row.sideblind_key or ""),
                "independent_registry_candidates": len(candidates),
                "unique_directed_records": len(directed_keys),
                "direction_status": "unique" if unique_direction else ("missing" if not directed_keys else "ambiguous"),
                "selected_reaction_id": selected["reaction_id"] if selected else "",
                "selected_source_layer": selected["source_layer"] if selected else "",
                "selected_directed_smiles": selected["reaction_smiles"] if selected else "",
                "rxnmapper_record_present": bool(selected and selected["reaction_id"] in mapping.index),
                "rxnmapper_success": map_success,
                "mapping_confidence": map_confidence,
                "clipzyme_graph_prereq": bool(compat["clipzyme_graph_prereq"]),
                "compatibility_reason": str(compat["compatibility_reason"]),
                "clipzyme_train_reaction_undirected_seen": clip_train_seen,
                "reaction_novel_to_clipzyme_train": bool(selected is not None and not clip_train_seen),
                "native_reaction_supported": native_reaction_supported,
            }
        )

    audit = pd.DataFrame(audit_rows)
    eligible = audit[audit["native_reaction_supported"]].copy()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "reaction_direction_audit.csv", index=False)
    eligible.to_csv(out / "eligible_reaction_queries.csv", index=False)

    direct_queries = int(len(eligible))
    decision = (
        "freeze_exact_native_subset_then_audit_author_protein_support"
        if direct_queries >= args.minimum_direct_queries
        else "reactzyme_direct_clipzyme_comparison_unsupported_use_separate_directed_reaction_novel_benchmark"
    )
    summary = {
        "status": "performance_blind_input_and_training_overlap_audit_no_model_scores",
        "test_unique_reaction_bags": int(len(audit)),
        "registry_rows": int(len(reactions)),
        "registry_parseable_directed_rows": int(reactions["oriented_key"].notna().sum()),
        "direction_unique_queries": int((audit["direction_status"] == "unique").sum()),
        "direction_ambiguous_queries": int((audit["direction_status"] == "ambiguous").sum()),
        "direction_missing_queries": int((audit["direction_status"] == "missing").sum()),
        "rxnmapper_success_queries": int(audit["rxnmapper_success"].sum()),
        "clipzyme_graph_prereq_queries": int(audit["clipzyme_graph_prereq"].sum()),
        "clipzyme_train_seen_queries": int(audit["clipzyme_train_reaction_undirected_seen"].sum()),
        "reaction_novel_native_clipzyme_supported_queries": direct_queries,
        "minimum_direct_queries": int(args.minimum_direct_queries),
        "protein_support_pending": True,
        "decision": decision,
        "selection_uses_model_scores": False,
        "selection_uses_test_performance": False,
        "selection_uses_test_labels_beyond_query_identity": False,
        "source_sha256": {
            "reactzyme_test": sha256_file(args.test.resolve()),
            "reaction_registry": sha256_file(args.reactions.resolve()),
            "rxnmapper_registry": sha256_file(args.mapping.resolve()),
            "clipzyme_cache": sha256_file(args.clipzyme_cache.resolve()),
        },
        "fairness_boundary": (
            "A query is eligible only if unordered ReactZyme chemistry maps to exactly one independently recorded directed reaction, "
            "that record satisfies the official CLIPZyme atom-map graph prerequisite, and the reaction was not present in the reconstructed "
            "CLIPZyme author training split. Protein-side author support is audited separately before any direct score matrix is formed."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
