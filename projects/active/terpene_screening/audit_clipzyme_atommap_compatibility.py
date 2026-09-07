from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPING = ROOT / "data/external/rxnmapper_current/general_merged_v1"
DEFAULT_BENCHMARK = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_atommap_compatibility_audit_v1"


def compatibility(mapped_rxn: str, rxnmapper_success: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "all_atoms_mapped": False,
        "reactant_unique_maps": False,
        "product_unique_maps": False,
        "map_sets_equal": False,
        "atom_counts_equal": False,
        "clipzyme_graph_prereq": False,
        "reactant_atoms": 0,
        "product_atoms": 0,
        "unmapped_reactant_atoms": 0,
        "unmapped_product_atoms": 0,
        "compatibility_reason": "rxnmapper_failed",
    }
    if not rxnmapper_success or ">>" not in mapped_rxn:
        return record
    left, right = mapped_rxn.split(">>", 1)
    left_mol, right_mol = Chem.MolFromSmiles(left), Chem.MolFromSmiles(right)
    if left_mol is None or right_mol is None:
        record["compatibility_reason"] = "rdkit_parse_failed"
        return record
    left_maps = [atom.GetAtomMapNum() for atom in left_mol.GetAtoms()]
    right_maps = [atom.GetAtomMapNum() for atom in right_mol.GetAtoms()]
    all_mapped = all(value > 0 for value in left_maps + right_maps)
    left_unique = all(value > 0 for value in left_maps) and len(left_maps) == len(set(left_maps))
    right_unique = all(value > 0 for value in right_maps) and len(right_maps) == len(set(right_maps))
    equal_sets = left_unique and right_unique and set(left_maps) == set(right_maps)
    prereq = bool(all_mapped and equal_sets)
    if not all_mapped:
        reason = "unmapped_atoms"
    elif not left_unique or not right_unique:
        reason = "duplicate_atom_maps"
    elif not equal_sets:
        reason = "reactant_product_map_sets_differ"
    else:
        reason = "compatible"
    record.update(
        {
            "all_atoms_mapped": all_mapped,
            "reactant_unique_maps": left_unique,
            "product_unique_maps": right_unique,
            "map_sets_equal": equal_sets,
            "atom_counts_equal": len(left_maps) == len(right_maps),
            "clipzyme_graph_prereq": prereq,
            "reactant_atoms": len(left_maps),
            "product_atoms": len(right_maps),
            "unmapped_reactant_atoms": sum(value <= 0 for value in left_maps),
            "unmapped_product_atoms": sum(value <= 0 for value in right_maps),
            "compatibility_reason": reason,
        }
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether RXNMapper outputs satisfy the structural atom-map assumptions of the official CLIPZyme difference-graph reaction encoder."
    )
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    mapping_path = args.mapping_dir.resolve() / "mapped_reactions.csv"
    mapping = pd.read_csv(mapping_path, dtype=str).fillna("")
    mapping["rxnmapper_success"] = mapping["success"].str.lower().eq("true")
    mapping["mapping_confidence"] = pd.to_numeric(mapping["confidence"], errors="coerce")
    records = []
    for row in mapping.itertuples(index=False):
        records.append(
            {
                "reaction_id": str(row.reaction_id),
                "rxnmapper_success": bool(row.rxnmapper_success),
                "mapping_confidence": row.mapping_confidence,
                **compatibility(str(row.mapped_rxn), bool(row.rxnmapper_success)),
            }
        )
    frame = pd.DataFrame(records)
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "reaction_compatibility.csv", index=False)
    benchmark_rows = []
    root = args.benchmark_root.resolve()
    for cell_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "test_pairs.csv").is_file()):
        test = pd.read_csv(cell_dir / "test_pairs.csv", dtype=str).fillna("")
        ids = pd.DataFrame({"reaction_id": sorted(test["reaction_id"].unique())})
        local = ids.merge(frame, on="reaction_id", how="left", validate="one_to_one")
        mapper_ok = local["rxnmapper_success"].fillna(False).astype(bool)
        prereq = local["clipzyme_graph_prereq"].fillna(False).astype(bool)
        all_mapped = local["all_atoms_mapped"].fillna(False).astype(bool)
        equal_sets = local["map_sets_equal"].fillna(False).astype(bool)
        benchmark_rows.append(
            {
                "cell": cell_dir.name,
                "unique_test_reactions": len(local),
                "rxnmapper_success": int(mapper_ok.sum()),
                "clipzyme_graph_prereq": int(prereq.sum()),
                "clipzyme_graph_prereq_fraction": float(prereq.mean()),
                "all_atoms_mapped": int(all_mapped.sum()),
                "map_sets_equal": int(equal_sets.sum()),
            }
        )
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(out / "benchmark_coverage.csv", index=False)
    summary = {
        "reaction_count": int(len(frame)),
        "rxnmapper_success_count": int(frame["rxnmapper_success"].sum()),
        "all_atoms_mapped_count": int(frame["all_atoms_mapped"].sum()),
        "equal_map_set_count": int(frame["map_sets_equal"].sum()),
        "clipzyme_graph_prereq_count": int(frame["clipzyme_graph_prereq"].sum()),
        "clipzyme_graph_prereq_fraction": float(frame["clipzyme_graph_prereq"].mean()),
        "definition": "Every reactant and product atom has a positive unique map number and both sides have identical atom-map sets. This is a pre-model input-domain requirement induced by the official CLIPZyme process_mapped_reaction and aligned difference-graph encoder, not a performance-derived filter.",
        "unsupported_policy": "report support explicitly; do not repair, zero-impute, or silently exclude incompatible reactions from end-to-end claims",
        "compatibility_reason_counts": frame["compatibility_reason"].value_counts(dropna=False).to_dict(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
