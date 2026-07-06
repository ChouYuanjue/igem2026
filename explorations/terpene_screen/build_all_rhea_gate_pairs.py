from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENZYMECAGE_ROOT = PROJECT_ROOT / "external_repos" / "EnzymeCAGE"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENZYMECAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENZYMECAGE_ROOT))

from retrieve import getRSim, get_mol_simi_dict  # type: ignore

from explorations.terpene_screen.common import (
    SOURCE_FILES,
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    canonicalize_reaction_smiles,
    coerce_text,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_table,
)


DEFAULT_POSITIVE_PATH = SOURCE_FILES["positive_labels"]
DEFAULT_CANDIDATE_PATH = SOURCE_FILES["candidate_enzymes"]
DEFAULT_OUTPUT_PAIRS = TERPENE_DATA_DIR / "all_rhea_gate_candidate_pairs.csv"
DEFAULT_OUTPUT_MANIFEST = TERPENE_RESULTS_DIR / "all_rhea_gate_reaction_manifest.csv"
DEFAULT_OUTPUT_SUMMARY = TERPENE_RESULTS_DIR / "all_rhea_gate_summary.json"


def _normalize_positive_labels(positive_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
    raw_df = read_table(positive_path)
    columns = identify_terpene_columns(raw_df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    seq_col = columns["sequence"]["column"]
    rxn_col = columns["reaction_smiles"]["column"]
    rhea_col = columns["rhea_id"]["column"]
    if id_col is None or seq_col is None or rhea_col is None:
        raise ValueError(f"Could not identify needed columns in {positive_path}: {raw_df.columns.tolist()}")

    reaction_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    true_map: dict[str, set[str]] = {}

    for rhea_id, group in raw_df.groupby(rhea_col, dropna=False):
        rhea_id_text = coerce_text(rhea_id)
        valid_rows: list[dict[str, Any]] = []
        reaction_smiles_value = ""
        canonical_smiles_value = ""
        enzyme_ids: set[str] = set()
        for _, row in group.iterrows():
            raw_id = coerce_text(row.get(id_col))
            uniprot_id = parse_uniprot_id(raw_id)
            sequence = coerce_text(row.get(seq_col))
            raw_rxn = coerce_text(row.get(rxn_col)) if rxn_col is not None else ""
            if not uniprot_id:
                continue
            enzyme_ids.add(uniprot_id)
            if raw_rxn and not reaction_smiles_value:
                reaction_smiles_value = raw_rxn
                canonical_smiles_value = canonicalize_reaction_smiles(raw_rxn) or raw_rxn
            valid_rows.append(
                {
                    "reaction_id": rhea_id_text,
                    "rhea_id": rhea_id_text,
                    "reaction_smiles": raw_rxn,
                    "CANO_RXN_SMILES": (canonicalize_reaction_smiles(raw_rxn) or raw_rxn) if raw_rxn else "",
                    "enzyme_id": raw_id or uniprot_id,
                    "uniprot_id": uniprot_id,
                    "sequence": sequence,
                    "label": 1,
                }
            )
            if raw_rxn:
                pair_key = (rhea_id_text, uniprot_id)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pair_rows.append(
                        {
                            "reaction_id": rhea_id_text,
                            "rhea_id": rhea_id_text,
                            "reaction_smiles": raw_rxn,
                            "CANO_RXN_SMILES": canonicalize_reaction_smiles(raw_rxn) or raw_rxn,
                            "enzyme_id": raw_id or uniprot_id,
                            "uniprot_id": uniprot_id,
                            "sequence": sequence,
                            "label": 1,
                        }
                    )

        if not valid_rows:
            reaction_rows.append(
                {
                    "reaction_id": rhea_id_text,
                    "rhea_id": rhea_id_text,
                    "reaction_smiles": "",
                    "CANO_RXN_SMILES": "",
                    "status": "no_valid_enzyme_id",
                    "n_true_enzymes": 0,
                }
            )
            true_map[rhea_id_text] = set()
            continue

        reaction_df_row = {
            "reaction_id": rhea_id_text,
            "rhea_id": rhea_id_text,
            "reaction_smiles": reaction_smiles_value,
            "CANO_RXN_SMILES": canonical_smiles_value,
            "status": "ok" if reaction_smiles_value else "no_smiles",
            "n_true_enzymes": len(enzyme_ids),
        }
        reaction_rows.append(reaction_df_row)
        true_map[rhea_id_text] = enzyme_ids

    reaction_df = pd.DataFrame(reaction_rows).drop_duplicates(subset=["reaction_id"]).reset_index(drop=True)
    pair_df = pd.DataFrame(pair_rows).reset_index(drop=True)
    return reaction_df, pair_df, true_map


def _load_candidate_universe(candidate_path: Path) -> pd.DataFrame:
    raw_df = read_table(candidate_path)
    columns = identify_terpene_columns(raw_df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    seq_col = columns["sequence"]["column"]
    if id_col is None or seq_col is None:
        raise ValueError(f"Could not identify candidate columns in {candidate_path}: {raw_df.columns.tolist()}")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in raw_df.iterrows():
        raw_id = coerce_text(row.get(id_col))
        uniprot_id = parse_uniprot_id(raw_id)
        sequence = coerce_text(row.get(seq_col))
        if not uniprot_id or not sequence or uniprot_id in seen:
            continue
        seen.add(uniprot_id)
        rows.append({"enzyme_id": raw_id or uniprot_id, "uniprot_id": uniprot_id, "sequence": sequence})
    return pd.DataFrame(rows).reset_index(drop=True)


def _rxn_similarity_matrix(reaction_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    unique_rxns = [rxn for rxn in reaction_df["CANO_RXN_SMILES"].drop_duplicates().tolist() if rxn]
    cpd_simi_dict, cand_mol_to_id_dict = get_mol_simi_dict(unique_rxns, unique_rxns)
    sim_map: dict[str, dict[str, float]] = {}

    for rxn_target in unique_rxns:
        rcts_smi_counter = Counter(rxn_target.split(">>")[0].split("."))
        pros_smi_counter = Counter(rxn_target.split(">>")[1].split("."))
        sim_row: dict[str, float] = {}
        for cand_rxn in unique_rxns:
            if cand_rxn == rxn_target:
                continue
            cand_rcts = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split(">>")[0].split(".")]
            cand_pros = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split(">>")[1].split(".")]
            cand_rcts_molid_counter, cand_pros_molid_counter = Counter(cand_rcts), Counter(cand_pros)
            s1, s2, _ = getRSim(rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, cpd_simi_dict)
            sim_row[cand_rxn] = max(s1, s2)
        sim_map[rxn_target] = sim_row
    return sim_map


def build_all_rhea_gate_pairs(
    positive_path: Path,
    candidate_path: Path,
    output_pairs: Path,
    output_manifest: Path,
    output_summary: Path,
    topk: int = 10,
) -> dict[str, Any]:
    reaction_df, true_pair_df, true_map = _normalize_positive_labels(positive_path)
    candidate_df = _load_candidate_universe(candidate_path)

    candidate_universe = set(candidate_df["uniprot_id"].astype(str))
    analyzable_reaction_df = reaction_df[(reaction_df["status"] == "ok") & (reaction_df["CANO_RXN_SMILES"] != "")].copy()
    reaction_smiles_map = (
        analyzable_reaction_df[["reaction_id", "rhea_id", "CANO_RXN_SMILES"]]
        .drop_duplicates()
        .set_index("reaction_id")
        .to_dict(orient="index")
    )
    cano_to_reaction_ids = (
        analyzable_reaction_df[["reaction_id", "CANO_RXN_SMILES"]]
        .drop_duplicates()
        .groupby("CANO_RXN_SMILES")["reaction_id"]
        .apply(list)
        .to_dict()
    )

    sim_map = _rxn_similarity_matrix(analyzable_reaction_df)

    manifest_rows: list[dict[str, Any]] = []
    gated_pair_rows: list[dict[str, Any]] = []
    gate_hit_reactions = 0
    total_positive_pairs = 0
    covered_positive_pairs = 0
    total_positive_pairs_all = int(sum(len(enzymes) for enzymes in true_map.values()))

    for reaction_id, meta in reaction_smiles_map.items():
        cano_rxn = meta["CANO_RXN_SMILES"]
        rhea_id = meta["rhea_id"]
        true_enzymes = true_map.get(reaction_id, set())
        total_positive_pairs += len(true_enzymes)

        candidates: list[tuple[str, float]] = sorted(
            sim_map.get(cano_rxn, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )[:topk]

        candidate_enzymes: set[str] = set()
        for similar_rxn, sim_score in candidates:
            for similar_reaction_id in cano_to_reaction_ids.get(similar_rxn, []):
                similar_pairs = true_pair_df[true_pair_df["reaction_id"] == similar_reaction_id]
                candidate_enzymes.update(similar_pairs["uniprot_id"].astype(str).tolist())

        candidate_enzymes &= candidate_universe
        hit_enzymes = candidate_enzymes & true_enzymes
        if hit_enzymes:
            gate_hit_reactions += 1
            covered_positive_pairs += len(hit_enzymes)

        manifest_rows.append(
            {
                "reaction_id": reaction_id,
                "rhea_id": rhea_id,
                "CANO_RXN_SMILES": cano_rxn,
                "status": "ok",
                "n_true_enzymes": len(true_enzymes),
                "n_gate_candidates": len(candidate_enzymes),
                "n_gate_positive_candidates": len(hit_enzymes),
                "gate_hit": bool(hit_enzymes),
                "positive_pair_coverage": (len(hit_enzymes) / len(true_enzymes)) if true_enzymes else None,
                "topk_similar_reactions": json.dumps([rxn for rxn, _ in candidates], ensure_ascii=False),
                "topk_similarity_scores": json.dumps([float(score) for _, score in candidates], ensure_ascii=False),
            }
        )

        if not hit_enzymes:
            continue

        for enzyme_id in sorted(candidate_enzymes):
            label = 1 if enzyme_id in true_enzymes else 0
            source_rows = candidate_df[candidate_df["uniprot_id"].astype(str) == enzyme_id]
            if source_rows.empty:
                sequence = ""
                enzyme_name = enzyme_id
            else:
                sequence = coerce_text(source_rows.iloc[0]["sequence"])
                enzyme_name = coerce_text(source_rows.iloc[0]["enzyme_id"]) or enzyme_id
            gated_pair_rows.append(
                {
                    "reaction_id": reaction_id,
                    "rhea_id": rhea_id,
                    "reaction_smiles": meta.get("CANO_RXN_SMILES") or "",
                    "CANO_RXN_SMILES": meta.get("CANO_RXN_SMILES") or "",
                    "enzyme_id": enzyme_name,
                    "uniprot_id": enzyme_id,
                    "UniprotID": enzyme_id,
                    "sequence": sequence,
                    "Sequence": sequence,
                    "label": label,
                    "Label": label,
                }
            )

    no_smiles_df = reaction_df[reaction_df["status"] != "ok"].copy()
    if not no_smiles_df.empty:
        for _, row in no_smiles_df.iterrows():
            reaction_id = coerce_text(row["reaction_id"])
            manifest_rows.append(
                {
                    "reaction_id": reaction_id,
                    "rhea_id": coerce_text(row["rhea_id"]),
                    "CANO_RXN_SMILES": "",
                    "status": "no_smiles",
                    "n_true_enzymes": int(row.get("n_true_enzymes", 0)),
                    "n_gate_candidates": 0,
                    "n_gate_positive_candidates": 0,
                    "gate_hit": False,
                    "positive_pair_coverage": None,
                    "topk_similar_reactions": json.dumps([], ensure_ascii=False),
                    "topk_similarity_scores": json.dumps([], ensure_ascii=False),
                }
            )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("reaction_id", kind="mergesort").reset_index(drop=True)
    gated_pair_df = pd.DataFrame(gated_pair_rows).sort_values(["reaction_id", "uniprot_id"], kind="mergesort").reset_index(drop=True)

    write_table(manifest_df, output_manifest, sep=",")
    write_table(gated_pair_df, output_pairs, sep=",")

    summary = {
        "n_query_reactions": int(len(manifest_df)),
        "n_reactions_with_positive_label": int((manifest_df["n_true_enzymes"] > 0).sum()),
        "n_reactions_with_smiles": int((manifest_df["status"] == "ok").sum()),
        "n_reactions_without_smiles": int((manifest_df["status"] == "no_smiles").sum()),
        "n_reactions_with_gate_hit": int(((manifest_df["status"] == "ok") & (manifest_df["gate_hit"])).sum()),
        "n_reactions_without_gate_hit": int(((manifest_df["status"] == "ok") & (~manifest_df["gate_hit"])).sum()),
        "n_true_enzymes_total": int(total_positive_pairs_all),
        "n_true_enzymes_covered_by_gate": int(covered_positive_pairs),
        "candidate_universe_total": int(len(candidate_df)),
        "candidate_universe_known_positive": int(len(candidate_universe & set(true_pair_df["uniprot_id"].astype(str)))),
        "n_gate_candidate_pairs": int(len(gated_pair_df)),
        "positive_pair_coverage": float(covered_positive_pairs / total_positive_pairs) if total_positive_pairs else 0.0,
        "positive_pair_coverage_all_reactions": float(covered_positive_pairs / total_positive_pairs_all) if total_positive_pairs_all else 0.0,
        "mean_positive_pair_coverage_per_reaction": float(manifest_df.loc[manifest_df["status"] == "ok", "positive_pair_coverage"].dropna().astype(float).mean())
        if manifest_df["positive_pair_coverage"].notna().any()
        else 0.0,
        "output_manifest": str(output_manifest),
        "output_pairs": str(output_pairs),
        "topk": int(topk),
    }
    safe_json_dump(summary, output_summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build similarity-gated terpene Rhea screening pairs for all known reactions.")
    parser.add_argument("--positive_path", type=str, default=str(DEFAULT_POSITIVE_PATH))
    parser.add_argument("--candidate_path", type=str, default=str(DEFAULT_CANDIDATE_PATH))
    parser.add_argument("--output_pairs", type=str, default=str(DEFAULT_OUTPUT_PAIRS))
    parser.add_argument("--output_manifest", type=str, default=str(DEFAULT_OUTPUT_MANIFEST))
    parser.add_argument("--output_summary", type=str, default=str(DEFAULT_OUTPUT_SUMMARY))
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    summary = build_all_rhea_gate_pairs(
        positive_path=Path(args.positive_path),
        candidate_path=Path(args.candidate_path),
        output_pairs=Path(args.output_pairs),
        output_manifest=Path(args.output_manifest),
        output_summary=Path(args.output_summary),
        topk=args.topk,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
