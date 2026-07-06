from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explorations.terpene_screen.common import (
    SOURCE_FILES,
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    canonicalize_reaction_smiles,
    coerce_text,
    dedupe_preserve_order,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_table,
)


def _normalize_candidate_records(df: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = identify_terpene_columns(df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    seq_col = columns["sequence"]["column"]

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, row in df.iterrows():
        raw_id = coerce_text(row.get(id_col)) if id_col else ""
        uniprot_id = parse_uniprot_id(raw_id)
        sequence = coerce_text(row.get(seq_col)) if seq_col else ""
        if not uniprot_id:
            failed.append(
                {
                    "source_file": str(source_path),
                    "row_index": int(idx),
                    "raw_id": raw_id,
                    "resolved_uniprot_id": "",
                    "reason": "unresolved_uniprot_id",
                    "sequence_present": bool(sequence),
                }
            )
            continue
        if not sequence:
            failed.append(
                {
                    "source_file": str(source_path),
                    "row_index": int(idx),
                    "raw_id": raw_id,
                    "resolved_uniprot_id": uniprot_id,
                    "reason": "missing_sequence",
                    "sequence_present": False,
                }
            )
            continue
        if uniprot_id in seen:
            continue
        seen.add(uniprot_id)
        rows.append(
            {
                "enzyme_id": raw_id or uniprot_id,
                "uniprot_id": uniprot_id,
                "sequence": sequence,
            }
        )

    records = pd.DataFrame(rows)
    if records.empty:
        records = pd.DataFrame(columns=["enzyme_id", "uniprot_id", "sequence"])
    failures = pd.DataFrame(failed)
    if failures.empty:
        failures = pd.DataFrame(
            columns=["source_file", "row_index", "raw_id", "resolved_uniprot_id", "reason", "sequence_present"]
        )
    return records, failures


def _normalize_positive_labels(df: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = identify_terpene_columns(df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    rhea_col = columns["rhea_id"]["column"]

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for idx, row in df.iterrows():
        raw_id = coerce_text(row.get(id_col)) if id_col else ""
        uniprot_id = parse_uniprot_id(raw_id)
        rhea_id = coerce_text(row.get(rhea_col)) if rhea_col else ""
        if not uniprot_id or not rhea_id:
            failed.append(
                {
                    "source_file": str(source_path),
                    "row_index": int(idx),
                    "raw_id": raw_id,
                    "resolved_uniprot_id": uniprot_id or "",
                    "reason": "unresolved_mapping",
                    "sequence_present": bool(coerce_text(row.get(columns["sequence"]["column"])) if columns["sequence"]["column"] else ""),
                }
            )
            continue
        key = (rhea_id, uniprot_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "rhea_id": rhea_id,
                "uniprot_id": uniprot_id,
            }
        )

    records = pd.DataFrame(rows)
    if records.empty:
        records = pd.DataFrame(columns=["rhea_id", "uniprot_id"])
    failures = pd.DataFrame(failed)
    if failures.empty:
        failures = pd.DataFrame(
            columns=["source_file", "row_index", "raw_id", "resolved_uniprot_id", "reason", "sequence_present"]
        )
    return records, failures


def _load_selected_reactions(path: Path) -> pd.DataFrame:
    df = read_table(path)
    columns = identify_terpene_columns(df)
    rhea_col = columns["rhea_id"]["column"]
    rxn_col = columns["reaction_smiles"]["column"]
    if rhea_col is None:
        raise ValueError(f"Could not detect Rhea ID column in {path}")
    if rxn_col is None:
        raise ValueError(f"Could not detect reaction SMILES column in {path}")

    records: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        raw_rxn = coerce_text(row.get(rxn_col))
        records.append(
            {
                "reaction_id": f"reaction_{idx + 1:02d}",
                "rhea_id": coerce_text(row.get(rhea_col)),
                "reaction_smiles": raw_rxn,
                "CANO_RXN_SMILES": canonicalize_reaction_smiles(raw_rxn) or raw_rxn,
            }
        )
    return pd.DataFrame(records)


def build_pairs(
    positive_path: Path,
    candidate_path: Path,
    selected_path: Path,
    output_path: Path,
    data_mirror_path: Path,
    failed_mapping_path: Path,
) -> dict[str, Any]:
    candidate_df_raw = read_table(candidate_path)
    positive_df_raw = read_table(positive_path)
    selected_df = _load_selected_reactions(selected_path)

    candidate_df, candidate_failures = _normalize_candidate_records(candidate_df_raw, candidate_path)
    positive_map_df, positive_failures = _normalize_positive_labels(positive_df_raw, positive_path)

    failed_df = pd.concat([candidate_failures, positive_failures], ignore_index=True)
    if failed_df.empty:
        failed_df = pd.DataFrame(
            columns=["source_file", "row_index", "raw_id", "resolved_uniprot_id", "reason", "sequence_present"]
        )
    write_table(failed_df, failed_mapping_path, sep="\t")

    candidate_df["__key"] = 1
    selected_df["__key"] = 1
    pair_df = selected_df.merge(candidate_df, on="__key", how="outer").drop(columns="__key")
    for column in ["enzyme_id", "uniprot_id", "sequence"]:
        if column not in pair_df.columns:
            pair_df[column] = ""
    pair_df = pair_df.merge(
        positive_map_df.assign(label=1),
        on=["rhea_id", "uniprot_id"],
        how="left",
    )
    pair_df["label"] = pair_df["label"].fillna(0).astype(int)
    pair_df["Label"] = pair_df["label"]
    pair_df["enzyme_id"] = pair_df["enzyme_id"].fillna(pair_df["uniprot_id"])
    pair_df["sequence"] = pair_df["sequence"].fillna("")
    pair_df["UniprotID"] = pair_df["uniprot_id"]
    pair_df["Sequence"] = pair_df["sequence"]
    pair_df["reaction_smiles"] = pair_df["reaction_smiles"]
    pair_df["CANO_RXN_SMILES"] = pair_df["CANO_RXN_SMILES"]

    ordered_columns = [
        "reaction_id",
        "rhea_id",
        "reaction_smiles",
        "CANO_RXN_SMILES",
        "enzyme_id",
        "uniprot_id",
        "UniprotID",
        "sequence",
        "Sequence",
        "label",
        "Label",
    ]
    pair_df = pair_df[ordered_columns]
    pair_df = pair_df.sort_values(["reaction_id", "uniprot_id"], kind="mergesort").reset_index(drop=True)

    write_table(pair_df, output_path, sep=",")
    write_table(pair_df, data_mirror_path, sep=",")

    summary = {
        "positive_label_rows": int(len(positive_df_raw)),
        "positive_label_pairs": int(len(positive_map_df)),
        "candidate_rows": int(len(candidate_df_raw)),
        "candidate_unique_enzymes": int(len(candidate_df)),
        "selected_reactions": int(len(selected_df)),
        "pair_rows": int(len(pair_df)),
        "expected_pair_rows": int(len(candidate_df) * len(selected_df)),
        "positive_pairs_in_pairs": int(pair_df["label"].sum()),
        "negative_pairs_in_pairs": int((pair_df["label"] == 0).sum()),
        "unresolved_mapping_rows": int(len(failed_df)),
        "output_csv": str(output_path),
        "data_mirror_csv": str(data_mirror_path),
        "failed_id_mapping": str(failed_mapping_path),
    }
    safe_json_dump(summary, output_path.with_suffix(".json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build terpene synthase screening pairs.")
    parser.add_argument("--positive_labels", default=str(SOURCE_FILES["positive_labels"]))
    parser.add_argument("--candidate_enzymes", default=str(SOURCE_FILES["candidate_enzymes"]))
    parser.add_argument("--selected_reactions", default=str(SOURCE_FILES["selected_reactions"]))
    parser.add_argument("--output_csv", default=str(TERPENE_RESULTS_DIR / "terpene_candidate_pairs.csv"))
    parser.add_argument("--data_mirror_csv", default=str(TERPENE_DATA_DIR / "terpene_candidate_pairs.csv"))
    parser.add_argument("--failed_id_mapping", default=str(TERPENE_RESULTS_DIR / "failed_id_mapping.tsv"))
    args = parser.parse_args()

    summary = build_pairs(
        positive_path=Path(args.positive_labels),
        candidate_path=Path(args.candidate_enzymes),
        selected_path=Path(args.selected_reactions),
        output_path=Path(args.output_csv),
        data_mirror_path=Path(args.data_mirror_csv),
        failed_mapping_path=Path(args.failed_id_mapping),
    )
    print(summary)


if __name__ == "__main__":
    main()
