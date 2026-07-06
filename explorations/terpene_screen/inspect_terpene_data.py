from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explorations.terpene_screen.common import (
    SOURCE_FILES,
    TERPENE_RESULTS_DIR,
    canonicalize_reaction_smiles,
    coerce_text,
    count_nonempty,
    first_rows,
    identify_terpene_columns,
    is_sequence_like,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_markdown,
)


def _inspect_file(path: Path) -> dict[str, Any]:
    df = read_table(path)
    columns = identify_terpene_columns(df)
    rows = len(df)
    id_col = columns["uniprot_id"]["column"] or columns["enzyme_id"]["column"]
    seq_col = columns["sequence"]["column"]
    rhea_col = columns["rhea_id"]["column"]
    rxn_col = columns["reaction_smiles"]["column"]

    unresolved_id_rows = 0
    missing_sequence_rows = 0
    missing_rhea_rows = 0
    missing_reaction_rows = 0
    for _, row in df.iterrows():
        if id_col:
            raw_id = coerce_text(row.get(id_col))
            if not parse_uniprot_id(raw_id):
                unresolved_id_rows += 1
        if seq_col and not is_sequence_like(row.get(seq_col)):
            missing_sequence_rows += 1
        if rhea_col and not coerce_text(row.get(rhea_col)):
            missing_rhea_rows += 1
        if rxn_col and not coerce_text(row.get(rxn_col)):
            missing_reaction_rows += 1

    return {
        "path": str(path.resolve()),
        "rows": rows,
        "columns": list(df.columns),
        "head": first_rows(df, 5),
        "identified_columns": columns,
        "counts": {
            "nonempty_enzyme_id": count_nonempty(df, id_col),
            "nonempty_uniprot_id": count_nonempty(df, columns["uniprot_id"]["column"]),
            "nonempty_sequence": count_nonempty(df, seq_col),
            "nonempty_rhea_id": count_nonempty(df, rhea_col),
            "nonempty_reaction_smiles": count_nonempty(df, rxn_col),
            "unresolved_id_rows": unresolved_id_rows,
            "missing_sequence_rows": missing_sequence_rows,
            "missing_rhea_rows": missing_rhea_rows,
            "missing_reaction_rows": missing_reaction_rows,
        },
    }


def _load_selected_reactions(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    df = read_table(path)
    columns = identify_terpene_columns(df)
    rhea_col = columns["rhea_id"]["column"]
    rxn_col = columns["reaction_smiles"]["column"]
    if rhea_col is None:
        raise ValueError(f"Could not identify Rhea ID column in {path}")
    if rxn_col is None:
        raise ValueError(f"Could not identify reaction SMILES column in {path}")
    return df, {"rhea_id": rhea_col, "reaction_smiles": rxn_col}


def build_audit_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_inspections = {name: _inspect_file(path) for name, path in SOURCE_FILES.items()}

    candidate_df = read_table(SOURCE_FILES["candidate_enzymes"])
    candidate_cols = identify_terpene_columns(candidate_df)
    candidate_id_col = candidate_cols["uniprot_id"]["column"] or candidate_cols["enzyme_id"]["column"]
    candidate_sequence_col = candidate_cols["sequence"]["column"]
    candidate_rows = []
    unresolved_candidate_ids = 0
    for idx, row in candidate_df.iterrows():
        raw_id = coerce_text(row.get(candidate_id_col)) if candidate_id_col else ""
        uniprot_id = parse_uniprot_id(raw_id)
        sequence = coerce_text(row.get(candidate_sequence_col)) if candidate_sequence_col else ""
        if not uniprot_id:
            unresolved_candidate_ids += 1
            continue
        candidate_rows.append(
            {
                "row_index": int(idx),
                "enzyme_id": raw_id or uniprot_id,
                "uniprot_id": uniprot_id,
                "sequence": sequence or None,
            }
        )

    positive_df = read_table(SOURCE_FILES["positive_labels"])
    positive_cols = identify_terpene_columns(positive_df)
    positive_id_col = positive_cols["uniprot_id"]["column"] or positive_cols["enzyme_id"]["column"]
    positive_rhea_col = positive_cols["rhea_id"]["column"]
    positive_sequence_col = positive_cols["sequence"]["column"]

    positive_pairs: set[tuple[str, str]] = set()
    reaction_to_positives: dict[str, set[str]] = {}
    unresolved_positive_ids = 0
    for _, row in positive_df.iterrows():
        raw_id = coerce_text(row.get(positive_id_col)) if positive_id_col else ""
        uniprot_id = parse_uniprot_id(raw_id)
        rhea_id = coerce_text(row.get(positive_rhea_col)) if positive_rhea_col else ""
        if not uniprot_id:
            unresolved_positive_ids += 1
            continue
        if not rhea_id:
            continue
        positive_pairs.add((rhea_id, uniprot_id))
        reaction_to_positives.setdefault(rhea_id, set()).add(uniprot_id)

    selected_df, selected_cols = _load_selected_reactions(SOURCE_FILES["selected_reactions"])
    selected_reaction_rows: list[dict[str, Any]] = []
    no_positive_label = []
    for idx, row in selected_df.iterrows():
        rhea_id = coerce_text(row.get(selected_cols["rhea_id"]))
        rxn_smiles = coerce_text(row.get(selected_cols["reaction_smiles"]))
        selected_reaction_rows.append(
            {
                "reaction_id": f"reaction_{idx + 1:02d}",
                "rhea_id": rhea_id,
                "reaction_smiles": rxn_smiles,
                "canonical_reaction_smiles": canonicalize_reaction_smiles(rxn_smiles),
                "positive_enzyme_count": len(reaction_to_positives.get(rhea_id, set())),
            }
        )
        if len(reaction_to_positives.get(rhea_id, set())) == 0:
            no_positive_label.append(rhea_id)

    candidate_total = len(candidate_rows)
    expected_pairs = len(selected_reaction_rows) * candidate_total

    return {
        "project_root": str(PROJECT_ROOT),
        "source_files": source_inspections,
        "identified_columns": {
            "candidate_enzymes": candidate_cols,
            "positive_labels": positive_cols,
            "selected_reactions": selected_cols,
        },
        "selected_reactions": selected_reaction_rows,
        "summary": {
            "n_selected_reactions": len(selected_reaction_rows),
            "n_selected_reactions_with_positive_label": sum(
                1 for row in selected_reaction_rows if row["positive_enzyme_count"] > 0
            ),
            "n_selected_reactions_without_positive_label": len(no_positive_label),
            "candidate_total": candidate_total,
            "expected_pair_count": expected_pairs,
            "unresolved_candidate_id_rows": unresolved_candidate_ids,
            "unresolved_positive_id_rows": unresolved_positive_ids,
            "unresolved_total_rows": unresolved_candidate_ids + unresolved_positive_ids,
            "positive_label_row_count": len(positive_df),
            "positive_label_unique_pairs": len(positive_pairs),
            "positive_label_unique_enzymes": len({enzyme for _, enzyme in positive_pairs}),
        },
        "per_reaction_positive_counts": {
            row["rhea_id"]: row["positive_enzyme_count"] for row in selected_reaction_rows
        },
    }


def build_markdown(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = [
        "# Terpene 数据审计",
        "",
        "## 概览",
        f"- 项目根目录: `{payload['project_root']}`",
        f"- 候选酶总数: `{payload['summary']['candidate_total']}`",
        f"- 10 条 reaction 中有 positive label 的数量: `{payload['summary']['n_selected_reactions_with_positive_label']}`",
        f"- 10 条 reaction 中没有 positive label 的数量: `{payload['summary']['n_selected_reactions_without_positive_label']}`",
        f"- 预计 pair 数: `{payload['summary']['expected_pair_count']}`",
        f"- 无法识别 ID 的记录数: `{payload['summary']['unresolved_total_rows']}`",
        "",
        "## 三个输入文件",
    ]

    for name, inspection in payload["source_files"].items():
        lines.extend(
            [
                f"### {name}",
                f"- 路径: `{inspection['path']}`",
                f"- 行数: `{inspection['rows']}`",
                f"- 列名: `{', '.join(inspection['columns'])}`",
                f"- 自动识别列: `{json.dumps(inspection['identified_columns'], ensure_ascii=False)}`",
                f"- 统计: `{json.dumps(inspection['counts'], ensure_ascii=False)}`",
                "- 前 5 行:",
                "```json",
                json.dumps(inspection["head"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## 每条 reaction 的 positive enzyme 数",
            "",
        ]
    )
    for row in payload["selected_reactions"]:
        lines.append(
            f"- `{row['reaction_id']}` / `{row['rhea_id']}`: `{row['positive_enzyme_count']}`"
        )

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the terpene synthase screening data.")
    parser.add_argument("--positive_labels", default=str(SOURCE_FILES["positive_labels"]))
    parser.add_argument("--candidate_enzymes", default=str(SOURCE_FILES["candidate_enzymes"]))
    parser.add_argument("--selected_reactions", default=str(SOURCE_FILES["selected_reactions"]))
    parser.add_argument("--output_md", default=str(TERPENE_RESULTS_DIR / "data_audit.md"))
    parser.add_argument("--output_json", default=str(TERPENE_RESULTS_DIR / "data_audit.json"))
    args = parser.parse_args()

    payload = build_audit_payload(args)
    write_markdown(Path(args.output_md), build_markdown(payload))
    safe_json_dump(payload, Path(args.output_json))
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
