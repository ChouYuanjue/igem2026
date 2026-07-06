from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ENZYME_COLUMNS = ["UniprotID", "uniprot_id", "uid", "protein_id", "enzyme_id"]
REACTION_COLUMNS = ["RHEA_ID", "rhea_id", "reaction_id", "rxn_id"]
REACTION_SMILES_COLUMNS = ["CANO_RXN_SMILES", "rxn_smiles", "reaction_smiles"]
STRUCTURE_COLUMNS = ["structure_path", "pdb_path", "cif_path"]
LABEL_COLUMNS = ["Label", "label", "y", "target"]


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize(column): column for column in columns}
    for candidate in candidates:
        key = _normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _dataset_csv(enzymecage_root: Path, source_dataset: str) -> Path | None:
    if source_dataset == "Enzyme-405":
        path = enzymecage_root / "dataset/internal-test-set/Enzyme-405/Enzyme-405.csv"
        return path if path.exists() else None
    if source_dataset == "Orphan-335":
        preferred = enzymecage_root / "dataset/internal-test-set/Orphan-335/Orphan-335_retrievel_cands.csv"
        fallback = enzymecage_root / "dataset/internal-test-set/Orphan-335/Orphan-335.csv"
        if preferred.exists():
            return preferred
        return fallback if fallback.exists() else None
    return None


def _build_structure_index(enzymecage_root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    full_structure_dirs = []
    preextracted_pocket_dirs = []
    dataset_root = enzymecage_root / "dataset"
    if not dataset_root.exists():
        return {}, {}

    for directory in dataset_root.rglob("*"):
        if not directory.is_dir():
            continue
        lower_parts = [part.lower() for part in directory.parts]
        name = directory.name.lower()
        if name in {"structures", "structure", "alphafold", "pdb", "cif"} or "alphafold" in lower_parts:
            full_structure_dirs.append(directory)
        if "pockets" in lower_parts or name == "pocket":
            preextracted_pocket_dirs.append(directory)

    full_index: dict[str, Path] = {}
    pocket_index: dict[str, Path] = {}
    for directory in full_structure_dirs:
        for path in directory.glob("*.pdb"):
            full_index.setdefault(path.stem, path)
        for path in directory.glob("*.cif"):
            full_index.setdefault(path.stem, path)
    for directory in preextracted_pocket_dirs:
        for path in directory.glob("*.pdb"):
            pocket_index.setdefault(path.stem, path)
        for path in directory.glob("*.cif"):
            pocket_index.setdefault(path.stem, path)
    return full_index, pocket_index


def _sample_smallset(df: pd.DataFrame, reaction_col: str, n_reactions: int, n_enzymes_per_reaction: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    reaction_ids = list(dict.fromkeys(df[reaction_col].dropna().astype(str).tolist()))
    rng.shuffle(reaction_ids)
    selected_reactions = set(reaction_ids[:n_reactions])
    sampled = df[df[reaction_col].astype(str).isin(selected_reactions)].copy()
    pieces = []
    for _, group in sampled.groupby(reaction_col, sort=False):
        if len(group) > n_enzymes_per_reaction:
            pieces.append(group.sample(n=n_enzymes_per_reaction, random_state=seed))
        else:
            pieces.append(group)
    return pd.concat(pieces, ignore_index=True) if pieces else sampled.head(0)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_sanity_label_report(
    small: pd.DataFrame,
    output_dir: Path,
    reaction_col: str,
    enzyme_col: str,
    label_col: str | None,
    n_reactions_requested: int,
    n_enzymes_per_reaction_requested: int,
    seed: int,
) -> dict[str, Any]:
    csv_path = output_dir / "sanity_label_report.csv"
    md_path = output_dir / "sanity_label_report.md"
    summary: dict[str, Any] = {
        "status": "initialized",
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "n_reactions_requested": n_reactions_requested,
        "n_enzymes_per_reaction_requested": n_enzymes_per_reaction_requested,
        "seed": seed,
    }

    if small.empty or label_col is None or label_col not in small.columns:
        summary["status"] = "blocked_missing_label_columns"
        summary["warnings"] = ["Could not identify label column in the sampled slice."]
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("# Sanity Label Report\n\nLabel column missing.\n", encoding="utf-8")
        return summary

    df = small.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    df[reaction_col] = df[reaction_col].astype(str)
    df[enzyme_col] = df[enzyme_col].astype(str)
    duplicate_pair_count = int(df.duplicated(subset=[reaction_col, enzyme_col]).sum())

    grouped = (
        df.groupby(reaction_col, dropna=False)
        .agg(
            n_candidates=(enzyme_col, "size"),
            n_positive_pairs=(label_col, "sum"),
            n_unique_enzymes=(enzyme_col, "nunique"),
        )
        .reset_index()
        .rename(columns={reaction_col: "reaction_id"})
    )
    grouped["n_positive_pairs"] = grouped["n_positive_pairs"].astype(int)
    grouped["has_positive"] = grouped["n_positive_pairs"] > 0

    summary.update(
        {
            "status": "completed",
            "n_reactions": int(grouped.shape[0]),
            "n_pairs": int(len(df)),
            "n_unique_enzymes": int(df[enzyme_col].nunique()),
            "n_positive_pairs": int((df[label_col] == 1).sum()),
            "n_valid_reactions": int(grouped["has_positive"].sum()),
            "reactions_without_positive": grouped.loc[~grouped["has_positive"], "reaction_id"].astype(str).tolist(),
            "reactions_with_less_than_10_candidates": grouped.loc[grouped["n_candidates"] < 10, "reaction_id"].astype(str).tolist(),
            "duplicate_pair_count": duplicate_pair_count,
            "topk_evaluation_scope": "Top-5/Top-10 are computed per reaction group and exclude reactions without positive pairs.",
            "generated_files": [str(csv_path), str(md_path)],
        }
    )
    grouped.to_csv(csv_path, index=False)

    lines = [
        "# Sanity Label Report",
        "",
        f"- n_reactions: {summary['n_reactions']}",
        f"- n_valid_reactions: {summary['n_valid_reactions']}",
        f"- n_pairs: {summary['n_pairs']}",
        f"- n_unique_enzymes: {summary['n_unique_enzymes']}",
        f"- n_positive_pairs: {summary['n_positive_pairs']}",
        f"- duplicate_pair_count: {summary['duplicate_pair_count']}",
        f"- reactions_without_positive: {len(summary['reactions_without_positive'])}",
        f"- reactions_with_less_than_10_candidates: {len(summary['reactions_with_less_than_10_candidates'])}",
        "",
        "Top-5/Top-10 are evaluated by reaction group, and any reaction without a positive pair is excluded from the success denominator.",
        "",
        "## Per-Reaction Summary",
        "",
        _dataframe_to_markdown(grouped),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def build_smallset(
    source_dataset: str,
    enzymecage_root: Path,
    output_dir: Path,
    n_reactions: int,
    n_enzymes_per_reaction: int,
    seed: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    structures_out = output_dir / "smallset_structures"
    structures_out.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "smallset_summary.json"
    pairs_path = output_dir / "smallset_pairs.csv"
    link_report_path = output_dir / "structure_link_report.csv"

    summary: dict[str, Any] = {
        "status": "initialized",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_dataset": source_dataset,
        "enzymecage_root": str(enzymecage_root),
        "output_dir": str(output_dir),
        "n_reactions_requested": n_reactions,
        "n_enzymes_per_reaction_requested": n_enzymes_per_reaction,
        "seed": seed,
        "fields": {},
        "warnings": [],
        "generated_files": [],
    }

    source_csv = _dataset_csv(enzymecage_root, source_dataset)
    if source_csv is None:
        summary["status"] = "derived_smallset_blocked"
        summary["blocked_reason"] = "blocked_missing_source_dataset"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    df = pd.read_csv(source_csv)
    columns = list(df.columns)
    enzyme_col = _find_column(columns, ENZYME_COLUMNS)
    reaction_col = _find_column(columns, REACTION_COLUMNS)
    reaction_smiles_col = _find_column(columns, REACTION_SMILES_COLUMNS)
    structure_col = _find_column(columns, STRUCTURE_COLUMNS)
    label_col = _find_column(columns, LABEL_COLUMNS)
    summary["source_csv"] = str(source_csv)
    summary["source_columns"] = columns
    summary["fields"] = {
        "enzyme": enzyme_col,
        "reaction": reaction_col,
        "reaction_smiles": reaction_smiles_col,
        "structure": structure_col,
        "label": label_col,
    }

    if enzyme_col is None:
        summary["status"] = "derived_smallset_blocked"
        summary["blocked_reason"] = "blocked_missing_structure_mapping"
        summary["warnings"].append("No enzyme id column could be identified.")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if reaction_col is None or reaction_smiles_col is None:
        summary["status"] = "derived_smallset_blocked"
        summary["blocked_reason"] = "blocked_missing_reaction_columns"
        summary["warnings"].append("No reaction id and/or reaction SMILES column could be identified.")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    small = _sample_smallset(df, reaction_col, n_reactions, n_enzymes_per_reaction, seed)
    sanity_summary = _write_sanity_label_report(
        small=small,
        output_dir=output_dir,
        reaction_col=reaction_col,
        enzyme_col=enzyme_col,
        label_col=label_col,
        n_reactions_requested=n_reactions,
        n_enzymes_per_reaction_requested=n_enzymes_per_reaction,
        seed=seed,
    )
    summary["sanity_label_report"] = sanity_summary
    full_index, pocket_index = _build_structure_index(enzymecage_root)
    rows = []
    link_rows = []

    for _, row in small.iterrows():
        enzyme_id = str(row[enzyme_col])
        source_structure: Path | None = None
        source_mode = "missing"
        if structure_col and pd.notna(row.get(structure_col)):
            candidate = Path(str(row[structure_col]))
            candidate = candidate if candidate.is_absolute() else enzymecage_root / candidate
            if candidate.exists():
                source_structure = candidate
                source_mode = "explicit_structure_column"
        if source_structure is None and enzyme_id in full_index:
            source_structure = full_index[enzyme_id]
            source_mode = "full_structure"
        pocket_path = pocket_index.get(enzyme_id)
        if source_structure is None and pocket_path is not None:
            source_mode = "preextracted_pocket_only"

        target_structure = ""
        if source_structure is not None:
            target = structures_out / f"{enzyme_id}{source_structure.suffix}"
            if not target.exists():
                try:
                    target.symlink_to(source_structure.resolve())
                except OSError:
                    shutil.copy2(source_structure, target)
            target_structure = str(target)

        output_row = {
            "reaction_id": row[reaction_col],
            "enzyme_id": enzyme_id,
            "UniprotID": enzyme_id,
            "structure_path": target_structure,
            "CANO_RXN_SMILES": row[reaction_smiles_col],
        }
        if "sequence" in columns:
            output_row["sequence"] = row["sequence"]
        if label_col:
            output_row["label"] = row[label_col]
            output_row["Label"] = row[label_col]
        rows.append(output_row)
        link_rows.append(
            {
                "enzyme_id": enzyme_id,
                "source_mode": source_mode,
                "full_structure_path": str(source_structure) if source_structure else "",
                "preextracted_pocket_path": str(pocket_path) if pocket_path else "",
                "linked_structure_path": target_structure,
            }
        )

    pd.DataFrame(rows).to_csv(pairs_path, index=False)
    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(link_report_path, index=False)
    summary["generated_files"] = [str(pairs_path), str(link_report_path), str(structures_out), *sanity_summary.get("generated_files", [])]
    summary["n_pairs"] = int(len(rows))
    summary["n_reactions"] = int(pd.DataFrame(rows)["reaction_id"].nunique()) if rows else 0
    summary["n_full_structures_linked"] = int((link_df["source_mode"].isin(["full_structure", "explicit_structure_column"])).sum()) if not link_df.empty else 0
    summary["n_preextracted_pocket_only"] = int((link_df["source_mode"] == "preextracted_pocket_only").sum()) if not link_df.empty else 0
    summary["n_missing_structure"] = int((link_df["source_mode"] == "missing").sum()) if not link_df.empty else 0

    if rows and summary["n_full_structures_linked"] == len(rows):
        summary["status"] = "derived_smallset_completed"
    elif summary["n_preextracted_pocket_only"] > 0 and summary["n_full_structures_linked"] == 0:
        summary["status"] = "derived_smallset_blocked"
        summary["blocked_reason"] = "blocked_missing_full_structure_for_p2rank"
        summary["warnings"].append("Only pre-extracted pocket PDBs were found for the sampled enzymes; full structures are required for P2Rank/fpocket extraction.")
    else:
        summary["status"] = "derived_smallset_blocked"
        summary["blocked_reason"] = "blocked_missing_structure_mapping"
        summary["warnings"].append("Could not map every sampled enzyme to a full structure file.")

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a small EnzymeCAGE official-data slice for pocket intervention.")
    parser.add_argument("--source_dataset", required=True, choices=["Enzyme-405", "Orphan-335"])
    parser.add_argument("--enzymecage_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_reactions", type=int, default=5)
    parser.add_argument("--n_enzymes_per_reaction", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_smallset(
        source_dataset=args.source_dataset,
        enzymecage_root=Path(args.enzymecage_root),
        output_dir=Path(args.output_dir),
        n_reactions=args.n_reactions,
        n_enzymes_per_reaction=args.n_enzymes_per_reaction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
