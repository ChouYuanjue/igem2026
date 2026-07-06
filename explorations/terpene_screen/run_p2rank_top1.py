from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explorations.terpene_screen.common import (
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    coerce_text,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    resolve_java_home,
    safe_json_dump,
    write_table,
)


POCKET_RANK = 1
P2RANK_HOME = PROJECT_ROOT / "data" / "assets" / "p2rank" / "p2rank_2.5.1"
ENZYMECAGE_P2RANK_SCRIPT = PROJECT_ROOT / "external_repos" / "EnzymeCAGE" / "scripts" / "extract_p2rank_pockets.py"
RAW_OUTPUT_ROOT = TERPENE_DATA_DIR / "p2rank_raw"
STAGE_ROOT = TERPENE_DATA_DIR / "_p2rank_stage"
POCKET_DIR = TERPENE_DATA_DIR / "pockets"
MANIFEST_CSV = TERPENE_RESULTS_DIR / "p2rank_pocket_manifest.csv"
POCKET_INFO_CSV = TERPENE_DATA_DIR / "pocket_info.csv"
FAILED_P2RANK_CSV = TERPENE_RESULTS_DIR / "failed_p2rank_pockets.csv"


def _candidate_uids(candidate_csv: Path) -> list[str]:
    df = read_table(candidate_csv)
    cols = identify_terpene_columns(df)
    uid_col = cols["uniprot_id"]["column"] or cols["enzyme_id"]["column"]
    if uid_col is None:
        raise ValueError(f"Could not detect UniProt/enzyme ID column in {candidate_csv}")
    uids = []
    for value in df[uid_col].tolist():
        uid = parse_uniprot_id(value) or coerce_text(value)
        if uid and uid not in uids:
            uids.append(uid)
    return uids


def _structure_files_for_uids(structure_dir: Path, uids: list[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for uid in uids:
        for suffix in (".cif", ".pdb", ".mmcif", ".ent"):
            candidate = structure_dir / f"{uid}{suffix}"
            if candidate.exists():
                files[uid] = candidate.resolve()
                break
    return files


def _load_existing_rows() -> pd.DataFrame:
    for path in (MANIFEST_CSV, POCKET_INFO_CSV):
        if path.exists():
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False)
            except Exception:
                continue
    return pd.DataFrame()


def _normalise_existing_manifest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    rename_map = {}
    for column in df.columns:
        normalized = column.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"uniprotid", "uniprot_id", "uid"}:
            rename_map[column] = "UniprotID"
        elif normalized.lower() == "enzyme_id":
            rename_map[column] = "enzyme_id"
        elif normalized.lower() == "pocket_path":
            rename_map[column] = "pocket_path"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _manifest_completed_uids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    if "UniprotID" in df.columns:
        uid_col = "UniprotID"
    elif "uniprot_id" in df.columns:
        uid_col = "uniprot_id"
    elif "enzyme_id" in df.columns:
        uid_col = "enzyme_id"
    else:
        return set()
    completed: set[str] = set()
    for _, row in df.iterrows():
        uid = coerce_text(row.get(uid_col))
        pocket_path = coerce_text(row.get("pocket_path")) or coerce_text(row.get("pocket_pdb_path"))
        if uid and pocket_path and Path(pocket_path).exists():
            completed.add(uid)
    return completed


def _prepare_stage_workspace(structure_files: dict[str, Path], uids: list[str]) -> tuple[Path, Path]:
    if STAGE_ROOT.exists():
        shutil.rmtree(STAGE_ROOT)
    stage_structure_dir = STAGE_ROOT / "structures"
    stage_structure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for uid in uids:
        source = structure_files[uid]
        link_path = stage_structure_dir / source.name
        try:
            os.symlink(source, link_path)
        except OSError:
            shutil.copy2(source, link_path)
        rows.append({"UniprotID": uid})
    stage_input_csv = STAGE_ROOT / "input.csv"
    pd.DataFrame(rows).to_csv(stage_input_csv, index=False)
    return stage_input_csv, stage_structure_dir


def _run_external_p2rank(stage_input_csv: Path, stage_structure_dir: Path, threads: int, java_home: Path | None) -> None:
    cmd = [
        "/home/runnel/miniconda3/envs/enzymecage/bin/python",
        str(ENZYMECAGE_P2RANK_SCRIPT),
        "--input_csv",
        str(stage_input_csv),
        "--structure_dir",
        str(stage_structure_dir),
        "--p2rank_home",
        str(P2RANK_HOME),
        "--output_dir",
        str(STAGE_ROOT),
        "--threads",
        str(threads),
    ]
    if java_home is not None:
        cmd.extend(["--java_home", str(java_home)])
    print(f"[cmd] {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def _merge_stage_results(stage_manifest_csv: Path, actual_pocket_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_pocket_dir = STAGE_ROOT / "pocket"
    if not stage_manifest_csv.exists():
        stage_df = pd.DataFrame()
    else:
        try:
            stage_df = pd.read_csv(stage_manifest_csv, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            stage_df = pd.DataFrame()
    actual_pocket_dir.mkdir(parents=True, exist_ok=True)

    if stage_pocket_dir.exists():
        for pocket_file in stage_pocket_dir.glob("*.pdb"):
            target = actual_pocket_dir / pocket_file.name
            if target.exists():
                target.unlink()
            shutil.move(str(pocket_file), str(target))

    failed_stage = STAGE_ROOT / "failed_p2rank_pockets.csv"
    if failed_stage.exists():
        try:
            failed_df = pd.read_csv(failed_stage, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            failed_df = pd.DataFrame(columns=["UniprotID", "structure_path", "error"])
    else:
        failed_df = pd.DataFrame(columns=["UniprotID", "structure_path", "error"])

    if stage_df.empty:
        empty_columns = [
            "run_id",
            "UniprotID",
            "enzyme_id",
            "structure_path",
            "pocket_method",
            "pocket_source",
            "pocket_rank",
            "pocket_global_id",
            "pocket_score",
            "pocket_probability",
            "pocket_center_x",
            "pocket_center_y",
            "pocket_center_z",
            "pocket_residues",
            "pocket_pdb_path",
            "pocket_path",
            "source_raw_dir",
            "pocket_pdb_mode",
            "status",
        ]
        return pd.DataFrame(columns=empty_columns), failed_df

    if "UniprotID" not in stage_df.columns and "uniprot_id" in stage_df.columns:
        stage_df = stage_df.rename(columns={"uniprot_id": "UniprotID"})
    if "pocket_path" in stage_df.columns:
        stage_df["pocket_path"] = stage_df["UniprotID"].astype(str).map(lambda uid: str(actual_pocket_dir / f"{uid}.pdb"))
    else:
        stage_df["pocket_path"] = stage_df["UniprotID"].astype(str).map(lambda uid: str(actual_pocket_dir / f"{uid}.pdb"))
    if "pocket_pdb_path" in stage_df.columns:
        stage_df["pocket_pdb_path"] = stage_df["pocket_path"]
    else:
        stage_df["pocket_pdb_path"] = stage_df["pocket_path"]
    if "score" in stage_df.columns and "pocket_score" not in stage_df.columns:
        stage_df["pocket_score"] = stage_df["score"]
    if "probability" in stage_df.columns and "pocket_probability" not in stage_df.columns:
        stage_df["pocket_probability"] = stage_df["probability"]
    if "center_x" in stage_df.columns and "pocket_center_x" not in stage_df.columns:
        stage_df["pocket_center_x"] = stage_df["center_x"]
    if "center_y" in stage_df.columns and "pocket_center_y" not in stage_df.columns:
        stage_df["pocket_center_y"] = stage_df["center_y"]
    if "center_z" in stage_df.columns and "pocket_center_z" not in stage_df.columns:
        stage_df["pocket_center_z"] = stage_df["center_z"]
    stage_df["status"] = "ok"
    stage_df["run_id"] = "terpene_p2rank_top1"
    if "pocket_method" not in stage_df.columns:
        stage_df["pocket_method"] = "p2rank_top1"
    if "pocket_source" not in stage_df.columns:
        stage_df["pocket_source"] = "p2rank"
    if "pocket_rank" not in stage_df.columns:
        stage_df["pocket_rank"] = POCKET_RANK
    if "pocket_global_id" not in stage_df.columns:
        stage_df["pocket_global_id"] = stage_df["UniprotID"].astype(str).map(
            lambda uid: f"{uid}__p2rank__rank{POCKET_RANK}"
        )
    if "structure_path" not in stage_df.columns:
        stage_df["structure_path"] = ""
    if "source_raw_dir" not in stage_df.columns:
        stage_df["source_raw_dir"] = str(STAGE_ROOT)
    if "pocket_pdb_mode" not in stage_df.columns:
        stage_df["pocket_pdb_mode"] = "cropped_pocket"
    return stage_df, failed_df


def run_p2rank_top1(candidate_csv: Path, structure_dir: Path, threads: int) -> dict[str, Any]:
    java_home = resolve_java_home()
    if java_home is None:
        raise RuntimeError(
            "Could not locate Java 17+. Please install a Java 17 runtime before running P2Rank."
        )

    target_uids = _candidate_uids(candidate_csv)
    existing_manifest = _normalise_existing_manifest(_load_existing_rows())
    completed_uids = _manifest_completed_uids(existing_manifest)

    available_structures = _structure_files_for_uids(structure_dir, target_uids)
    missing_structure_uids = [uid for uid in target_uids if uid not in available_structures]
    runnable_uids = [uid for uid in target_uids if uid in available_structures and uid not in completed_uids]
    new_manifest_rows: list[dict[str, Any]] = []
    new_failed_rows: list[dict[str, Any]] = []

    if runnable_uids:
        stage_input_csv, stage_structure_dir = _prepare_stage_workspace(available_structures, runnable_uids)
        _run_external_p2rank(stage_input_csv, stage_structure_dir, threads, java_home)
        stage_manifest, stage_failed = _merge_stage_results(STAGE_ROOT / "pocket_info.csv", POCKET_DIR)
        new_manifest_rows = stage_manifest.to_dict("records")
        new_failed_rows = stage_failed.to_dict("records")
    else:
        POCKET_DIR.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, Any]] = []
    if not existing_manifest.empty:
        combined_rows.extend(existing_manifest.to_dict("records"))
    combined_rows.extend(new_manifest_rows)

    manifest_df = pd.DataFrame(combined_rows)
    if not manifest_df.empty:
        if "UniprotID" not in manifest_df.columns and "uniprot_id" in manifest_df.columns:
            manifest_df["UniprotID"] = manifest_df["uniprot_id"]
        if "enzyme_id" not in manifest_df.columns and "UniprotID" in manifest_df.columns:
            manifest_df["enzyme_id"] = manifest_df["UniprotID"]
        if "status" not in manifest_df.columns:
            manifest_df["status"] = "ok"
        manifest_df = manifest_df.drop_duplicates(subset=["UniprotID"], keep="last")
        manifest_df = manifest_df.sort_values("UniprotID", kind="mergesort").reset_index(drop=True)
    else:
        manifest_df = pd.DataFrame(
            columns=[
                "run_id",
                "UniprotID",
                "enzyme_id",
                "structure_path",
                "pocket_method",
                "pocket_source",
                "pocket_rank",
                "pocket_global_id",
                "pocket_score",
                "pocket_probability",
                "pocket_center_x",
                "pocket_center_y",
                "pocket_center_z",
                "pocket_residues",
                "pocket_pdb_path",
                "pocket_path",
                "source_raw_dir",
                "pocket_pdb_mode",
                "status",
            ]
        )

    existing_failed = pd.DataFrame()
    if FAILED_P2RANK_CSV.exists():
        try:
            existing_failed = pd.read_csv(FAILED_P2RANK_CSV, dtype=str, keep_default_na=False)
        except Exception:
            existing_failed = pd.DataFrame()
    failed_df = pd.concat([existing_failed, pd.DataFrame(new_failed_rows)], ignore_index=True)
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=["UniprotID", "structure_path", "error"])

    write_table(manifest_df, MANIFEST_CSV, sep=",")
    write_table(manifest_df.rename(columns={"pocket_residues": "pocket_residues"}), POCKET_INFO_CSV, sep=",")
    write_table(failed_df, FAILED_P2RANK_CSV, sep=",")

    summary = {
        "candidate_csv": str(candidate_csv),
        "structure_dir": str(structure_dir),
        "p2rank_home": str(P2RANK_HOME),
        "raw_output_root": str(RAW_OUTPUT_ROOT),
        "pocket_dir": str(POCKET_DIR),
        "manifest_csv": str(MANIFEST_CSV),
        "pocket_info_csv": str(POCKET_INFO_CSV),
        "failed_csv": str(FAILED_P2RANK_CSV),
        "n_target_uids": int(len(target_uids)),
        "n_available_structures": int(len(available_structures)),
        "n_missing_structures": int(len(missing_structure_uids)),
        "n_completed_uids_before": int(len(completed_uids)),
        "n_new_pockets": int(len(new_manifest_rows)),
        "n_failed_new_pockets": int(len(new_failed_rows)),
        "n_manifest_rows": int(len(manifest_df)),
    }
    safe_json_dump(summary, MANIFEST_CSV.with_suffix(".json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P2Rank top-1 pocket extraction for terpene candidates.")
    parser.add_argument("--candidate_pairs", default=str(TERPENE_DATA_DIR / "terpene_candidate_pairs.csv"))
    parser.add_argument("--structure_dir", default=str(TERPENE_DATA_DIR / "structures"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    summary = run_p2rank_top1(
        candidate_csv=Path(args.candidate_pairs),
        structure_dir=Path(args.structure_dir),
        threads=args.threads,
    )
    print(summary)


if __name__ == "__main__":
    main()
