from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.active.pocket_robustness.adapters.pocket_manifest import (
    PocketRecord,
    make_pocket_global_id,
    write_manifest,
)


ENZYME_ID_COLUMNS = ["UniprotID", "uniprot_id", "enzyme_id", "protein_id"]
STRUCTURE_SUFFIXES = [".pdb", ".cif", ".mmcif", ".ent"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column(column): column for column in columns}
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _read_input_enzyme_ids(input_csv: Path) -> list[str]:
    df = pd.read_csv(input_csv)
    column = _find_column(list(df.columns), ENZYME_ID_COLUMNS)
    if column is None:
        raise ValueError(
            f"Could not find enzyme id column in {input_csv}. "
            f"Accepted columns: {ENZYME_ID_COLUMNS}"
        )
    enzyme_ids = []
    seen = set()
    for value in df[column].dropna().astype(str):
        enzyme_id = value.strip()
        if enzyme_id and enzyme_id not in seen:
            enzyme_ids.append(enzyme_id)
            seen.add(enzyme_id)
    return enzyme_ids


def _find_structure(enzyme_id: str, structure_dir: Path) -> Path | None:
    for suffix in STRUCTURE_SUFFIXES:
        candidate = structure_dir / f"{enzyme_id}{suffix}"
        if candidate.exists():
            return candidate
    for suffix in STRUCTURE_SUFFIXES:
        matches = sorted(structure_dir.rglob(f"{enzyme_id}{suffix}"))
        if matches:
            return matches[0]
    return None


def _find_p2rank_executable(p2rank_home: Path) -> Path | None:
    if p2rank_home.is_file():
        return p2rank_home
    for candidate in [
        p2rank_home / "distro" / "prank",
        p2rank_home / "distro" / "prank.sh",
        p2rank_home / "prank",
        p2rank_home / "prank.sh",
        p2rank_home / "bin" / "prank",
        p2rank_home / "bin" / "prank.sh",
    ]:
        if candidate.exists():
            return candidate
    return None


def _run_command(
    command: list[str],
    summary: dict[str, Any],
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"[cmd] {shlex.join(command)}")
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        cwd=str(cwd) if cwd else None,
    )
    summary.setdefault("commands", []).append(
        {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", skipinitialspace=True)


def _find_prediction_csv(raw_dir: Path, enzyme_id: str) -> Path | None:
    prediction_dirs = [raw_dir / "predictions", raw_dir]
    for directory in prediction_dirs:
        if not directory.exists():
            continue
        exact = directory / f"{enzyme_id}_predictions.csv"
        if exact.exists():
            return exact
        matches = sorted(directory.glob(f"{enzyme_id}*predictions*.csv"))
        if matches:
            return matches[0]
    matches = sorted(raw_dir.rglob("*predictions*.csv"))
    return matches[0] if matches else None


def _find_residue_csv(raw_dir: Path, enzyme_id: str) -> Path | None:
    prediction_dirs = [raw_dir / "predictions", raw_dir]
    for directory in prediction_dirs:
        if not directory.exists():
            continue
        exact = directory / f"{enzyme_id}_residues.csv"
        if exact.exists():
            return exact
        matches = sorted(directory.glob(f"{enzyme_id}*residues*.csv"))
        if matches:
            return matches[0]
    matches = sorted(raw_dir.rglob("*residues*.csv"))
    return matches[0] if matches else None


def _parse_residue_csv(residue_csv: Path | None) -> dict[int, dict[str, Any]]:
    if residue_csv is None or not residue_csv.exists():
        return {}
    df = _read_csv(residue_csv)
    if df.empty:
        return {}

    rank_col = _find_column(list(df.columns), ["pocket", "pocket_rank", "rank"])
    chain_col = _find_column(list(df.columns), ["chain", "chain_id"])
    residue_col = _find_column(list(df.columns), ["residue_label", "residue", "residue_id", "residue_number"])
    if rank_col is None or residue_col is None:
        return {}

    result: dict[int, dict[str, Any]] = {}
    for _, row in df.iterrows():
        rank = _optional_int(row[rank_col])
        if rank is None:
            continue
        chain = str(row[chain_col]).strip() if chain_col else ""
        residue = str(row[residue_col]).strip()
        result.setdefault(rank, {"labels": [], "rows": []})
        # EnzymeCAGE's feature code expects comma-separated numeric residue ids.
        # Keep chain separately in residue_rows for cropping, but do not include
        # chain prefixes in the manifest-level pocket_residues field.
        parsed = _parse_residue_label(residue)
        result[rank]["labels"].append(str(parsed[0]) if parsed else residue)
        result[rank]["rows"].append({"chain": chain, "residue_label": residue})
    return result


def _parse_predictions(
    prediction_csv: Path,
    residue_by_rank: dict[int, dict[str, Any]],
    top_k: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    df = _read_csv(prediction_csv)
    if df.empty:
        return []

    columns = list(df.columns)
    rank_col = _find_column(columns, ["rank", "pocket_rank", "prediction_rank"])
    score_col = _find_column(columns, ["score", "probability", "confidence", "pocket_score", "prob"])
    center_x_col = _find_column(columns, ["center_x", "x", "pocket_center_x"])
    center_y_col = _find_column(columns, ["center_y", "y", "pocket_center_y"])
    center_z_col = _find_column(columns, ["center_z", "z", "pocket_center_z"])
    residues_col = _find_column(columns, ["residue_ids", "residues", "pocket_residues", "residue_list"])

    if rank_col is None:
        warning = f"No rank column found in {prediction_csv}; using row order."
        print(f"[warning] {warning}")
        warnings.append(warning)
    if score_col is None:
        warning = f"No score column found in {prediction_csv}; pocket_score will be empty."
        print(f"[warning] {warning}")
        warnings.append(warning)

    pockets = []
    for index, row in df.iterrows():
        rank = _optional_int(row[rank_col]) if rank_col else None
        rank = rank if rank is not None else int(index) + 1
        residue_labels = ""
        residue_rows = []
        if rank in residue_by_rank:
            residue_labels = ",".join(residue_by_rank[rank]["labels"])
            residue_rows = residue_by_rank[rank]["rows"]
        elif residues_col:
            residue_labels = str(row[residues_col])

        pockets.append(
            {
                "pocket_rank": rank,
                "pocket_score": _optional_float(row[score_col]) if score_col else None,
                "pocket_center_x": _optional_float(row[center_x_col]) if center_x_col else None,
                "pocket_center_y": _optional_float(row[center_y_col]) if center_y_col else None,
                "pocket_center_z": _optional_float(row[center_z_col]) if center_z_col else None,
                "pocket_residues": residue_labels,
                "residue_rows": residue_rows,
            }
        )

    pockets.sort(key=lambda item: int(item["pocket_rank"]))
    return pockets[:top_k]


def _parse_residue_label(label: str) -> tuple[int, str] | None:
    match = re.match(r"^\s*(-?\d+)([A-Za-z]?)\s*$", str(label))
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def _try_crop_pocket(
    structure_path: Path,
    residue_rows: list[dict[str, str]],
    output_path: Path,
    warnings: list[str],
) -> str:
    if not residue_rows:
        return "no residue rows available"
    try:
        from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
    except Exception as exc:
        return f"Biopython unavailable for cropping: {exc}"

    selected = set()
    for row in residue_rows:
        parsed = _parse_residue_label(row["residue_label"])
        if parsed is None:
            continue
        residue_number, insertion_code = parsed
        selected.add((row.get("chain", ""), residue_number, insertion_code))

    if not selected:
        return "no parseable residue labels available"

    class PocketSelect(Select):
        def accept_residue(self, residue):  # type: ignore[no-untyped-def]
            chain = residue.get_parent().id
            _, residue_number, insertion_code = residue.id
            return (chain, residue_number, insertion_code.strip()) in selected

    parser = PDBParser(QUIET=True) if structure_path.suffix.lower() == ".pdb" else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(structure_path.stem, str(structure_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path), PocketSelect())
        if output_path.exists() and output_path.stat().st_size > 0:
            return ""
    except Exception as exc:
        return f"cropping failed: {exc}"
    warnings.append(f"Cropped pocket file was empty for {structure_path}; using placeholder.")
    return "cropped pocket file was empty"


def _write_pocket_pdb(
    enzyme_id: str,
    structure_path: Path,
    pocket_rank: int,
    residue_rows: list[dict[str, str]],
    pockets_dir: Path,
    warnings: list[str],
) -> tuple[Path, str]:
    pocket_path = pockets_dir / f"{_safe_name(enzyme_id)}__p2rank__rank{pocket_rank}.pdb"
    if pocket_path.exists() and pocket_path.stat().st_size > 0:
        if structure_path.exists() and pocket_path.stat().st_size >= structure_path.stat().st_size:
            return pocket_path, "full_structure_placeholder"
        return pocket_path, "cropped_pocket"
    reason = _try_crop_pocket(structure_path, residue_rows, pocket_path, warnings)
    if not reason:
        return pocket_path, "cropped_pocket"

    # TODO: replace full-structure placeholder with residue-level pocket cropping where
    # P2Rank residue labels cannot be mapped robustly.
    shutil.copy2(structure_path, pocket_path)
    warning = f"Using full-structure placeholder for {enzyme_id} rank {pocket_rank}: {reason}"
    print(f"[warning] {warning}")
    warnings.append(warning)
    return pocket_path, "full_structure_placeholder"


def run_adapter(
    input_csv: Path,
    structure_dir: Path,
    p2rank_home: Path,
    output_dir: Path,
    top_k: int,
    threads: int,
    run_id: str,
    java_home: str | None = None,
) -> dict[str, Any]:
    raw_dir = output_dir / "raw_p2rank"
    pockets_dir = output_dir / "pockets"
    manifests_dir = output_dir / "manifests"
    for directory in [raw_dir, pockets_dir, manifests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": _now(),
        "status": "initialized",
        "input_csv": str(input_csv),
        "structure_dir": str(structure_dir),
        "p2rank_home": str(p2rank_home),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "threads": threads,
        "commands": [],
        "warnings": [],
        "errors": [],
        "generated_files": [],
        "n_enzymes": 0,
        "n_records": 0,
    }

    manifest_path = manifests_dir / "pocket_manifest.csv"
    records: list[PocketRecord] = []

    def process_enzyme(enzyme_id: str) -> list[PocketRecord]:
        local_records: list[PocketRecord] = []
        structure_path = _find_structure(enzyme_id, structure_dir)
        if structure_path is None:
            warning = f"No structure file found for enzyme {enzyme_id} in {structure_dir}."
            print(f"[warning] {warning}")
            summary["warnings"].append(warning)
            return local_records

        enzyme_raw_root = raw_dir / _safe_name(enzyme_id)
        enzyme_raw_root.mkdir(parents=True, exist_ok=True)
        dataset_path = enzyme_raw_root / "p2rank_input.ds"
        dataset_path.write_text(str(structure_path.resolve()) + "\n", encoding="utf-8")
        raw_output = enzyme_raw_root / "raw"
        existing_prediction_csv = _find_prediction_csv(raw_output, enzyme_id)
        if existing_prediction_csv is not None:
            warning = f"Reusing existing P2Rank output for {enzyme_id} from {raw_output}."
            print(f"[info] {warning}")
            summary["warnings"].append(warning)
            residue_csv = _find_residue_csv(raw_output, enzyme_id)
            residue_by_rank = _parse_residue_csv(residue_csv)
            pockets = _parse_predictions(
                existing_prediction_csv,
                residue_by_rank=residue_by_rank,
                top_k=top_k,
                warnings=summary["warnings"],
            )
            for pocket in pockets:
                pocket_rank = int(pocket["pocket_rank"])
                pocket_path, pocket_mode = _write_pocket_pdb(
                    enzyme_id=enzyme_id,
                    structure_path=structure_path,
                    pocket_rank=pocket_rank,
                    residue_rows=pocket["residue_rows"],
                    pockets_dir=pockets_dir,
                    warnings=summary["warnings"],
                )
                local_records.append(
                    PocketRecord(
                        run_id=run_id,
                        enzyme_id=enzyme_id,
                        structure_path=str(structure_path),
                        pocket_method="p2rank_topk",
                        pocket_source="p2rank",
                        pocket_rank=pocket_rank,
                        pocket_global_id=make_pocket_global_id(enzyme_id, "p2rank", pocket_rank),
                        pocket_score=pocket["pocket_score"],
                        pocket_center_x=pocket["pocket_center_x"],
                        pocket_center_y=pocket["pocket_center_y"],
                        pocket_center_z=pocket["pocket_center_z"],
                        pocket_residues=str(pocket["pocket_residues"]),
                        pocket_pdb_path=str(pocket_path),
                        source_raw_dir=str(raw_output),
                        pocket_pdb_mode=pocket_mode,
                    )
                )
            return local_records

        command = [
            str(p2rank_executable),
            "predict",
            "-threads",
            "1",
            "-c",
            "alphafold",
            "-visualizations",
            "0",
            "-o",
            str(enzyme_raw_root / "raw"),
            str(dataset_path),
        ]
        completed = _run_command(command, summary, env=env, cwd=p2rank_executable.parent)
        if completed.returncode != 0:
            error = f"P2Rank failed for {enzyme_id} with return code {completed.returncode}."
            print(f"[warning] {error}")
            summary["errors"].append({"enzyme_id": enzyme_id, "error": error})
            return local_records

        raw_output = enzyme_raw_root / "raw"
        prediction_csv = _find_prediction_csv(raw_output, enzyme_id)
        residue_csv = _find_residue_csv(raw_output, enzyme_id)
        if prediction_csv is None:
            warning = f"No P2Rank prediction CSV found for {enzyme_id} under {raw_output}."
            print(f"[warning] {warning}")
            summary["warnings"].append(warning)
            return local_records

        residue_by_rank = _parse_residue_csv(residue_csv)
        pockets = _parse_predictions(
            prediction_csv,
            residue_by_rank=residue_by_rank,
            top_k=top_k,
            warnings=summary["warnings"],
        )

        for pocket in pockets:
            pocket_rank = int(pocket["pocket_rank"])
            pocket_path, pocket_mode = _write_pocket_pdb(
                enzyme_id=enzyme_id,
                structure_path=structure_path,
                pocket_rank=pocket_rank,
                residue_rows=pocket["residue_rows"],
                pockets_dir=pockets_dir,
                warnings=summary["warnings"],
            )
            local_records.append(
                PocketRecord(
                    run_id=run_id,
                    enzyme_id=enzyme_id,
                    structure_path=str(structure_path),
                    pocket_method="p2rank_topk",
                    pocket_source="p2rank",
                    pocket_rank=pocket_rank,
                    pocket_global_id=make_pocket_global_id(enzyme_id, "p2rank", pocket_rank),
                    pocket_score=pocket["pocket_score"],
                    pocket_center_x=pocket["pocket_center_x"],
                    pocket_center_y=pocket["pocket_center_y"],
                    pocket_center_z=pocket["pocket_center_z"],
                    pocket_residues=str(pocket["pocket_residues"]),
                    pocket_pdb_path=str(pocket_path),
                    source_raw_dir=str(raw_output),
                    pocket_pdb_mode=pocket_mode,
                )
            )
        return local_records

    try:
        enzyme_ids = _read_input_enzyme_ids(input_csv)
    except Exception as exc:
        summary["status"] = "failed_input_parse"
        summary["errors"].append(str(exc))
        write_manifest(records, manifest_path)
        summary["generated_files"].append(str(manifest_path))
        _write_summary(summary, output_dir)
        return summary

    summary["n_enzymes"] = len(enzyme_ids)
    p2rank_executable = _find_p2rank_executable(p2rank_home)
    if p2rank_executable is None:
        warning = f"Could not find P2Rank executable under {p2rank_home}."
        print(f"[warning] {warning}")
        summary["status"] = "blocked_p2rank_missing"
        summary["warnings"].append(warning)
        write_manifest(records, manifest_path)
        summary["generated_files"].append(str(manifest_path))
        _write_summary(summary, output_dir)
        return summary

    env = os.environ.copy()
    if java_home:
        env["JAVA_HOME"] = str(Path(java_home).resolve())

    if threads > 1 and len(enzyme_ids) > 1:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_enzyme = {executor.submit(process_enzyme, enzyme_id): enzyme_id for enzyme_id in enzyme_ids}
            for future in as_completed(future_to_enzyme):
                records.extend(future.result())
    else:
        for enzyme_id in enzyme_ids:
            records.extend(process_enzyme(enzyme_id))

    write_manifest(records, manifest_path)
    summary["n_records"] = len(records)
    summary["generated_files"].append(str(manifest_path))
    summary["status"] = "completed" if records else "partial_no_pockets"
    _write_summary(summary, output_dir)
    return summary


def _write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary["generated_files"] = sorted(set([*summary.get("generated_files", []), str(summary_path)]))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P2Rank top-k pocket extraction and write a pocket manifest.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--p2rank_home", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--java_home")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_adapter(
        input_csv=Path(args.input_csv),
        structure_dir=Path(args.structure_dir),
        p2rank_home=Path(args.p2rank_home),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        threads=args.threads,
        run_id=args.run_id,
        java_home=args.java_home,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
