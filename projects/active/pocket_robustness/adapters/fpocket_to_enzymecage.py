from __future__ import annotations

import argparse
import json
import re
import shutil
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, PDBParser

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
    seen = set()
    enzyme_ids = []
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


def _resolve_fpocket_bin(fpocket_bin: str) -> Path | None:
    path = Path(fpocket_bin)
    if path.exists():
        return path
    resolved = shutil.which(fpocket_bin)
    return Path(resolved) if resolved else None


def _parse_pocket_atom_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    suffix = path.suffix.lower()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if suffix in {".cif", ".mmcif"}:
            tokens = line.split()
            if len(tokens) < 12:
                continue
            try:
                serial = int(tokens[1])
            except ValueError:
                serial = len(records) + 1
            try:
                x = float(tokens[9])
                y = float(tokens[10])
                z = float(tokens[11])
            except (ValueError, IndexError):
                continue
            records.append(
                {
                    "serial": serial,
                    "atom_name": tokens[3].strip(),
                    "altloc": "",
                    "resname": tokens[5].strip(),
                    "chain_id": "" if tokens[6] in {"?", "."} else tokens[6].strip(),
                    "residue_number": tokens[7].strip(),
                    "insertion_code": "" if tokens[8] in {"?", "."} else tokens[8].strip(),
                    "x": x,
                    "y": y,
                    "z": z,
                    "occupancy": float(tokens[12]) if len(tokens) > 12 and tokens[12] not in {"?", "."} else 1.0,
                    "temp_factor": float(tokens[13]) if len(tokens) > 13 and tokens[13] not in {"?", "."} else 0.0,
                    "element": (tokens[2].strip() or tokens[3].strip()[:1]).upper(),
                }
            )
        else:
            if len(line) < 54:
                continue
            try:
                serial = int(line[6:11])
            except ValueError:
                serial = len(records) + 1
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            resname = line[17:20].strip()
            chain_id = line[21].strip()
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            occupancy = 1.0
            temp_factor = 0.0
            try:
                occupancy = float(line[54:60])
            except ValueError:
                pass
            try:
                temp_factor = float(line[60:66])
            except ValueError:
                pass
            element = line[76:78].strip() if len(line) >= 78 else atom_name[:1]
            records.append(
                {
                    "serial": serial,
                    "atom_name": atom_name,
                    "altloc": altloc,
                    "resname": resname,
                    "chain_id": chain_id,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "x": x,
                    "y": y,
                    "z": z,
                    "occupancy": occupancy,
                    "temp_factor": temp_factor,
                    "element": element.upper(),
                }
            )
    return records


def _run_command(command: list[str], summary: dict[str, Any], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print(f"[cmd] {shlex.join(command)}")
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
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


def _parse_pocket_rank(path: Path) -> int:
    match = re.search(r"pocket(\d+)", path.name)
    return int(match.group(1)) if match else 999999


def _parse_info_scores(info_path: Path | None) -> dict[int, float]:
    if info_path is None or not info_path.exists():
        return {}
    scores: dict[int, float] = {}
    current_rank: int | None = None
    for line in info_path.read_text(encoding="utf-8", errors="replace").splitlines():
        rank_match = re.search(r"Pocket\s+(\d+)", line, flags=re.I)
        if rank_match:
            current_rank = int(rank_match.group(1))
            continue
        score_match = re.search(r"(?:score|druggability score)\s*[:=]\s*(-?\d+(?:\.\d+)?)", line, flags=re.I)
        if current_rank is not None and score_match:
            scores[current_rank] = float(score_match.group(1))
    return scores


def _structure_center(path: Path) -> tuple[float | None, float | None, float | None]:
    coords = [(record["x"], record["y"], record["z"]) for record in _parse_pocket_atom_records(path)]
    if not coords:
        return None, None, None
    n = len(coords)
    return (
        sum(coord[0] for coord in coords) / n,
        sum(coord[1] for coord in coords) / n,
        sum(coord[2] for coord in coords) / n,
    )


def _structure_residue_labels(path: Path) -> str:
    residues = []
    seen = set()
    for record in _parse_pocket_atom_records(path):
        chain_id = record["chain_id"].strip()
        residue_number = str(record["residue_number"]).strip()
        insertion_code = str(record["insertion_code"]).strip()
        label = f"{chain_id}:{residue_number}{insertion_code}" if chain_id else f"{residue_number}{insertion_code}"
        if label in seen:
            continue
        seen.add(label)
        residues.append(label)
    return ",".join(residues)


def _convert_pocket_to_pdb(source: Path, target: Path) -> None:
    records = _parse_pocket_atom_records(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            try:
                residue_number = int(float(record["residue_number"]))
            except Exception:
                residue_number = 0
            handle.write(
                f"ATOM  {record['serial']:5d} {record['atom_name']:<4}{record['altloc'][:1]:1}"
                f"{record['resname']:>3} {record['chain_id'][:1]:1}{residue_number:4d}"
                f"{record['insertion_code'][:1]:1}   {record['x']:8.3f}{record['y']:8.3f}"
                f"{record['z']:8.3f}{record['occupancy']:6.2f}{record['temp_factor']:6.2f}"
                f"          {record['element'][:2]:>2}\n"
            )
        handle.write("END\n")


def _find_fpocket_output(work_dir: Path, structure_copy: Path) -> Path | None:
    expected = work_dir / f"{structure_copy.stem}_out"
    if expected.exists():
        return expected
    matches = sorted(work_dir.glob("*_out"))
    return matches[0] if matches else None


def _find_info_file(fpocket_output: Path) -> Path | None:
    matches = sorted(fpocket_output.glob("*_info.txt"))
    if matches:
        return matches[0]
    matches = sorted(fpocket_output.rglob("*info*.txt"))
    return matches[0] if matches else None


def _find_pocket_pdbs(fpocket_output: Path) -> list[Path]:
    pockets_dir = fpocket_output / "pockets"
    roots = [pockets_dir, fpocket_output] if pockets_dir.exists() else [fpocket_output]
    paths: list[Path] = []
    for root in roots:
        paths.extend(sorted(root.glob("pocket*_atm.pdb")))
        paths.extend(sorted(root.glob("pocket*_atm.cif")))
        paths.extend(sorted(root.glob("pocket*_atm.mmcif")))
        paths.extend(sorted(root.glob("pocket*.pdb")))
        paths.extend(sorted(root.glob("pocket*.cif")))
        paths.extend(sorted(root.glob("pocket*.mmcif")))
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return sorted(unique, key=_parse_pocket_rank)


def run_adapter(
    input_csv: Path,
    structure_dir: Path,
    output_dir: Path,
    top_k: int,
    run_id: str,
    fpocket_bin: str,
    threads: int,
) -> dict[str, Any]:
    raw_dir = output_dir / "raw_fpocket"
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
        "fpocket_bin": fpocket_bin,
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

    fpocket_path = _resolve_fpocket_bin(fpocket_bin)
    if fpocket_path is None:
        warning = f"fpocket command is not available: {fpocket_bin}"
        print(f"[warning] {warning}")
        summary["status"] = "blocked_fpocket_missing"
        summary["warnings"].append(warning)
        write_manifest(records, manifest_path)
        summary["generated_files"].append(str(manifest_path))
        _write_summary(summary, output_dir)
        return summary

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
    for enzyme_id in enzyme_ids:
        structure_path = _find_structure(enzyme_id, structure_dir)
        if structure_path is None:
            warning = f"No structure file found for enzyme {enzyme_id} in {structure_dir}."
            print(f"[warning] {warning}")
            summary["warnings"].append(warning)
            continue

        enzyme_raw_dir = raw_dir / _safe_name(enzyme_id)
        enzyme_raw_dir.mkdir(parents=True, exist_ok=True)
        structure_copy = enzyme_raw_dir / structure_path.name
        shutil.copy2(structure_path, structure_copy)

        fpocket_output = _find_fpocket_output(enzyme_raw_dir, structure_copy)
        pocket_files = _find_pocket_pdbs(fpocket_output) if fpocket_output else []
        if not pocket_files:
            completed = _run_command([str(fpocket_path), "-f", str(structure_copy.name)], summary, cwd=enzyme_raw_dir)
            if completed.returncode != 0:
                error = f"fpocket failed for {enzyme_id} with return code {completed.returncode}."
                print(f"[warning] {error}")
                summary["errors"].append({"enzyme_id": enzyme_id, "error": error})
                continue
            fpocket_output = _find_fpocket_output(enzyme_raw_dir, structure_copy)
            pocket_files = _find_pocket_pdbs(fpocket_output) if fpocket_output else []
        else:
            summary["warnings"].append(f"Reused existing fpocket output for {enzyme_id} from {fpocket_output}")

        if fpocket_output is None:
            warning = f"No fpocket output directory found for {enzyme_id} under {enzyme_raw_dir}."
            print(f"[warning] {warning}")
            summary["warnings"].append(warning)
            continue

        scores = _parse_info_scores(_find_info_file(fpocket_output))
        for pocket_file in pocket_files[:top_k]:
            rank = _parse_pocket_rank(pocket_file)
            output_pdb = pockets_dir / f"{_safe_name(enzyme_id)}__fpocket__rank{rank}.pdb"
            if pocket_file.suffix.lower() in {".cif", ".mmcif"}:
                try:
                    _convert_pocket_to_pdb(pocket_file, output_pdb)
                except Exception as exc:
                    summary["errors"].append({"enzyme_id": enzyme_id, "error": f"failed to convert pocket CIF {pocket_file.name}: {exc}"})
                    continue
            else:
                shutil.copy2(pocket_file, output_pdb)
            center_x, center_y, center_z = _structure_center(output_pdb)
            records.append(
                PocketRecord(
                    run_id=run_id,
                    enzyme_id=enzyme_id,
                    structure_path=str(structure_path),
                    pocket_method="fpocket_topk",
                    pocket_source="fpocket",
                    pocket_rank=rank,
                    pocket_global_id=make_pocket_global_id(enzyme_id, "fpocket", rank),
                    pocket_score=scores.get(rank),
                    pocket_center_x=center_x,
                    pocket_center_y=center_y,
                    pocket_center_z=center_z,
                    pocket_residues=_structure_residue_labels(output_pdb),
                    pocket_pdb_path=str(output_pdb),
                    source_raw_dir=str(fpocket_output),
                    pocket_pdb_mode="cropped_pocket",
                )
            )

    write_manifest(records, manifest_path)
    summary["n_records"] = len(records)
    summary["generated_files"].append(str(manifest_path))
    if records:
        summary["status"] = "completed"
    elif summary["errors"]:
        summary["status"] = "blocked_fpocket_runtime_error"
        summary["failed_step"] = "pocket_extraction"
    else:
        summary["status"] = "blocked_no_valid_pockets"
        summary["failed_step"] = "pocket_extraction"
    _write_summary(summary, output_dir)
    return summary


def _write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary["generated_files"] = sorted(set([*summary.get("generated_files", []), str(summary_path)]))
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fpocket top-k pocket extraction and write a pocket manifest.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--structure_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--fpocket_bin", required=True)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_adapter(
        input_csv=Path(args.input_csv),
        structure_dir=Path(args.structure_dir),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        run_id=args.run_id,
        fpocket_bin=args.fpocket_bin,
        threads=args.threads,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
