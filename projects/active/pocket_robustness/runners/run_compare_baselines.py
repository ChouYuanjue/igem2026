from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STOP_STEPS = [
    "clone_inspect",
    "official_eval",
    "smallset_build",
    "pocket_extraction",
    "feature_generation",
    "inference",
    "aggregation",
    "analysis",
]
ENZYME_ID_COLUMNS = ["UniprotID", "uniprot_id", "enzyme_id", "protein_id"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _normalize_column(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column(column): column for column in columns}
    for candidate in candidates:
        key = _normalize_column(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _parse_pdb_residue_numbers(pdb_path: Path) -> str:
    """Return comma-separated residue numbers from a PDB-format pocket file."""
    residue_numbers: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    try:
        with pdb_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                chain_id = line[21].strip()
                residue_number = line[22:26].strip()
                insertion_code = line[26].strip()
                if not residue_number:
                    continue
                key = (chain_id, residue_number, insertion_code)
                if key in seen:
                    continue
                seen.add(key)
                residue_numbers.append(residue_number)
    except OSError:
        return ""
    return ",".join(residue_numbers)


class ExperimentRunner:
    def __init__(self, config_path: Path, dry_run: bool = False, resume: bool = False, stop_after: str | None = None):
        self.config_path = _resolve_path(config_path)
        self.config = _load_yaml(self.config_path)
        self.dry_run = dry_run
        self.resume = resume
        self.stop_after = stop_after

        self.run_id = self.config["project"]["run_id"]
        self.working_dir = _resolve_path(self.config["data"]["working_dir"])
        self.output_dir = _resolve_path(self.config["data"]["output_dir"])
        self.current_structure_dir: Path | None = None
        self.dirs = {
            "logs": self.output_dir / "logs",
            "manifests": self.output_dir / "manifests",
            "pocket_inputs": self.output_dir / "pocket_inputs",
            "feature_inputs": self.output_dir / "feature_inputs",
            "predictions": self.output_dir / "predictions",
            "aggregation": self.output_dir / "aggregation",
            "metrics": self.output_dir / "metrics",
            "analysis": self.output_dir / "analysis",
        }
        self.commands_jsonl = self.output_dir / "commands.jsonl"
        self.command_counter = 0
        self.summary: dict[str, Any] = {
            "run_id": self.run_id,
            "baseline_name": self.config["project"].get("baseline_name", self.run_id),
            "timestamp": _now(),
            "config_path": str(self.config_path),
            "copied_config_path": str(self.output_dir / "config.yaml"),
            "working_dir": str(self.working_dir),
            "output_dir": str(self.output_dir),
            "commands_jsonl": str(self.commands_jsonl),
            "dry_run": dry_run,
            "resume": resume,
            "stop_after": stop_after,
            "commands": [],
            "external_repos": {},
            "status": "initialized",
            "failed_step": None,
            "error": None,
            "warnings": [],
            "generated_files": [],
            "key_files": {},
            "data_mode": self.config.get("data", {}).get("data_mode", self.config.get("project", {}).get("data_mode", "demo_mining")),
        }

    def setup_dirs(self) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config_path, self.output_dir / "config.yaml")
        self.commands_jsonl.write_text("", encoding="utf-8")
        self.summary["generated_files"].extend(
            [str(self.output_dir / "config.yaml"), str(self.commands_jsonl)]
        )

    def write_summary(self) -> Path:
        summary_path = self.output_dir / "run_summary.json"
        self.summary["generated_files"] = sorted(set([*self.summary["generated_files"], str(summary_path)]))
        summary_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")
        return summary_path

    def load_existing_completed_summary(self) -> dict[str, Any] | None:
        summary_path = self.output_dir / "run_summary.json"
        if not summary_path.exists():
            return None
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if summary.get("status") != "completed":
            return None
        predictions = self.dirs["predictions"] / "pocket_level_predictions.csv"
        metrics = self.dirs["metrics"] / "metrics_top5_top10.json"
        aggregation_files = list(self.dirs["aggregation"].glob("enzyme_level_*.csv"))
        if not predictions.exists() or not metrics.exists() or not aggregation_files:
            return None
        return summary

    def log_command_record(self, record: dict[str, Any]) -> None:
        self.summary["commands"].append(record)
        with self.commands_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def run_command(
        self,
        step: str,
        command: list[str],
        cwd: Path | None = None,
        check: bool = True,
        expected_output: Path | None = None,
    ) -> subprocess.CompletedProcess[Any] | None:
        self.command_counter += 1
        stdout_path = self.dirs["logs"] / f"{self.command_counter:03d}_{step}.stdout.log"
        stderr_path = self.dirs["logs"] / f"{self.command_counter:03d}_{step}.stderr.log"
        command_text = shlex.join(command)
        print(f"[cmd:{step}] {command_text}")

        if self.resume and expected_output and expected_output.exists():
            record = {
                "step": step,
                "command": command,
                "cwd": str(cwd) if cwd else None,
                "status": "skipped_existing_output",
                "expected_output": str(expected_output),
                "dry_run": self.dry_run,
            }
            self.log_command_record(record)
            return None

        if self.dry_run:
            record = {
                "step": step,
                "command": command,
                "cwd": str(cwd) if cwd else None,
                "status": "dry_run_not_executed",
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
                "dry_run": True,
            }
            self.log_command_record(record)
            return None

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=False,
            capture_output=True,
            env=env,
            check=False,
        )
        stdout_text = _decode_output(completed.stdout)
        stderr_text = _decode_output(completed.stderr)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        record = {
            "step": step,
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": completed.returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "dry_run": False,
        }
        self.log_command_record(record)
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text)
        if check and completed.returncode != 0:
            self.summary["failed_step"] = step
            raise RuntimeError(f"{step} failed with return code {completed.returncode}: {command_text}")
        return completed

    def mark_blocked(self, status: str, failed_step: str, warning: str) -> None:
        self.summary["status"] = status
        self.summary["failed_step"] = failed_step
        self.summary["warnings"].append(warning)
        print(f"[blocked:{failed_step}] {warning}")

    def classify_failed_command(self, completed: subprocess.CompletedProcess[Any] | None) -> str:
        if completed is None:
            return "dry_run_not_executed"
        text = "\n".join([_decode_output(completed.stdout), _decode_output(completed.stderr)])
        if "ModuleNotFoundError" in text or "No module named" in text:
            return "failed_environment_missing_dependency"
        if "FileNotFoundError" in text or "No such file or directory" in text or "not found" in text or "AssertionError" in text:
            return "failed_missing_referenced_path"
        return "failed"

    def should_stop(self, step: str) -> bool:
        if self.stop_after == step:
            self.summary["status"] = f"stopped_after_{step}"
            self.summary["failed_step"] = None
            return True
        return False

    def git_commit(self, path: Path) -> str | None:
        if not path.exists():
            return None
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    def record_external_state(self) -> None:
        for name, raw_path in self.config.get("external", {}).items():
            if name.endswith("_bin"):
                self.summary["external_repos"][name] = {"path": raw_path, "git_commit": None}
                continue
            path = _resolve_path(raw_path)
            self.summary["external_repos"][name] = {
                "path": str(path),
                "git_commit": self.git_commit(path),
            }

    def check_external_layout(self) -> bool:
        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        required = [
            "README.md",
            "infer.py",
            "feature/main.py",
            "scripts/prepare_mining_input.py",
            "scripts/extract_p2rank_pockets.py",
        ]
        missing = [relative for relative in required if not (enzymecage_root / relative).exists()]
        self.summary["key_files"]["enzymecage_root"] = str(enzymecage_root)
        if missing:
            self.mark_blocked(
                "blocked_missing_external_scripts",
                "clone_inspect",
                "Missing EnzymeCAGE files: " + ", ".join(missing),
            )
            return False
        return True

    def prepare_pair_csv(self) -> Path | None:
        reaction_csv = _resolve_path(self.config["data"]["reaction_csv"])
        structure_dir = _resolve_path(self.config["data"]["structure_dir"])
        pair_csv = self.dirs["feature_inputs"] / "mining.csv"
        self.summary["key_files"]["reaction_csv"] = str(reaction_csv)
        self.summary["key_files"]["structure_dir"] = str(structure_dir)

        if not reaction_csv.exists() or not structure_dir.exists():
            missing = []
            if not reaction_csv.exists():
                missing.append(str(reaction_csv))
            if not structure_dir.exists():
                missing.append(str(structure_dir))
            self.mark_blocked("blocked_missing_demo_assets", "clone_inspect", "Missing demo input data: " + "; ".join(missing))
            return None

        df = pd.read_csv(reaction_csv)
        enzyme_col = _find_column(list(df.columns), ENZYME_ID_COLUMNS)
        if enzyme_col is not None:
            if not self.dry_run:
                shutil.copy2(reaction_csv, pair_csv)
            self.summary["generated_files"].append(str(pair_csv))
            return pair_csv

        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        command = [
            sys.executable,
            str(enzymecage_root / "scripts/prepare_mining_input.py"),
            "--reaction_path",
            str(reaction_csv),
            "--structure_dir",
            str(structure_dir),
            "--output_csv",
            str(pair_csv),
        ]
        self.run_command("prepare_mining_input", command, cwd=enzymecage_root, expected_output=pair_csv)
        self.summary["generated_files"].append(str(pair_csv))
        return pair_csv

    def run_pocket_adapter(self, source: str, pair_csv: Path, top_k: int, sub_run_id: str) -> tuple[Path | None, dict[str, Any] | None]:
        structure_dir = self.current_structure_dir or _resolve_path(self.config["data"]["structure_dir"])
        adapter_dir = self.working_dir / f"{source}_adapter"
        if source == "p2rank":
            command = [
                sys.executable,
                str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/p2rank_multi_to_enzymecage.py"),
                "--input_csv",
                str(pair_csv),
                "--structure_dir",
                str(structure_dir),
                "--p2rank_home",
                str(_resolve_path(self.config["external"]["p2rank_home"])),
                "--output_dir",
                str(adapter_dir),
                "--top_k",
                str(top_k),
                "--threads",
                str(self.config.get("runtime", {}).get("threads", 1)),
                "--run_id",
                sub_run_id,
            ]
        elif source == "fpocket":
            command = [
                sys.executable,
                str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/fpocket_to_enzymecage.py"),
                "--input_csv",
                str(pair_csv),
                "--structure_dir",
                str(structure_dir),
                "--output_dir",
                str(adapter_dir),
                "--top_k",
                str(top_k),
                "--run_id",
                sub_run_id,
                "--fpocket_bin",
                str(self.config["external"].get("fpocket_bin", "fpocket")),
                "--threads",
                str(self.config.get("runtime", {}).get("threads", 1)),
            ]
        elif source == "official_precomputed":
            pocket_dir = _resolve_path(
                self.config.get("pocket", {}).get(
                    "official_pocket_dir",
                    "external_repos/EnzymeCAGE/dataset/RHEA/2025-02-05/pockets/pocket",
                )
            )
            command = [
                sys.executable,
                str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/official_precomputed_to_enzymecage.py"),
                "--input_csv",
                str(pair_csv),
                "--pocket_dir",
                str(pocket_dir),
                "--output_dir",
                str(adapter_dir),
                "--run_id",
                sub_run_id,
            ]
        else:
            raise ValueError(f"Unsupported pocket source: {source}")

        manifest_path = adapter_dir / "manifests/pocket_manifest.csv"
        summary_path = adapter_dir / "summary.json"
        should_skip_existing_output = False
        if manifest_path.exists() and summary_path.exists():
            try:
                existing_adapter_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                should_skip_existing_output = existing_adapter_summary.get("status") == "completed"
            except Exception:
                should_skip_existing_output = False
        expected_output = manifest_path if should_skip_existing_output else None
        self.run_command(f"{source}_pocket_extraction", command, cwd=PROJECT_ROOT, expected_output=expected_output)
        if self.dry_run:
            return manifest_path, None
        adapter_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
        if adapter_summary:
            status = adapter_summary.get("status")
            if status in {
                "blocked_p2rank_missing",
                "blocked_fpocket_missing",
                "blocked_no_official_precomputed_pockets",
                "blocked_no_valid_pockets",
                "blocked_fpocket_runtime_error",
            }:
                self.mark_blocked(status, "pocket_extraction", f"{source} adapter status: {status}")
            self.summary["warnings"].extend(adapter_summary.get("warnings", []))
        if manifest_path.exists():
            copied_manifest = self.dirs["manifests"] / f"{source}_pocket_manifest.csv"
            shutil.copy2(manifest_path, copied_manifest)
            self.summary["generated_files"].append(str(copied_manifest))
            if adapter_summary and str(adapter_summary.get("status", "")).startswith("blocked"):
                return None, adapter_summary
            return copied_manifest, adapter_summary
        return None, adapter_summary

    def reuse_manifest_from_run(
        self,
        source_run_dir: Path,
        target_manifest: Path,
        run_id: str,
        pocket_method: str | None = None,
        pocket_source: str | None = None,
        rank_filter: int | None = None,
    ) -> Path | None:
        source_manifest = source_run_dir / "manifests/pocket_manifest.csv"
        if not source_manifest.exists():
            return None

        df = pd.read_csv(source_manifest)
        if rank_filter is not None and "pocket_rank" in df.columns:
            df = df[df["pocket_rank"] == rank_filter].copy()
        if df.empty:
            return None

        if "run_id" in df.columns:
            df["run_id"] = run_id
        if pocket_method is not None and "pocket_method" in df.columns:
            df["pocket_method"] = pocket_method
        if pocket_source is not None and "pocket_source" in df.columns:
            df["pocket_source"] = pocket_source

        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_manifest, index=False)
        self.summary["generated_files"].append(str(target_manifest))
        reuse_note = f"Reused manifest from {source_run_dir}"
        if rank_filter is not None:
            reuse_note += f" with rank_filter={rank_filter}"
        self.summary["warnings"].append(reuse_note)
        print(f"[reuse] {reuse_note}")
        return target_manifest

    def extract_pockets(self, pair_csv: Path) -> Path | None:
        pocket = self.config["pocket"]
        source = pocket["source"]
        selection = pocket.get("selection", "top1")

        if source == "enzymecage_official_or_p2rank":
            top_k = 1
            return self.extract_pockets_for_sources(pair_csv, ["p2rank"], top_k)
        if source == "official_precomputed":
            return self.extract_pockets_for_sources(pair_csv, ["official_precomputed"], 1)
        if source == "p2rank+fpocket" or selection == "union_topk":
            top_k_each = int(pocket.get("top_k_each", 5))
            fpocket_bin = str(self.config["external"].get("fpocket_bin", "fpocket"))
            if "fpocket" in pocket.get("union_sources", ["p2rank", "fpocket"]) and shutil.which(fpocket_bin) is None:
                self.mark_blocked(
                    "blocked_fpocket_missing",
                    "pocket_extraction",
                    f"fpocket executable is not available: {fpocket_bin}",
                )
                return None
            manifests = []
            for item in pocket.get("union_sources", ["p2rank", "fpocket"]):
                if item == "p2rank" and pocket.get("p2rank_source_run_dir"):
                    source_run_dir = _resolve_path(pocket["p2rank_source_run_dir"])
                    reused_manifest = self.reuse_manifest_from_run(
                        source_run_dir=source_run_dir,
                        target_manifest=self.working_dir / f"{item}_adapter/manifests/pocket_manifest.csv",
                        run_id=f"{self.run_id}_{item}",
                        pocket_method=None,
                        pocket_source=None,
                    )
                    if reused_manifest and reused_manifest.exists():
                        copied_manifest = self.dirs["manifests"] / f"{item}_pocket_manifest.csv"
                        shutil.copy2(reused_manifest, copied_manifest)
                        manifests.append(copied_manifest)
                        continue
                manifest, _ = self.run_pocket_adapter(item, pair_csv, top_k_each, f"{self.run_id}_{item}")
                if manifest and manifest.exists():
                    manifests.append(manifest)
            if len(manifests) < 2:
                blocked_status = self.summary["status"] if self.summary["status"].startswith("blocked_") else "partial_union_source_missing"
                self.mark_blocked(
                    blocked_status,
                    "pocket_extraction",
                    "Union baseline needs both P2Rank and fpocket manifests.",
                )
                return manifests[0] if manifests else None
            merged = self.dirs["manifests"] / "pocket_manifest.csv"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/merge_pocket_manifests.py"),
                "--manifest_csvs",
                *[str(path) for path in manifests],
                "--output_csv",
                str(merged),
                "--run_id",
                self.run_id,
            ]
            self.run_command("merge_pocket_manifests", command, cwd=PROJECT_ROOT, expected_output=merged)
            self.summary["generated_files"].append(str(merged))
            return merged

        top_k = 1 if selection == "top1" else int(pocket.get("top_k", 5))
        if source == "fpocket" and selection == "top1" and pocket.get("source_run_dir"):
            source_run_dir = _resolve_path(pocket["source_run_dir"])
            reused_manifest = self.reuse_manifest_from_run(
                source_run_dir=source_run_dir,
                target_manifest=self.dirs["manifests"] / "pocket_manifest.csv",
                run_id=self.run_id,
                pocket_method="fpocket_top1",
                pocket_source="fpocket",
                rank_filter=1,
            )
            if reused_manifest and reused_manifest.exists():
                return reused_manifest
        return self.extract_pockets_for_sources(pair_csv, [source], top_k)

    def extract_pockets_for_sources(self, pair_csv: Path, sources: list[str], top_k: int) -> Path | None:
        manifests = []
        adapter_summaries: list[dict[str, Any] | None] = []
        for source in sources:
            manifest, adapter_summary = self.run_pocket_adapter(source, pair_csv, top_k, f"{self.run_id}_{source}")
            adapter_summaries.append(adapter_summary)
            if manifest and manifest.exists():
                manifests.append(manifest)
        if not manifests:
            for adapter_summary in adapter_summaries:
                if adapter_summary and str(adapter_summary.get("status", "")).startswith("blocked"):
                    self.mark_blocked(
                        str(adapter_summary.get("status")),
                        "pocket_extraction",
                        f"{sources[0]} adapter status: {adapter_summary.get('status')}",
                    )
                    break
            return None
        final_manifest = self.dirs["manifests"] / "pocket_manifest.csv"
        shutil.copy2(manifests[0], final_manifest)
        self.summary["generated_files"].append(str(final_manifest))
        return final_manifest

    def build_pocket_variant_inputs(self, pair_csv: Path, manifest_csv: Path) -> tuple[Path, Path] | None:
        if self.dry_run:
            return self.dirs["feature_inputs"] / "pocket_variant_input.csv", self.dirs["pocket_inputs"] / "pocket"
        df_pairs = pd.read_csv(pair_csv)
        df_manifest = pd.read_csv(manifest_csv)
        if df_manifest.empty:
            self.mark_blocked("blocked_no_pockets", "pocket_extraction", f"No pockets in manifest: {manifest_csv}")
            return None

        enzyme_col = _find_column(list(df_pairs.columns), ENZYME_ID_COLUMNS)
        if enzyme_col is None:
            self.mark_blocked("failed", "feature_generation", f"Pair CSV lacks enzyme id column: {pair_csv}")
            return None
        if enzyme_col != "UniprotID":
            df_pairs = df_pairs.rename(columns={enzyme_col: "UniprotID"})

        pocket_dir = self.dirs["pocket_inputs"] / "pocket"
        pocket_dir.mkdir(parents=True, exist_ok=True)
        manifest_by_enzyme = {
            enzyme_id: group.to_dict("records")
            for enzyme_id, group in df_manifest.groupby("enzyme_id")
        }
        structure_cache: dict[str, Any] = {}

        def write_subset_pdb(structure_path_value: str, residue_numbers: list[int], target_pdb: Path) -> bool:
            structure_path = _resolve_path(structure_path_value)
            if not structure_path.exists():
                return False
            cache_key = str(structure_path)
            structure = structure_cache.get(cache_key)
            if structure is None:
                parser = MMCIFParser(QUIET=True) if structure_path.suffix.lower() in {".cif", ".mmcif"} else PDBParser(QUIET=True)
                structure = parser.get_structure(structure_path.stem, str(structure_path))
                structure_cache[cache_key] = structure

            residue_set = set(residue_numbers)

            class PocketSelect(Select):
                def accept_residue(self, residue: Any) -> bool:
                    hetero_flag, residue_number, _insertion_code = residue.id
                    return hetero_flag == " " and residue_number in residue_set

            target_pdb.parent.mkdir(parents=True, exist_ok=True)
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(target_pdb), select=PocketSelect())
            return target_pdb.exists() and target_pdb.stat().st_size > 0

        rows = []
        pocket_info_rows = []
        seen_pocket_info: set[str] = set()
        for _, pair in df_pairs.iterrows():
            enzyme_id = str(pair["UniprotID"])
            for pocket in manifest_by_enzyme.get(enzyme_id, []):
                pocket_global_id = str(pocket["pocket_global_id"])
                pocket_residues = pocket.get("pocket_residues")
                if pd.isna(pocket_residues):
                    pocket_residues = ""
                pocket_residues = str(pocket_residues).strip()
                if pocket_residues:
                    normalized_tokens: list[str] = []
                    for token in pocket_residues.split(","):
                        token = token.strip().strip("'\"")
                        if not token:
                            continue
                        match = re.search(r"-?\d+(?:\.\d+)?", token)
                        if match:
                            normalized_tokens.append(match.group(0))
                    pocket_residues = ",".join(normalized_tokens)

                row = pair.to_dict()
                row["original_enzyme_id"] = enzyme_id
                row["UniprotID"] = pocket_global_id
                row["pocket_global_id"] = pocket_global_id
                row["pocket_source"] = pocket["pocket_source"]
                row["pocket_rank"] = pocket["pocket_rank"]
                row["pocket_score_original"] = pocket.get("pocket_score")
                rows.append(row)
                target_pdb = pocket_dir / f"{pocket_global_id}.pdb"
                structure_path_value = str(pair.get("structure_path", "")).strip()
                wrote_subset = False
                if structure_path_value and pocket_residues:
                    try:
                        wrote_subset = write_subset_pdb(
                            structure_path_value,
                            [int(token) for token in pocket_residues.split(",") if token],
                            target_pdb,
                        )
                    except Exception as exc:
                        self.summary["warnings"].append(
                            f"Could not write pocket subset from {structure_path_value} for {pocket_global_id}: {exc}"
                        )
                if not wrote_subset:
                    source_pdb = Path(pocket["pocket_pdb_path"])
                    if source_pdb.exists():
                        shutil.copy2(source_pdb, target_pdb)
                if pocket_global_id not in seen_pocket_info:
                    if not pocket_residues and target_pdb.exists():
                        pocket_residues = _parse_pdb_residue_numbers(target_pdb)
                    pocket_info_rows.append(
                        {
                            "UniprotID": pocket_global_id,
                            "pocket_residues": pocket_residues,
                        }
                    )
                    seen_pocket_info.add(pocket_global_id)

        variant_csv = self.dirs["feature_inputs"] / "pocket_variant_input.csv"
        pocket_info_csv = self.dirs["pocket_inputs"] / "pocket_info.csv"
        variant_df = pd.DataFrame(rows)
        for label_column in ["label", "Label", "y", "target"]:
            if label_column in variant_df.columns:
                variant_df[label_column] = pd.to_numeric(variant_df[label_column], errors="coerce").fillna(0).astype(int)
        variant_df.to_csv(variant_csv, index=False)
        pd.DataFrame(pocket_info_rows).to_csv(pocket_info_csv, index=False)
        self.summary["generated_files"].extend([str(variant_csv), str(pocket_dir), str(pocket_info_csv)])
        return variant_csv, pocket_dir

    def write_infer_config(self, data_path: Path) -> Path:
        data_dir = data_path.parent
        checkpoint_dir = _resolve_path(self.config["model"]["checkpoint_dir"])
        model_name = self.config["model"]["checkpoint_name"]
        result_dir = self.dirs["predictions"]
        config = {
            "model": "EnzymeCAGE",
            "interaction_method": "geo-enhanced-interaction",
            "rxn_inner_interaction": True,
            "pocket_inner_interaction": True,
            "use_prods_info": False,
            "use_structure": True,
            "use_drfp": True,
            "use_esm": True,
            "esm_model": "ESM-C_600M",
            "batch_size": 64,
            "model_list": [model_name],
            "data_path": str(data_path.resolve()),
            "ckpt_dir": str(checkpoint_dir.resolve()),
            "result_dir": str(result_dir.resolve()),
            "rxn_fp": str((data_dir / "feature/reaction/drfp/rxn2fp.pkl").resolve()),
            "mol_conformation": str((data_dir / "feature/reaction/molecule_conformation").resolve()),
            "reaction_center": str((data_dir / "feature/reaction/reacting_center/reacting_center.pkl").resolve()),
            "protein_gvp_feat": str((data_dir / "feature/protein/gvp_feature/gvp_protein_feature.pt").resolve()),
            "esm_mean_feature": str((data_dir / "feature/protein/ESM-C_600M/protein_level/seq2feature.pkl").resolve()),
            "esm_node_feature": str((data_dir / "feature/protein/ESM-C_600M/pocket_node_feature/esm_node_feature.pt").resolve()),
        }
        infer_config = self.dirs["feature_inputs"] / "infer_config.yaml"
        _write_yaml(config, infer_config)
        self.summary["generated_files"].append(str(infer_config))
        return infer_config

    def run_feature_generation(self, variant_csv: Path, pocket_dir: Path) -> bool:
        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        command = [
            sys.executable,
            str(enzymecage_root / "feature/main.py"),
            "--data_path",
            str(variant_csv),
            "--pocket_dir",
            str(pocket_dir),
        ]
        self.run_command("feature_generation", command, cwd=enzymecage_root / "feature")
        return True

    def run_inference(self, infer_config: Path, variant_csv: Path, manifest_csv: Path) -> Path | None:
        checkpoint = _resolve_path(self.config["model"]["checkpoint_dir"]) / self.config["model"]["checkpoint_name"]
        if not checkpoint.exists():
            self.mark_blocked("blocked_checkpoint_missing", "inference", f"Checkpoint missing: {checkpoint}")
            return None
        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        command = [sys.executable, str(enzymecage_root / "infer.py"), "--config", str(infer_config)]
        self.run_command("inference", command, cwd=enzymecage_root)

        result_name = f"{variant_csv.stem}_{self.config['model']['checkpoint_name'].replace('.pth', '.csv')}"
        result_path = self.dirs["predictions"] / result_name
        if not result_path.exists():
            self.mark_blocked("partial_prediction_missing", "inference", f"Expected inference result missing: {result_path}")
            return None
        return self.convert_predictions(result_path, manifest_csv)

    def convert_predictions(self, raw_prediction_csv: Path, manifest_csv: Path) -> Path:
        df_pred = pd.read_csv(raw_prediction_csv)
        df_manifest = pd.read_csv(manifest_csv)
        df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()].copy()
        score_col = _find_column(list(df_pred.columns), ["pred", "score", "prediction", "catalytic_score", "y_pred"])
        if score_col is None:
            raise ValueError(f"Could not identify prediction score in {raw_prediction_csv}")
        rename_map = {score_col: "cage_score"} if score_col != "cage_score" else {}
        if "pocket_global_id" not in df_pred.columns:
            enzyme_col = _find_column(list(df_pred.columns), ENZYME_ID_COLUMNS) or "UniprotID"
            rename_map[enzyme_col] = "pocket_global_id"
        df_pred = df_pred.rename(columns=rename_map)
        df_pred = df_pred.loc[:, ~df_pred.columns.duplicated()].copy()
        manifest_metadata = df_manifest[
            [
                "enzyme_id",
                "pocket_global_id",
                "pocket_source",
                "pocket_rank",
                "pocket_score",
            ]
        ].rename(columns={"enzyme_id": "manifest_enzyme_id"})
        merged = df_pred.merge(
            manifest_metadata,
            on="pocket_global_id",
            how="left",
            suffixes=("", "_manifest"),
        )
        if "manifest_enzyme_id" in merged.columns:
            if "enzyme_id" not in merged.columns:
                merged["enzyme_id"] = merged["manifest_enzyme_id"]
            else:
                merged["enzyme_id"] = merged["manifest_enzyme_id"].fillna(merged["enzyme_id"])
        if "original_enzyme_id" in merged.columns:
            merged["enzyme_id"] = merged["enzyme_id"].fillna(merged["original_enzyme_id"])
        for column in ["pocket_source", "pocket_rank", "pocket_score"]:
            manifest_column = f"{column}_manifest"
            if manifest_column in merged.columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(merged[manifest_column])
                else:
                    merged[column] = merged[manifest_column]
        output = self.dirs["predictions"] / "pocket_level_predictions.csv"
        keep_cols = [
            column
            for column in [
                "reaction_id",
                "enzyme_id",
                "pocket_global_id",
                "pocket_source",
                "pocket_rank",
                "pocket_score",
                "cage_score",
            ]
            if column in merged.columns
        ]
        merged[keep_cols].rename(columns={"pocket_score": "pocket_score_original"}).to_csv(output, index=False)
        self.summary["generated_files"].append(str(output))
        return output

    def run_aggregation(self, pocket_prediction_csv: Path, manifest_csv: Path) -> Path:
        method = self.config["aggregation"]["method"]
        output_csv = self.dirs["aggregation"] / f"enzyme_level_{method}.csv"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "projects/active/pocket_robustness/analysis/aggregate_pocket_scores.py"),
            "--prediction_csv",
            str(pocket_prediction_csv),
            "--manifest_csv",
            str(manifest_csv),
            "--output_csv",
            str(output_csv),
            "--method",
            method,
            "--temperature",
            str(self.config["aggregation"].get("temperature", 0.2)),
        ]
        if "source_weights" in self.config["aggregation"]:
            command.extend(["--source_weights", json.dumps(self.config["aggregation"]["source_weights"])])
        self.run_command("aggregation", command, cwd=PROJECT_ROOT, expected_output=output_csv)
        self.summary["generated_files"].append(str(output_csv))
        return output_csv

    def run_evaluation(self, aggregated_csv: Path) -> None:
        label_csv_value = self.config.get("evaluation", {}).get("label_csv")
        if label_csv_value in (None, "null", "None", ""):
            self.summary["warnings"].append("No label_csv configured; top-k evaluation skipped.")
            return
        label_csv = _resolve_path(label_csv_value)
        output_json = self.dirs["metrics"] / "metrics_top5_top10.json"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "projects/active/pocket_robustness/analysis/evaluate_topk.py"),
            "--prediction_csv",
            str(aggregated_csv),
            "--label_csv",
            str(label_csv),
            "--output_json",
            str(output_json),
            "--topk",
            *[str(k) for k in self.config.get("evaluation", {}).get("topk", [1, 3, 5, 10])],
        ]
        self.run_command("analysis", command, cwd=PROJECT_ROOT, check=False)
        self.summary["generated_files"].append(str(output_json))

    def run_official_eval(self) -> dict[str, Any]:
        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        config_source_value = self.config.get("data", {}).get("config_source")
        if not config_source_value:
            self.mark_blocked("official_eval_failed", "official_eval", "No data.config_source configured for official_eval baseline.")
            return self.summary
        config_source = _resolve_path(config_source_value)
        self.summary["key_files"]["official_config"] = str(config_source)
        if not config_source.exists():
            self.mark_blocked("official_eval_failed", "official_eval", f"Official config missing: {config_source}")
            return self.summary
        checkpoint = _resolve_path(self.config["model"]["checkpoint_dir"]) / self.config["model"]["checkpoint_name"]
        self.summary["key_files"]["checkpoint"] = str(checkpoint)
        if not checkpoint.exists():
            self.mark_blocked("blocked_checkpoint_missing", "official_eval", f"Checkpoint missing: {checkpoint}")
            return self.summary
        command = [sys.executable, str(enzymecage_root / "infer.py"), "--config", str(config_source)]
        completed = self.run_command("official_eval_infer", command, cwd=enzymecage_root, check=False)
        if self.dry_run:
            self.summary["status"] = "dry_run"
            return self.summary
        if completed is not None and completed.returncode == 0:
            self.summary["status"] = "official_eval_completed"
            self.summary["failed_step"] = None
        else:
            status = self.classify_failed_command(completed)
            self.summary["status"] = status if status != "failed" else "official_eval_failed"
            self.summary["failed_step"] = "official_eval"
            self.summary["error"] = f"Official EnzymeCAGE infer failed with return code {completed.returncode if completed else 'unknown'}"
        return self.summary

    def build_derived_smallset(self) -> Path | None:
        data_conf = self.config.get("data", {})
        source_dataset = data_conf.get("source_dataset")
        if not source_dataset:
            self.mark_blocked("derived_smallset_blocked", "smallset_build", "No source_dataset configured.")
            return None
        enzymecage_root = _resolve_path(self.config["external"]["enzymecage_root"])
        command = [
            sys.executable,
            str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/build_official_smallset.py"),
            "--source_dataset",
            str(source_dataset),
            "--enzymecage_root",
            str(enzymecage_root),
            "--output_dir",
            str(self.working_dir),
            "--n_reactions",
            str(data_conf.get("n_reactions", 5)),
            "--n_enzymes_per_reaction",
            str(data_conf.get("n_enzymes_per_reaction", 20)),
            "--seed",
            str(self.config["project"].get("seed", 42)),
        ]
        summary_path = self.working_dir / "smallset_summary.json"
        self.run_command("build_official_smallset", command, cwd=PROJECT_ROOT, check=False, expected_output=summary_path)
        self.summary["generated_files"].extend(
            [
                str(self.working_dir / "smallset_pairs.csv"),
                str(summary_path),
                str(self.working_dir / "structure_link_report.csv"),
                str(self.working_dir / "sanity_label_report.csv"),
                str(self.working_dir / "sanity_label_report.md"),
            ]
        )
        if self.dry_run:
            self.summary["status"] = "dry_run"
            return self.working_dir / "smallset_pairs.csv"
        if not summary_path.exists():
            self.mark_blocked("derived_smallset_blocked", "smallset_build", f"Smallset summary missing: {summary_path}")
            return None
        smallset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.summary["smallset_summary"] = smallset_summary
        self.summary["warnings"].extend(smallset_summary.get("warnings", []))
        for filename in ["smallset_summary.json", "structure_link_report.csv", "sanity_label_report.csv", "sanity_label_report.md"]:
            source_path = self.working_dir / filename
            if source_path.exists():
                target_path = self.output_dir / filename
                shutil.copy2(source_path, target_path)
                self.summary["generated_files"].append(str(target_path))
        status = smallset_summary.get("status")
        source = self.config.get("pocket", {}).get("source", "")
        base_pair_csv = self.working_dir / "smallset_pairs.csv"
        if source == "official_precomputed":
            if not base_pair_csv.exists():
                self.mark_blocked("derived_smallset_blocked", "smallset_build", f"Smallset pair CSV missing: {base_pair_csv}")
                return None
            if "label" in pd.read_csv(base_pair_csv, nrows=1).columns:
                self.config.setdefault("evaluation", {})["label_csv"] = str(base_pair_csv)
            return base_pair_csv

        if status != "derived_smallset_completed":
            # Official eval assets often include precomputed pockets but no full
            # structures. For intervention baselines, download the needed full
            # AlphaFold CIFs instead of treating precomputed pockets as structures.
            download_output = self.working_dir / "smallset_pairs_with_structures.csv"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "projects/active/pocket_robustness/adapters/download_alphafold_structures.py"),
                "--input_csv",
                str(base_pair_csv),
                "--output_root",
                str(PROJECT_ROOT / "data/assets/alphafold_structures"),
                "--output_pairs_csv",
                str(download_output),
            ]
            download_summary_path = PROJECT_ROOT / "data/assets/alphafold_structures/alphafold_download_summary.json"
            self.run_command("download_alphafold_structures", command, cwd=PROJECT_ROOT, check=False, expected_output=download_summary_path)
            self.summary["generated_files"].extend(
                [
                    str(download_output),
                    str(download_summary_path),
                    str(PROJECT_ROOT / "data/assets/alphafold_structures/alphafold_download_report.csv"),
                ]
            )
            if self.dry_run:
                self.current_structure_dir = PROJECT_ROOT / "data/assets/alphafold_structures/cif"
                return download_output

            if not download_output.exists() or pd.read_csv(download_output).empty:
                cached_output = PROJECT_ROOT / "data/assets/alphafold_structures/combined_smallset_pairs_with_structures.csv"
                if cached_output.exists() and cached_output.stat().st_size > 0:
                    download_output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cached_output, download_output)
                    self.summary["generated_files"].append(str(download_output))
                    self.summary["warnings"].append(f"Reused cached AlphaFold pair CSV from {cached_output}")
                    print(f"[reuse] Copied cached AlphaFold pairs from {cached_output} -> {download_output}")

            if not download_output.exists() or pd.read_csv(download_output).empty:
                blocked_reason = smallset_summary.get("blocked_reason", "blocked_missing_structure_mapping")
                self.mark_blocked(blocked_reason, "smallset_build", "Could not create smallset with downloaded full AlphaFold structures.")
                return None

            pair_csv = download_output
            self.current_structure_dir = PROJECT_ROOT / "data/assets/alphafold_structures/cif"
            if "label" in pd.read_csv(pair_csv, nrows=1).columns:
                self.config.setdefault("evaluation", {})["label_csv"] = str(pair_csv)
            self.config["data"]["structure_dir"] = str(self.current_structure_dir)
            self.config["data"]["reaction_csv"] = str(pair_csv)
            return pair_csv

        if status != "derived_smallset_completed":
            blocked_reason = smallset_summary.get("blocked_reason", "derived_smallset_blocked")
            self.mark_blocked(blocked_reason, "smallset_build", f"Smallset builder status: {blocked_reason}")
            return None
        pair_csv = self.working_dir / "smallset_pairs.csv"
        self.current_structure_dir = self.working_dir / "smallset_structures"
        if "label" in pd.read_csv(pair_csv, nrows=1).columns:
            self.config.setdefault("evaluation", {})["label_csv"] = str(pair_csv)
        self.config["data"]["structure_dir"] = str(self.current_structure_dir)
        self.config["data"]["reaction_csv"] = str(pair_csv)
        return pair_csv

    def run(self) -> dict[str, Any]:
        if self.resume:
            existing_summary = self.load_existing_completed_summary()
            if existing_summary is not None:
                print(f"[resume] Existing completed run found for {self.run_id}; skipping rerun.")
                return existing_summary

        self.setup_dirs()
        try:
            self.record_external_state()
            if not self.check_external_layout():
                return self.summary
            if self.should_stop("clone_inspect"):
                return self.summary

            data_mode = self.config.get("data", {}).get("data_mode", "demo_mining")
            self.summary["data_mode"] = data_mode
            if data_mode == "official_eval":
                self.run_official_eval()
                return self.summary

            if data_mode == "derived_smallset":
                pair_csv = self.build_derived_smallset()
                if pair_csv is None or self.summary["status"].startswith("blocked_"):
                    return self.summary
                if self.should_stop("smallset_build"):
                    return self.summary
            else:
                pair_csv = self.prepare_pair_csv()
            if pair_csv is None or self.summary["status"].startswith("blocked_"):
                return self.summary
            manifest_csv = self.extract_pockets(pair_csv)
            if manifest_csv is None:
                if not self.summary["status"].startswith("blocked_"):
                    self.mark_blocked("blocked_no_manifest", "pocket_extraction", "No pocket manifest was generated.")
                return self.summary
            if self.summary["status"].startswith("blocked_"):
                return self.summary
            if self.should_stop("pocket_extraction"):
                return self.summary

            variant = self.build_pocket_variant_inputs(pair_csv, manifest_csv)
            if variant is None or self.summary["status"].startswith("blocked_"):
                return self.summary
            variant_csv, pocket_dir = variant

            checkpoint = _resolve_path(self.config["model"]["checkpoint_dir"]) / self.config["model"]["checkpoint_name"]
            if not checkpoint.exists():
                self.mark_blocked("blocked_checkpoint_missing", "inference", f"Checkpoint missing: {checkpoint}")
                return self.summary

            self.run_feature_generation(variant_csv, pocket_dir)
            if self.should_stop("feature_generation"):
                return self.summary

            infer_config = self.write_infer_config(variant_csv)
            pocket_prediction_csv = self.run_inference(infer_config, variant_csv, manifest_csv)
            if pocket_prediction_csv is None or self.summary["status"].startswith("blocked_"):
                return self.summary
            if self.should_stop("inference"):
                return self.summary

            aggregated_csv = self.run_aggregation(pocket_prediction_csv, manifest_csv)
            if self.should_stop("aggregation"):
                return self.summary

            self.run_evaluation(aggregated_csv)
            self.summary["status"] = "completed"
            self.summary["failed_step"] = None
            return self.summary
        except Exception as exc:
            if not self.summary.get("failed_step"):
                self.summary["failed_step"] = "unknown"
            self.summary["status"] = "failed"
            self.summary["error"] = str(exc)
            print(f"[failed] {exc}")
            return self.summary
        finally:
            summary_path = self.write_summary()
            print(f"[summary] {summary_path}")


def run_experiment(config_path: Path, dry_run: bool = False, resume: bool = False, stop_after: str | None = None) -> dict[str, Any]:
    runner = ExperimentRunner(config_path=config_path, dry_run=dry_run, resume=resume, stop_after=stop_after)
    return runner.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a pocket baseline comparison experiment.")
    parser.add_argument("--experiment_config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop_after", choices=STOP_STEPS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_experiment(
        Path(args.experiment_config),
        dry_run=args.dry_run,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
