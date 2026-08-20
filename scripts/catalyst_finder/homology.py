from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import threading
from collections import defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "results/catalyst_finder_runtime"
CACHE_ROOT = RUNTIME_ROOT / "cache"
CURRENT_SEQUENCES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
PRODUCTION_REGISTRY = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/protein_registry.csv"
DEFAULT_MMSEQS = ROOT / "data/assets/mmseqs2/mmseqs/bin/mmseqs"
MMSEQS_PATH_FILE = ROOT / "data/assets/mmseqs2/MMSEQS_PATH"
CLUSTER_CACHE = CACHE_ROOT / "protein_clusters_production_id50.csv"
CLUSTER_META = CACHE_ROOT / "protein_clusters_production_id50.meta.json"

# This is the repository's formal protein-family boundary used by its
# protein-cluster-cold / cross-cluster evaluations.
MIN_SEQUENCE_IDENTITY = 0.50
MIN_COVERAGE = 0.80
COV_MODE = 2
CLUSTER_MODE = 0


def _clean_sequence(value: str) -> str:
    return "".join(char for char in "".join(str(value or "").upper().split()) if "A" <= char <= "Z")


def _source_signature() -> dict[str, object]:
    paths = [CURRENT_SEQUENCES, PRODUCTION_REGISTRY]
    return {
        "inputs": [
            {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
        "min_sequence_identity": MIN_SEQUENCE_IDENTITY,
        "min_coverage": MIN_COVERAGE,
        "cov_mode": COV_MODE,
        "cluster_mode": CLUSTER_MODE,
    }


def _resolve_mmseqs() -> Path:
    candidates: list[Path] = [DEFAULT_MMSEQS]
    if MMSEQS_PATH_FILE.is_file():
        raw = MMSEQS_PATH_FILE.read_text(encoding="utf-8").strip()
        if raw:
            candidates.insert(0, Path(raw))
    binary = shutil.which("mmseqs")
    if binary:
        candidates.append(Path(binary))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError("MMseqs2 binary is unavailable; cross-cluster filtering cannot be built")


def _load_production_sequences() -> dict[str, str]:
    current: dict[str, str] = {}
    with CURRENT_SEQUENCES.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            identifier = str(row.get("Entry") or "").strip()
            sequence = _clean_sequence(str(row.get("Sequence") or ""))
            if identifier and sequence:
                current.setdefault(identifier, sequence)

    combined = dict(current)
    with PRODUCTION_REGISTRY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            identifier = str(row.get("protein_id") or "").strip()
            sequence = _clean_sequence(str(row.get("sequence") or ""))
            if identifier and sequence:
                combined.setdefault(identifier, sequence)

    if not combined:
        raise RuntimeError("No production enzyme sequences are available for homology clustering")
    return combined


class ProteinHomologyIndex:
    """MMseqs2 50%-identity cluster index for the deployed 2,085-protein universe.

    The repository already uses exactly this family boundary for protein-cluster-
    cold evaluation: 50% minimum sequence identity with 80% coverage. The isolated
    interface rebuilds the same definition over the *deployed* current + registered
    candidate universe so a novelty filter can cover all candidates, not only the
    original 1,391 current proteins.
    """

    def __init__(self, cache_path: Path = CLUSTER_CACHE) -> None:
        self.cache_path = cache_path
        self.meta_path = cache_path.with_suffix(".meta.json") if cache_path != CLUSTER_CACHE else CLUSTER_META
        self._lock = threading.Lock()
        self._cluster_by_id: dict[str, str] | None = None
        self._members_by_cluster: dict[str, set[str]] | None = None

    @property
    def ready(self) -> bool:
        return self.cache_path.is_file() and self.meta_path.is_file()

    def ensure(self) -> None:
        if self._cluster_by_id is not None:
            return
        with self._lock:
            if self._cluster_by_id is not None:
                return
            signature = _source_signature()
            if self.cache_path.is_file() and self.meta_path.is_file():
                try:
                    existing = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = None
                if existing == signature:
                    self._load()
                    return
            self._build(signature)
            self._load()

    def _build(self, signature: dict[str, object]) -> None:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        work = CACHE_ROOT / "mmseqs_production_id50_work"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        fasta = work / "production_candidates.fasta"
        sequences = _load_production_sequences()
        with fasta.open("w", encoding="utf-8") as handle:
            for identifier in sorted(sequences):
                handle.write(f">{identifier}\n{sequences[identifier]}\n")

        mmseqs = _resolve_mmseqs()
        prefix = work / "clusters"
        tmp = work / "tmp"
        command = [
            str(mmseqs), "easy-cluster", str(fasta), str(prefix), str(tmp),
            "--min-seq-id", str(MIN_SEQUENCE_IDENTITY),
            "-c", str(MIN_COVERAGE),
            "--cov-mode", str(COV_MODE),
            "--cluster-mode", str(CLUSTER_MODE),
            "--threads", str(max(1, min(8, os.cpu_count() or 1))),
            "--remove-tmp-files", "1",
        ]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        cluster_tsv = Path(f"{prefix}_cluster.tsv")
        if completed.returncode or not cluster_tsv.is_file():
            shutil.rmtree(work, ignore_errors=True)
            message = completed.stderr[-2000:] or completed.stdout[-2000:] or "MMseqs2 did not create cluster output"
            raise RuntimeError(message)

        assignments: dict[str, str] = {}
        with cluster_tsv.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    assignments[parts[1]] = parts[0]
        for identifier in sequences:
            assignments.setdefault(identifier, identifier)

        tmp_csv = self.cache_path.with_suffix(".tmp.csv")
        with tmp_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["protein_id", "cluster_id"])
            for identifier in sorted(assignments):
                writer.writerow([identifier, assignments[identifier]])
        tmp_csv.replace(self.cache_path)
        self.meta_path.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)

    def _load(self) -> None:
        cluster_by_id: dict[str, str] = {}
        members: dict[str, set[str]] = defaultdict(set)
        with self.cache_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                identifier = str(row.get("protein_id") or "").strip()
                cluster = str(row.get("cluster_id") or "").strip()
                if identifier and cluster:
                    cluster_by_id[identifier] = cluster
                    members[cluster].add(identifier)
        self._cluster_by_id = cluster_by_id
        self._members_by_cluster = dict(members)

    def cluster_id(self, protein_id: str) -> str | None:
        self.ensure()
        assert self._cluster_by_id is not None
        return self._cluster_by_id.get(str(protein_id))

    def exclusion_set(self, anchors: Iterable[str]) -> tuple[set[str], dict[str, object]]:
        self.ensure()
        assert self._cluster_by_id is not None
        assert self._members_by_cluster is not None
        anchor_ids = [str(value) for value in anchors if str(value)]
        cluster_ids: list[str] = []
        excluded: set[str] = set()
        unresolved: list[str] = []
        for anchor in anchor_ids:
            cluster = self._cluster_by_id.get(anchor)
            if not cluster:
                unresolved.append(anchor)
                continue
            if cluster not in cluster_ids:
                cluster_ids.append(cluster)
            excluded.update(self._members_by_cluster.get(cluster, {anchor}))
        return excluded, {
            "definition": "MMseqs2 50% sequence-identity cluster, coverage >= 80%",
            "min_sequence_identity": MIN_SEQUENCE_IDENTITY,
            "min_coverage": MIN_COVERAGE,
            "cluster_ids": cluster_ids,
            "anchor_ids": anchor_ids,
            "unresolved_anchor_ids": unresolved,
            "excluded_count": len(excluded),
        }

    def summary(self) -> dict[str, object]:
        self.ensure()
        assert self._cluster_by_id is not None
        assert self._members_by_cluster is not None
        sizes = [len(values) for values in self._members_by_cluster.values()]
        return {
            "candidate_count": len(self._cluster_by_id),
            "cluster_count": len(self._members_by_cluster),
            "largest_cluster": max(sizes, default=0),
            "definition": "MMseqs2 50% sequence-identity cluster, coverage >= 80%",
        }
