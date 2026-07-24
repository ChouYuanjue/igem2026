from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "data/terpene_sequence_clusters"
DEFAULT_THRESHOLDS = (0.30, 0.50, 0.70, 0.90)


def resolve_mmseqs(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    path_file = ROOT / "data/assets/mmseqs2/MMSEQS_PATH"
    if path_file.exists():
        candidates.append(Path(path_file.read_text(encoding="utf-8").strip()))
    binary = shutil.which("mmseqs")
    if binary:
        candidates.append(Path(binary))
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError("MMseqs2 binary not found; install it under data/assets/mmseqs2 or pass --mmseqs.")


def clean_sequence(value: object) -> str:
    sequence = "".join(str(value).upper().split())
    return "".join(char for char in sequence if "A" <= char <= "Z")


def write_fasta(input_path: Path, fasta_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, sep="\t", dtype=str).fillna("")
    required = {"Entry", "Sequence"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame.drop_duplicates("Entry", keep="first").copy()
    frame["Sequence"] = frame["Sequence"].map(clean_sequence)
    frame = frame[frame["Sequence"].str.len() > 0].sort_values("Entry").reset_index(drop=True)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in frame.itertuples(index=False):
            handle.write(f">{row.Entry}\n{row.Sequence}\n")
    return frame


def run_cluster(mmseqs: Path, fasta_path: Path, output_dir: Path, threshold: float, threads: int, force: bool) -> Path:
    tag = f"id{int(round(threshold * 100)):02d}"
    prefix = output_dir / tag
    tmp_dir = output_dir / f"tmp_{tag}"
    cluster_tsv = Path(f"{prefix}_cluster.tsv")
    if cluster_tsv.exists() and not force:
        return cluster_tsv
    for path in output_dir.glob(f"{tag}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    command = [
        str(mmseqs),
        "easy-cluster",
        str(fasta_path),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(threshold),
        "-c",
        "0.8",
        "--cov-mode",
        "2",
        "--cluster-mode",
        "0",
        "--threads",
        str(threads),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"MMseqs2 did not create {cluster_tsv}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return cluster_tsv


def parse_clusters(cluster_tsv: Path, threshold: float, universe: set[str]) -> pd.DataFrame:
    rows = pd.read_csv(cluster_tsv, sep="\t", names=["representative", "entry"], dtype=str)
    rows = rows.drop_duplicates("entry", keep="first")
    missing = sorted(universe - set(rows["entry"]))
    if missing:
        rows = pd.concat(
            [rows, pd.DataFrame({"representative": missing, "entry": missing})],
            ignore_index=True,
        )
    rows["cluster_id"] = rows["representative"]
    rows["min_sequence_identity"] = threshold
    rows["cluster_size"] = rows.groupby("cluster_id")["entry"].transform("size")
    return rows.sort_values(["cluster_id", "entry"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-threshold MMseqs2 clusters for terpene synthase candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--thresholds", default=",".join(str(value) for value in DEFAULT_THRESHOLDS))
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--mmseqs", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    thresholds = tuple(float(value.strip()) for value in args.thresholds.split(",") if value.strip())
    if not thresholds or any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("Thresholds must be in (0, 1].")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mmseqs = resolve_mmseqs(args.mmseqs)
    fasta_path = output_dir / "candidate_sequences.fasta"
    candidates = write_fasta(args.input.resolve(), fasta_path)
    universe = set(candidates["Entry"])

    summaries = []
    combined = []
    for threshold in thresholds:
        cluster_tsv = run_cluster(mmseqs, fasta_path, output_dir, threshold, args.threads, args.force)
        assignments = parse_clusters(cluster_tsv, threshold, universe)
        tag = f"id{int(round(threshold * 100)):02d}"
        assignment_path = output_dir / f"clusters_{tag}.csv"
        assignments.to_csv(assignment_path, index=False)
        combined.append(assignments)
        sizes = assignments.groupby("cluster_id")["entry"].size()
        summaries.append(
            {
                "threshold": threshold,
                "n_sequences": int(len(assignments)),
                "n_clusters": int(sizes.size),
                "n_singletons": int((sizes == 1).sum()),
                "largest_cluster": int(sizes.max()),
                "median_cluster_size": float(sizes.median()),
                "assignments": str(assignment_path),
                "raw_cluster_tsv": str(cluster_tsv),
            }
        )

    pd.concat(combined, ignore_index=True).to_csv(output_dir / "clusters_all_thresholds.csv", index=False)
    summary = {
        "input": str(args.input.resolve()),
        "mmseqs": str(mmseqs),
        "n_unique_candidate_ids": int(candidates["Entry"].nunique()),
        "n_unique_sequences": int(candidates["Sequence"].nunique()),
        "thresholds": summaries,
    }
    (output_dir / "cluster_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
