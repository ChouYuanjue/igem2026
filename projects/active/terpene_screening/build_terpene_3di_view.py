from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTEINS = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_COVERAGE = ROOT / "data/terpene_3di_v1/afdb_coverage.csv"
DEFAULT_OUTPUT = ROOT / "data/terpene_3di_v1/view"
DEFAULT_EXISTING_AFDB = ROOT / "results/clipzyme_native_extension_v1/structures/af_v6"
DEFAULT_FOLDSEEK = ROOT / "tools/external/foldseek/bin/foldseek"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def download_one(url: str, destination: Path) -> tuple[str, bool, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination), True, "cached"
    tmp = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if tmp.stat().st_size <= 0:
            raise RuntimeError("empty download")
        os.replace(tmp, destination)
        return str(destination), True, "downloaded"
    except Exception as exc:  # pragma: no cover - network dependent
        tmp.unlink(missing_ok=True)
        return str(destination), False, repr(exc)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def parse_foldseek_tsv(path: Path, accession_to_row: dict[str, int], n: int) -> tuple[np.ndarray, dict[str, int]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query", "target", "evalue", "bits", "fident", "alnlen"],
        dtype={"query": str, "target": str},
    )
    frame["bits"] = pd.to_numeric(frame.bits, errors="coerce").fillna(0.0)
    frame["evalue"] = pd.to_numeric(frame.evalue, errors="coerce").fillna(np.inf)
    frame["fident"] = pd.to_numeric(frame.fident, errors="coerce").fillna(0.0)
    frame["alnlen"] = pd.to_numeric(frame.alnlen, errors="coerce").fillna(0).astype(int)

    def key_to_row(value: str) -> int | None:
        # Input files are named by UniProt accession: <accession>.cif.
        key = str(value)
        for suffix in (".cif", ".pdb"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        return accession_to_row.get(key)

    qrows = frame["query"].map(key_to_row)
    trows = frame["target"].map(key_to_row)
    keep = qrows.notna() & trows.notna()
    frame = frame.loc[keep].copy()
    frame["qrow"] = qrows.loc[keep].astype(int)
    frame["trow"] = trows.loc[keep].astype(int)

    self_bits = np.zeros(n, dtype=np.float64)
    self_rows = frame[frame.qrow.eq(frame.trow)]
    for row in self_rows.itertuples(index=False):
        self_bits[int(row.qrow)] = max(self_bits[int(row.qrow)], float(row.bits))

    similarity = np.zeros((n, n), dtype=np.float32)
    np.fill_diagonal(similarity, 1.0)
    directional: dict[tuple[int, int], float] = {}
    for row in frame.itertuples(index=False):
        i, j = int(row.qrow), int(row.trow)
        if i == j:
            continue
        denom = float(np.sqrt(self_bits[i] * self_bits[j]))
        if denom <= 0:
            continue
        score = float(np.clip(float(row.bits) / denom, 0.0, 1.0))
        key = (i, j)
        directional[key] = max(directional.get(key, 0.0), score)
    # Search is top-k directional. Symmetrize with max so a relationship found from
    # either endpoint is retained instead of treating truncation as negative evidence.
    for (i, j), score in directional.items():
        symmetric = max(score, directional.get((j, i), 0.0))
        if symmetric > similarity[i, j]:
            similarity[i, j] = symmetric
            similarity[j, i] = symmetric

    stats = {
        "alignment_rows": int(len(frame)),
        "self_score_rows": int(np.sum(self_bits > 0)),
        "directed_nonself_edges": int(len(directional)),
        "symmetric_nonzero_edges": int((np.count_nonzero(similarity) - n) // 2),
    }
    return similarity, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a low-cost AFDB/Foldseek structural geometry view for MARTS proteins.")
    parser.add_argument("--proteins", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--existing-afdb", type=Path, default=DEFAULT_EXISTING_AFDB)
    parser.add_argument("--foldseek", type=Path, default=DEFAULT_FOLDSEEK)
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--max-seqs", type=int, default=64)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "structures"
    input_dir = output / "foldseek_input"
    foldseek_dir = output / "foldseek"
    raw_dir.mkdir(exist_ok=True)
    input_dir.mkdir(exist_ok=True)
    foldseek_dir.mkdir(exist_ok=True)

    proteins = pd.read_csv(args.proteins, dtype=str).fillna("")
    coverage = pd.read_csv(args.coverage, dtype=str).fillna("")
    coverage["http_status"] = pd.to_numeric(coverage.http_status, errors="coerce").fillna(0).astype(int)
    if len(proteins) != len(coverage) or set(proteins.protein_id) != set(coverage.protein_id):
        raise ValueError("coverage manifest does not align to protein universe")
    coverage = proteins[["protein_id"]].merge(coverage, on="protein_id", how="left", validate="one_to_one")

    source_rows: list[dict[str, object]] = []
    downloads: list[tuple[str, Path]] = []
    for row in coverage.itertuples(index=False):
        accession = str(row.uniprot)
        available = int(row.http_status) == 200 and bool(accession)
        destination = raw_dir / f"{accession}.cif" if accession else raw_dir / "__missing__.cif"
        existing = args.existing_afdb.resolve() / f"AF-{accession}-F1-model_v6.cif" if accession else Path("")
        source = "unavailable"
        path = Path("")
        if available and existing.is_file():
            source = "existing_cache"
            path = existing
        elif available and destination.is_file() and destination.stat().st_size > 0:
            source = "download_cache"
            path = destination
        elif available and not args.skip_download:
            downloads.append((str(row.url), destination))
            source = "pending_download"
            path = destination
        source_rows.append(
            {
                "protein_id": str(row.protein_id),
                "uniprot": accession,
                "available": available,
                "source": source,
                "path": str(path) if available else "",
                "url": str(row.url),
            }
        )

    if downloads:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            futures = [executor.submit(download_one, url, dest) for url, dest in downloads]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        failures = [result for result in results if not result[1]]
        if failures:
            print(f"WARNING: {len(failures)} downloads failed; they remain unavailable", flush=True)

    source_frame = pd.DataFrame(source_rows)
    actual_available = np.zeros(len(proteins), dtype=bool)
    accession_to_row: dict[str, int] = {}
    for index, row in source_frame.iterrows():
        accession = str(row.uniprot)
        if not bool(row.available) or not accession:
            continue
        existing = args.existing_afdb.resolve() / f"AF-{accession}-F1-model_v6.cif"
        downloaded = raw_dir / f"{accession}.cif"
        chosen = existing if existing.is_file() else downloaded if downloaded.is_file() else None
        if chosen is None:
            continue
        actual_available[index] = True
        accession_to_row[accession] = index
        link_or_copy(chosen, input_dir / f"{accession}.cif")
        source_frame.loc[index, "path"] = str(chosen)
        source_frame.loc[index, "source"] = "existing_cache" if chosen == existing else "download_cache"

    source_frame["actual_available"] = actual_available
    source_frame.to_csv(output / "structure_manifest.csv", index=False)
    np.save(output / "available.npy", actual_available)

    if not np.any(actual_available):
        raise RuntimeError("no structures available")
    foldseek = args.foldseek.resolve()
    if not foldseek.is_file():
        raise FileNotFoundError(foldseek)
    version = subprocess.check_output([str(foldseek), "version"], text=True).strip()

    db = foldseek_dir / "db"
    result_db = foldseek_dir / "result"
    tmp = foldseek_dir / "tmp"
    tsv = foldseek_dir / "neighbors.tsv"
    three_di = foldseek_dir / "3di.fasta"
    for stale in [db, result_db]:
        for path in foldseek_dir.glob(stale.name + "*"):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
    shutil.rmtree(tmp, ignore_errors=True)

    run([str(foldseek), "createdb", str(input_dir), str(db), "--threads", str(args.threads), "-v", "2"])
    run(
        [
            str(foldseek),
            "search",
            str(db),
            str(db),
            str(result_db),
            str(tmp),
            "--max-seqs",
            str(args.max_seqs),
            "-e",
            "1000",
            "-s",
            str(args.sensitivity),
            "--threads",
            str(args.threads),
            "-v",
            "2",
        ]
    )
    run(
        [
            str(foldseek),
            "convertalis",
            str(db),
            str(db),
            str(result_db),
            str(tsv),
            "--format-output",
            "query,target,evalue,bits,fident,alnlen",
            "--threads",
            str(args.threads),
            "-v",
            "2",
        ]
    )

    # db_ss is the Foldseek 3Di alphabet database. Reuse db_h as its FASTA header DB.
    for suffix in ("", ".dbtype", ".index"):
        target = Path(str(db) + "_ss_h" + suffix)
        source = Path(str(db) + "_h" + suffix)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.name)
    run([str(foldseek), "convert2fasta", str(db) + "_ss", str(three_di), "-v", "1"])

    similarity, alignment_stats = parse_foldseek_tsv(tsv, accession_to_row, len(proteins))
    unavailable = ~actual_available
    similarity[unavailable, :] = 0.0
    similarity[:, unavailable] = 0.0
    np.save(output / "similarity.npy", similarity)

    manifest = {
        "version": "terpene-3di-structural-geometry-v1",
        "semantic_role": "explicit structure-derived protein geometry for relation memory; missing structure means no structural evidence",
        "protein_count": int(len(proteins)),
        "available_count": int(actual_available.sum()),
        "available_fraction": float(actual_available.mean()),
        "source": "AlphaFold DB v6 cached structures; Foldseek 3Di structural search",
        "foldseek_version": version,
        "foldseek_max_seqs": int(args.max_seqs),
        "foldseek_sensitivity": float(args.sensitivity),
        "similarity": "bits(q,t)/sqrt(bits(q,q)*bits(t,t)), clipped to [0,1], max-symmetrized",
        "input_sha256": {
            "protein_entities.csv": sha256_file(args.proteins.resolve()),
            "afdb_coverage.csv": sha256_file(args.coverage.resolve()),
        },
        "outputs": {
            "similarity": str(output / "similarity.npy"),
            "available": str(output / "available.npy"),
            "structure_manifest": str(output / "structure_manifest.csv"),
            "neighbors": str(tsv),
            "three_di_fasta": str(three_di),
        },
        **alignment_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
