from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_PROTEINS = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_CANDIDATES = ROOT / "data/terpene_p2rank_current_v1/candidates.csv"
DEFAULT_POCKET_MANIFEST = ROOT / "results/terpene_p2rank_current_v1/p2rank_pocket_manifest.csv"
DEFAULT_OUTPUT = ROOT / "data/terpene_pocket_3di_v1/view"
DEFAULT_FOLDSEEK = ROOT / "tools/external/foldseek/bin/foldseek"

from projects.active.fibre.pipelines.whole_structure import (  # noqa: E402
    parse_foldseek_tsv,
    run,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a current-MARTS P2Rank-top1 pocket-local Foldseek/3Di geometry view."
    )
    parser.add_argument("--proteins", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--pocket-manifest", type=Path, default=DEFAULT_POCKET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--foldseek", type=Path, default=DEFAULT_FOLDSEEK)
    parser.add_argument("--max-seqs", type=int, default=64)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    proteins = pd.read_csv(args.proteins, dtype=str).fillna("")
    candidates = pd.read_csv(args.candidates, dtype=str).fillna("")
    pockets = pd.read_csv(args.pocket_manifest, dtype=str).fillna("")
    if not {"UniprotID", "protein_id"}.issubset(candidates.columns):
        raise ValueError("candidate manifest requires UniprotID and protein_id")
    if "UniprotID" not in pockets.columns:
        raise ValueError("P2Rank manifest requires UniprotID")

    candidate_by_uid = (
        candidates[["UniprotID", "protein_id"]]
        .drop_duplicates()
        .set_index("UniprotID")["protein_id"]
        .to_dict()
    )
    protein_to_row = {str(pid): i for i, pid in enumerate(proteins.protein_id.astype(str))}

    output = args.output_dir.resolve()
    input_dir = output / "foldseek_input"
    foldseek_dir = output / "foldseek"
    output.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(input_dir, ignore_errors=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    foldseek_dir.mkdir(parents=True, exist_ok=True)

    available = np.zeros(len(proteins), dtype=bool)
    uid_to_row: dict[str, int] = {}
    mapping_rows: list[dict[str, object]] = []
    seen_rows: set[int] = set()

    # Keep only successful top-1 pocket files.  Missing/failed pockets are absent
    # geometric observations, never negative evidence.
    for row in pockets.itertuples(index=False):
        uid = str(row.UniprotID)
        protein_id = str(candidate_by_uid.get(uid, ""))
        pidx = protein_to_row.get(protein_id)
        pocket_path = Path(str(getattr(row, "pocket_pdb_path", "") or getattr(row, "pocket_path", "")))
        ok = (
            pidx is not None
            and pidx not in seen_rows
            and str(getattr(row, "status", "ok")).lower() in {"", "ok", "success"}
            and pocket_path.is_file()
            and pocket_path.stat().st_size > 0
        )
        mapping_rows.append(
            {
                "protein_id": protein_id,
                "uniprot": uid,
                "row": -1 if pidx is None else int(pidx),
                "pocket_path": str(pocket_path) if pocket_path else "",
                "available": bool(ok),
                "pocket_score": str(getattr(row, "pocket_score", "")),
                "pocket_probability": str(getattr(row, "pocket_probability", "")),
            }
        )
        if not ok:
            continue
        target = input_dir / f"{uid}.pdb"
        # The pocket files already live in our current-version asset directory;
        # a symlink is sufficient because Foldseek uses the visible input basename.
        target.symlink_to(pocket_path.resolve())
        available[int(pidx)] = True
        uid_to_row[uid] = int(pidx)
        seen_rows.add(int(pidx))

    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(output / "pocket_structure_manifest.csv", index=False)
    np.save(output / "available.npy", available)
    if not np.any(available):
        raise RuntimeError("no pocket structures available")

    foldseek = args.foldseek.resolve()
    if not foldseek.is_file():
        raise FileNotFoundError(foldseek)
    version = subprocess.check_output([str(foldseek), "version"], text=True).strip()

    db = foldseek_dir / "db"
    result_db = foldseek_dir / "result"
    tmp = foldseek_dir / "tmp"
    tsv = foldseek_dir / "neighbors.tsv"
    three_di = foldseek_dir / "3di.fasta"
    for prefix in [db.name, result_db.name]:
        for path in foldseek_dir.glob(prefix + "*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    shutil.rmtree(tmp, ignore_errors=True)

    run([str(foldseek), "createdb", str(input_dir), str(db), "--threads", str(args.threads), "-v", "2"])
    run(
        [
            str(foldseek), "search", str(db), str(db), str(result_db), str(tmp),
            "--max-seqs", str(args.max_seqs), "-e", "1000", "-s", str(args.sensitivity),
            "--threads", str(args.threads), "-v", "2",
        ]
    )
    run(
        [
            str(foldseek), "convertalis", str(db), str(db), str(result_db), str(tsv),
            "--format-output", "query,target,evalue,bits,fident,alnlen",
            "--threads", str(args.threads), "-v", "2",
        ]
    )

    for suffix in ("", ".dbtype", ".index"):
        target = Path(str(db) + "_ss_h" + suffix)
        source = Path(str(db) + "_h" + suffix)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.name)
    run([str(foldseek), "convert2fasta", str(db) + "_ss", str(three_di), "-v", "1"])

    similarity, stats = parse_foldseek_tsv(tsv, uid_to_row, len(proteins))
    unavailable = ~available
    similarity[unavailable, :] = 0.0
    similarity[:, unavailable] = 0.0
    np.save(output / "similarity.npy", similarity)

    manifest = {
        "version": "terpene-pocket-3di-geometry-v1",
        "semantic_role": (
            "P2Rank top-1 catalytic-local structural geometry for positive relation memory; "
            "missing pocket means no pocket-geometric evidence"
        ),
        "protein_count": int(len(proteins)),
        "available_count": int(available.sum()),
        "available_fraction": float(available.mean()),
        "source": "current MARTS AFDB-v6 structures -> P2Rank 2.5.1 top-1 cropped pocket -> Foldseek structural search",
        "foldseek_version": version,
        "foldseek_max_seqs": int(args.max_seqs),
        "foldseek_sensitivity": float(args.sensitivity),
        "similarity": "bits(q,t)/sqrt(bits(q,q)*bits(t,t)), clipped to [0,1], max-symmetrized",
        "input_sha256": {
            "protein_entities.csv": sha256_file(args.proteins.resolve()),
            "candidates.csv": sha256_file(args.candidates.resolve()),
            "p2rank_pocket_manifest.csv": sha256_file(args.pocket_manifest.resolve()),
        },
        "outputs": {
            "similarity": str(output / "similarity.npy"),
            "available": str(output / "available.npy"),
            "pocket_structure_manifest": str(output / "pocket_structure_manifest.csv"),
            "neighbors": str(tsv),
            "three_di_fasta": str(three_di),
        },
        **stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
