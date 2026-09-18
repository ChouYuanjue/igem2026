from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ot
from Bio.PDB import PDBIO

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_repos.EnzymeCAGE.scripts.extract_p2rank_pockets import (  # noqa: E402
    ResidueSelect,
    build_structure_mappings,
    load_structure,
    normalize_columns,
    resolve_prediction_file,
    resolve_residue,
)
from projects.active.terpene_screening.build_terpene_3di_view import (  # noqa: E402
    run,
    sha256_file,
)

DEFAULT_PROTEINS = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_CANDIDATES = ROOT / "data/terpene_p2rank_current_v1/candidates.csv"
DEFAULT_RAW = ROOT / "data/terpene_p2rank_current_v1/_p2rank_stage/raw"
DEFAULT_OUTPUT = ROOT / "data/terpene_pocket_ot_v1/view"
DEFAULT_FOLDSEEK = ROOT / "tools/external/foldseek/bin/foldseek"


def parse_instance_foldseek_tsv(
    path: Path, instance_to_index: dict[str, int], n: int, chunk_size: int = 500_000
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Parse exhaustive Foldseek output without treating unrepresentable pockets as negatives.

    Foldseek may omit extremely short pockets entirely (in practice 2--3 residues).
    Such instances have no self score and therefore no defined structural ground cost.
    They are returned as invalid observations and are removed from the protein measure
    before mass renormalization. Remaining observed pockets must form a complete
    exhaustive Cartesian product.
    """
    names = ["query", "target", "evalue", "bits", "fident", "alnlen"]

    def clean(series: pd.Series) -> pd.Series:
        return series.astype(str).str.replace(r"\.(?:pdb|cif)$", "", regex=True)

    self_bits = np.zeros(n, dtype=np.float64)
    row_count = 0
    # First pass: determine which pocket instances possess a valid structural self score.
    for chunk in pd.read_csv(path, sep="\t", header=None, names=names, dtype={"query": str, "target": str}, chunksize=chunk_size):
        row_count += len(chunk)
        q = clean(chunk["query"]); t = clean(chunk["target"])
        same = q.eq(t)
        if not same.any():
            continue
        bits = pd.to_numeric(chunk.loc[same, "bits"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        for key, bit in zip(q.loc[same].tolist(), bits):
            idx = instance_to_index.get(str(key))
            if idx is not None:
                self_bits[idx] = max(self_bits[idx], float(bit))
    observed = self_bits > 0
    observed_indices = np.flatnonzero(observed)
    observed_count = int(observed.sum())
    if observed_count == 0:
        raise RuntimeError("Foldseek produced no positive pocket self scores")

    similarity = np.zeros((n, n), dtype=np.float32)
    similarity[observed_indices, observed_indices] = 1.0
    seen = np.zeros((n, n), dtype=bool)
    # Second pass: fill all observed directed ground similarities. Exhaustive search
    # should give exactly observed_count^2 rows after unrepresentable pockets vanish.
    for chunk in pd.read_csv(path, sep="\t", header=None, names=names, dtype={"query": str, "target": str}, chunksize=chunk_size):
        q = clean(chunk["query"]); t = clean(chunk["target"])
        qi = q.map(instance_to_index); ti = t.map(instance_to_index)
        keep = qi.notna() & ti.notna()
        if not keep.any():
            continue
        qv = qi.loc[keep].astype(int).to_numpy(); tv = ti.loc[keep].astype(int).to_numpy()
        bits = pd.to_numeric(chunk.loc[keep, "bits"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        valid = observed[qv] & observed[tv]
        qv = qv[valid]; tv = tv[valid]; bits = bits[valid]
        denom = np.sqrt(self_bits[qv] * self_bits[tv])
        scores = np.clip(np.divide(bits, denom, out=np.zeros_like(bits), where=denom > 0), 0.0, 1.0).astype(np.float32)
        # convertalis is one row per exhaustive ordered pair here; maximum.at keeps
        # this correct even if a future Foldseek version emits duplicate alignments.
        np.maximum.at(similarity, (qv, tv), scores)
        seen[qv, tv] = True

    expected_rows = observed_count * observed_count
    observed_rows = int(seen[np.ix_(observed_indices, observed_indices)].sum())
    if observed_rows != expected_rows:
        raise RuntimeError(
            f"incomplete exhaustive Foldseek geometry: observed {observed_rows}/{expected_rows} ordered pocket pairs"
        )
    similarity = np.maximum(similarity, similarity.T)
    similarity[~observed, :] = 0.0; similarity[:, ~observed] = 0.0
    directed_nonself = observed_rows - observed_count
    return similarity, observed, {
        "alignment_rows": int(row_count),
        "self_score_rows": observed_count,
        "unrepresentable_pocket_count": int((~observed).sum()),
        "observed_cartesian_rows": observed_rows,
        "expected_observed_cartesian_rows": expected_rows,
        "directed_nonself_edges": int(directed_nonself),
        "directed_pair_coverage": 1.0,
        "symmetric_nonzero_edges": int((np.count_nonzero(similarity) - observed_count) // 2),
    }


def extract_top_pockets(
    candidates: pd.DataFrame,
    raw_dir: Path,
    pocket_dir: Path,
    top_k: int,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Materialize rank-1/2/3 pockets with the same residue semantics as EnzymeCAGE.

    The optimization is purely operational: each protein's residue mapping and parsed
    structure are constructed once, rather than once per pocket rank.
    """
    pocket_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for cand in candidates.itertuples(index=False):
        uid = str(cand.UniprotID)
        protein_id = str(cand.protein_id)
        structure_path = Path(str(cand.structure_path)).resolve()
        try:
            residue_csv = resolve_prediction_file(raw_dir, uid, "_residues.csv")
            prediction_csv = resolve_prediction_file(raw_dir, uid, "_predictions.csv")
            df_residue = normalize_columns(pd.read_csv(residue_csv, skipinitialspace=True))
            required = {"chain", "residue_label", "pocket"}
            if not required.issubset(df_residue.columns):
                raise ValueError(f"residue table missing columns {sorted(required - set(df_residue.columns))}")
            df_residue["pocket"] = pd.to_numeric(df_residue["pocket"], errors="coerce")
            df_prediction = normalize_columns(pd.read_csv(prediction_csv, skipinitialspace=True))
            if "rank" not in df_prediction.columns:
                raise ValueError("prediction table has no rank column")
            df_prediction["rank"] = pd.to_numeric(df_prediction["rank"], errors="coerce")
            chain_map, label_map = build_structure_mappings(structure_path)
            structure = load_structure(structure_path)
        except Exception as exc:
            failures.append({"UniprotID": uid, "protein_id": protein_id, "pocket_rank": "", "error": repr(exc)})
            continue

        for rank in range(1, top_k + 1):
            try:
                pocket_rows = df_residue[df_residue["pocket"] == rank]
                if pocket_rows.empty:
                    raise ValueError(f"no residues assigned to pocket rank {rank}")
                selected_residues = []
                for row in pocket_rows.itertuples(index=False):
                    chain_id = str(getattr(row, "chain")).strip()
                    residue_label = str(getattr(row, "residue_label")).strip()
                    selected_residues.append(resolve_residue(chain_id, residue_label, chain_map, label_map))
                dedup = []
                seen = set()
                for residue in selected_residues:
                    k = (residue.get_parent().id, residue.id)
                    if k not in seen:
                        seen.add(k)
                        dedup.append(residue)
                dedup.sort(key=lambda residue: (residue.get_parent().id, residue.id[1], residue.id[2].strip()))
                selected_keys = {(residue.get_parent().id, residue.id) for residue in dedup}
                pocket_residues = [str(residue.id[1]) for residue in dedup]

                pred_rows = df_prediction[df_prediction["rank"] == rank]
                if pred_rows.empty:
                    raise ValueError(f"no P2Rank prediction metadata for pocket rank {rank}")
                metadata = pred_rows.iloc[0].to_dict()
                probability = pd.to_numeric(pd.Series([metadata.get("probability", np.nan)]), errors="coerce").iloc[0]
                if not np.isfinite(probability) or float(probability) <= 0:
                    raise ValueError(f"missing/nonpositive P2Rank probability for pocket rank {rank}: {probability!r}")

                instance_id = f"{uid}__p{rank}"
                out = pocket_dir / f"{instance_id}.pdb"
                io = PDBIO()
                io.set_structure(structure)
                io.save(str(out), ResidueSelect(selected_keys))
                if not out.is_file() or out.stat().st_size <= 0:
                    raise RuntimeError("empty pocket PDB")
                rows.append(
                    {
                        "instance_id": instance_id,
                        "UniprotID": uid,
                        "protein_id": protein_id,
                        "pocket_rank": int(rank),
                        "p2rank_probability": float(probability),
                        "p2rank_score": metadata.get("score", ""),
                        "pocket_residues": ",".join(pocket_residues),
                        "structure_path": str(structure_path),
                        "pocket_path": str(out.resolve()),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "UniprotID": uid,
                        "protein_id": protein_id,
                        "pocket_rank": str(rank),
                        "error": repr(exc),
                    }
                )
    return pd.DataFrame(rows), failures


def build_protein_ot_similarity(
    proteins: pd.DataFrame,
    pockets: pd.DataFrame,
    pocket_similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    protein_to_row = {str(pid): i for i, pid in enumerate(proteins.protein_id.astype(str))}
    instance_to_index = {str(x): i for i, x in enumerate(pockets.instance_id.astype(str))}

    grouped: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    available = np.zeros(len(proteins), dtype=bool)
    mass_table = np.zeros((len(proteins), 3), dtype=np.float32)
    for protein_id, group in pockets.groupby("protein_id", sort=False):
        pidx = protein_to_row.get(str(protein_id))
        if pidx is None:
            continue
        group = group.sort_values("pocket_rank")
        indices = np.asarray([instance_to_index[str(x)] for x in group.instance_id], dtype=np.int64)
        weights = group.p2rank_probability.astype(float).to_numpy()
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            continue
        weights = weights / total
        grouped[int(pidx)] = (indices, weights)
        available[int(pidx)] = True
        for rank, weight in zip(group.pocket_rank.astype(int), weights):
            if 1 <= int(rank) <= 3:
                mass_table[int(pidx), int(rank) - 1] = float(weight)

    similarity = np.zeros((len(proteins), len(proteins)), dtype=np.float32)
    available_rows = np.flatnonzero(available)
    for ai, i in enumerate(available_rows):
        ii, wi = grouped[int(i)]
        similarity[i, i] = 1.0
        for j in available_rows[ai + 1 :]:
            jj, wj = grouped[int(j)]
            ground_cost = 1.0 - pocket_similarity[np.ix_(ii, jj)].astype(np.float64)
            np.clip(ground_cost, 0.0, 1.0, out=ground_cost)
            cost = float(ot.emd2(wi.astype(np.float64), wj.astype(np.float64), ground_cost, numItermax=10000))
            score = float(np.clip(1.0 - cost, 0.0, 1.0))
            similarity[i, j] = score
            similarity[j, i] = score

    stats = {
        "available_count": int(available.sum()),
        "available_fraction": float(available.mean()),
        "available_pair_count": int(len(available_rows) * (len(available_rows) - 1) // 2),
    }
    return similarity, available, mass_table, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build top-3 P2Rank pocket-measure geometry using exact balanced Kantorovich transport.")
    parser.add_argument("--proteins", type=Path, default=DEFAULT_PROTEINS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--foldseek", type=Path, default=DEFAULT_FOLDSEEK)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--limit-proteins", type=int, default=0, help="Debug/smoke only; 0 means full current structured universe.")
    parser.add_argument("--postprocess-existing", action="store_true", help="Reuse existing pocket_instance_manifest.csv and exhaustive Foldseek TSV; skip extraction/search.")
    args = parser.parse_args()
    if args.top_k != 3:
        raise ValueError("v1 is frozen to top_k=3")

    proteins = pd.read_csv(args.proteins, dtype=str).fillna("")
    candidates = pd.read_csv(args.candidates, dtype=str).fillna("")
    required = {"UniprotID", "protein_id", "structure_path"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"candidates missing {sorted(required - set(candidates.columns))}")
    if args.limit_proteins > 0:
        candidates = candidates.head(args.limit_proteins).copy()

    output = args.output_dir.resolve()
    pocket_dir = output / "pockets_top3"
    input_dir = output / "foldseek_input"
    foldseek_dir = output / "foldseek"
    output.mkdir(parents=True, exist_ok=True)
    foldseek = args.foldseek.resolve()
    if not foldseek.is_file():
        raise FileNotFoundError(foldseek)
    version = subprocess.check_output([str(foldseek), "version"], text=True).strip()
    tsv = foldseek_dir / "all_pairs.tsv"
    if args.postprocess_existing:
        pockets = pd.read_csv(output / "pocket_instance_manifest.csv", dtype=str).fillna("")
        pockets["pocket_rank"] = pd.to_numeric(pockets.pocket_rank, errors="raise").astype(int)
        pockets["p2rank_probability"] = pd.to_numeric(pockets.p2rank_probability, errors="raise").astype(float)
        failures_path = output / "pocket_instance_failures.csv"
        failures = pd.read_csv(failures_path, dtype=str).fillna("").to_dict("records") if failures_path.is_file() else []
        if not tsv.is_file():
            raise FileNotFoundError(f"postprocess requested but missing {tsv}")
    else:
        shutil.rmtree(pocket_dir, ignore_errors=True)
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(foldseek_dir, ignore_errors=True)
        pocket_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        foldseek_dir.mkdir(parents=True, exist_ok=True)
        pockets, failures = extract_top_pockets(candidates, args.raw_dir.resolve(), pocket_dir, args.top_k)
        if pockets.empty:
            raise RuntimeError("no top-3 pocket instances extracted")
        pockets = pockets.sort_values(["protein_id", "pocket_rank", "UniprotID"]).reset_index(drop=True)
        pockets.to_csv(output / "pocket_instance_manifest.csv", index=False)
        pd.DataFrame(failures).to_csv(output / "pocket_instance_failures.csv", index=False)
        for row in pockets.itertuples(index=False):
            target = input_dir / f"{row.instance_id}.pdb"
            target.symlink_to(Path(str(row.pocket_path)).resolve())
        n_pockets = len(pockets)
        db = foldseek_dir / "db"; result_db = foldseek_dir / "result"; tmp = foldseek_dir / "tmp"
        run([str(foldseek), "createdb", str(input_dir), str(db), "--threads", str(args.threads), "-v", "2"])
        run([str(foldseek), "search", str(db), str(db), str(result_db), str(tmp), "--exhaustive-search", "1", "--max-seqs", str(n_pockets), "-e", "1000000", "--threads", str(args.threads), "-v", "2"])
        run([str(foldseek), "convertalis", str(db), str(db), str(result_db), str(tsv), "--format-output", "query,target,evalue,bits,fident,alnlen", "--threads", str(args.threads), "-v", "2"])

    n_pockets = len(pockets)
    instance_to_index = {str(x): i for i, x in enumerate(pockets.instance_id.astype(str))}
    pocket_similarity, pocket_observed, foldseek_stats = parse_instance_foldseek_tsv(tsv, instance_to_index, n_pockets)
    pockets["foldseek_observed"] = pocket_observed
    pockets.to_csv(output / "pocket_instance_manifest.csv", index=False)
    np.save(output / "pocket_similarity.npy", pocket_similarity)
    observed_pockets = pockets.loc[pocket_observed].copy().reset_index(drop=True)
    observed_similarity = pocket_similarity[np.ix_(pocket_observed, pocket_observed)]
    protein_similarity, available, mass_table, ot_stats = build_protein_ot_similarity(proteins, observed_pockets, observed_similarity)
    np.save(output / "similarity.npy", protein_similarity)
    np.save(output / "available.npy", available)
    np.save(output / "pocket_mass.npy", mass_table)

    rank_counts = pockets.groupby("protein_id").size()
    manifest = {
        "version": "terpene-pocket-kantorovich-geometry-v1",
        "semantic_role": (
            "Each structured protein is a discrete probability measure over its top-3 P2Rank pockets. "
            "Pocket ground similarity is exhaustive Foldseek self-normalized bit similarity; ground cost is 1-s. "
            "Protein structural affinity is 1 minus exact balanced Kantorovich transport cost. "
            "The Foldseek-derived ground cost is not asserted to be a metric. Missing structure is absent evidence."
        ),
        "protein_count": int(len(proteins)),
        "candidate_structure_count": int(len(candidates)),
        "pocket_instance_count": int(len(pockets)),
        "foldseek_observed_pocket_count": int(pocket_observed.sum()),
        "foldseek_unrepresentable_pocket_count": int((~pocket_observed).sum()),
        "proteins_with_1_pocket": int((rank_counts == 1).sum()),
        "proteins_with_2_pockets": int((rank_counts == 2).sum()),
        "proteins_with_3_pockets": int((rank_counts == 3).sum()),
        "pocket_extraction_failure_count": int(len(failures)),
        "pocket_mass": "P2Rank probability normalized over structurally observed top-3 pockets per protein; Foldseek-unrepresentable pockets are absent observations, not negative evidence",
        "pocket_similarity": "bits(q,t)/sqrt(bits(q,q)*bits(t,t)), clipped to [0,1], max-symmetrized",
        "ground_cost": "1 - pocket_similarity",
        "transport": "exact balanced Kantorovich optimal transport via POT ot.emd2; no entropic regularization",
        "protein_similarity": "1 - transport_cost, clipped to [0,1]",
        "extraction_execution": "one structure parse and one residue mapping per protein; ranks 1/2/3 are cropped from the same parsed structure",
        "foldseek_version": version,
        "foldseek_search": {"exhaustive_search": 1, "max_seqs": int(n_pockets), "evalue": 1000000},
        "input_sha256": {
            "protein_entities.csv": sha256_file(args.proteins.resolve()),
            "candidates.csv": sha256_file(args.candidates.resolve()),
        },
        "outputs": {
            "similarity": str(output / "similarity.npy"),
            "available": str(output / "available.npy"),
            "pocket_mass": str(output / "pocket_mass.npy"),
            "pocket_similarity": str(output / "pocket_similarity.npy"),
            "pocket_instance_manifest": str(output / "pocket_instance_manifest.csv"),
            "pocket_instance_failures": str(output / "pocket_instance_failures.csv"),
            "foldseek_all_pairs": str(tsv),
        },
        **foldseek_stats,
        **ot_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
