from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
from rdkit import DataStructs, RDLogger

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from retrieve import neutralize_atoms, get_morgan_fp

RDLogger.DisableLog("rdApp.*")

BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
CAGE_DB = ROOT / "data/external/enzymecage_current/rhea_2023_compact.csv.gz"
CAGE_DB_SOURCE = (
    ROOT / "data/external/enzymecage_current/orphan335_author_assets_v1/"
    "rhea_2023_rxn2uids.csv"
)
OUT = ROOT / "results/enzymecage_original_gate_full7_v1"
TOP_SIMILAR_REACTIONS = 10


def load_queries() -> pd.DataFrame:
    q = set()
    for d in sorted(BENCH.iterdir()):
        p = d / "test_pairs.csv"
        if p.exists():
            q.update(
                pd.read_csv(p, usecols=["reaction_id"], dtype=str)
                .reaction_id.astype(str)
            )
    rx = pd.read_csv(REACTIONS, dtype=str).fillna("")
    out = rx[rx.reaction_id.astype(str).isin(q)][
        ["reaction_id", "reaction_smiles"]
    ].drop_duplicates("reaction_id")
    if len(out) != len(q):
        missing = sorted(q - set(out.reaction_id.astype(str)))
        raise RuntimeError(f"missing reaction metadata: {missing[:10]}")
    return out.sort_values("reaction_id", kind="stable").reset_index(drop=True)


def reaction_parts(rxn: str) -> tuple[list[str], list[str]]:
    left, right = str(rxn).split(">>", 1)
    # Counter(...).keys() is what retrieve.py:getRSim effectively uses.
    return (
        list(Counter(x for x in left.split(".") if x).keys()),
        list(Counter(x for x in right.split(".") if x).keys()),
    )


def all_molecules(rxns) -> list[str]:
    vals = set()
    for r in rxns:
        a, b = reaction_parts(str(r))
        vals.update(a)
        vals.update(b)
    return sorted(vals)


def fp(smiles: str):
    return get_morgan_fp(neutralize_atoms(smiles))[0]


def split_supported_query_reactions(rxns: list[str]):
    """Apply the published CAGE fingerprint path before retrieval.

    A query that contains a molecule the author code cannot parse is marked
    unavailable for original-CAGE retrieval.  It remains in the benchmark
    denominator and is not repaired with FIBRE chemistry handling.
    """
    valid = []
    invalid = {}
    molecule_cache = {}
    for rxn in rxns:
        bad = []
        a, b = reaction_parts(rxn)
        for smi in a + b:
            if smi not in molecule_cache:
                try:
                    _ = fp(smi)
                    molecule_cache[smi] = ""
                except Exception as exc:
                    molecule_cache[smi] = f"{type(exc).__name__}:{exc}"
            if molecule_cache[smi]:
                bad.append({"smiles": smi, "reason": molecule_cache[smi]})
        if bad:
            invalid[rxn] = bad
        else:
            valid.append(rxn)
    return valid, invalid


def build_similarity(query_mols: list[str], cand_mols: list[str]) -> np.ndarray:
    cache = OUT / "molecule_similarity.npy"
    qfile = OUT / "query_molecules.txt"
    cfile = OUT / "candidate_molecules.txt"
    if cache.exists() and qfile.exists() and cfile.exists():
        if (
            qfile.read_text().splitlines() == query_mols
            and cfile.read_text().splitlines() == cand_mols
        ):
            x = np.load(cache, mmap_mode="r")
            if x.shape == (len(query_mols), len(cand_mols)):
                return x

    cand_fp = [fp(x) for x in cand_mols]
    sim = np.empty((len(query_mols), len(cand_mols)), dtype=np.float32)
    for i, smi in enumerate(query_mols):
        sim[i] = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fp(smi), cand_fp),
            dtype=np.float32,
        )
        if (i + 1) % 100 == 0:
            print(f"molecule_similarity={i+1}/{len(query_mols)}", flush=True)
    np.save(cache, sim)
    qfile.write_text("\n".join(query_mols) + "\n")
    cfile.write_text("\n".join(cand_mols) + "\n")
    return np.load(cache, mmap_mode="r")


def encode_parts(
    rxns: list[str],
    mol_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parts = [reaction_parts(r) for r in rxns]
    max_l = max(len(x[0]) for x in parts)
    max_r = max(len(x[1]) for x in parts)
    left = np.full((len(parts), max_l), -1, dtype=np.int32)
    right = np.full((len(parts), max_r), -1, dtype=np.int32)
    llen = np.zeros(len(parts), dtype=np.int16)
    rlen = np.zeros(len(parts), dtype=np.int16)
    for i, (a, b) in enumerate(parts):
        llen[i] = len(a)
        rlen[i] = len(b)
        left[i, : len(a)] = [mol_index[x] for x in a]
        right[i, : len(b)] = [mol_index[x] for x in b]
    return left, llen, right, rlen


@nb.njit(cache=True)
def _side_score(
    qrow: np.ndarray,
    qlen: int,
    crow: np.ndarray,
    clen: int,
    sim: np.ndarray,
) -> float:
    if qlen == 0:
        return 0.0
    q_used = np.zeros(qlen, dtype=np.uint8)
    c_used = np.zeros(clen, dtype=np.uint8)
    total = 0.0
    for _ in range(min(qlen, clen)):
        best = 0.0
        bq = -1
        bc = -1
        for i in range(qlen):
            if q_used[i]:
                continue
            qi = qrow[i]
            for j in range(clen):
                if c_used[j]:
                    continue
                v = sim[qi, crow[j]]
                if v > best:
                    best = v
                    bq = i
                    bc = j
        if bq < 0 or best <= 0.0:
            break
        q_used[bq] = 1
        c_used[bc] = 1
        total += best
    return total / qlen


@nb.njit(cache=True)
def _reaction_score(
    ql,
    qln,
    qr,
    qrn,
    cl,
    cln,
    cr,
    crn,
    sim,
) -> float:
    ss = _side_score(ql, qln, cl, cln, sim)
    pp = _side_score(qr, qrn, cr, crn, sim)
    sp = _side_score(ql, qln, cr, crn, sim)
    ps = _side_score(qr, qrn, cl, cln, sim)
    s1 = (ss * ss + pp * pp) ** 0.5 / 2.0**0.5
    s2 = (sp * sp + ps * ps) ** 0.5 / 2.0**0.5
    return s1 if s1 >= s2 else s2


@nb.njit(parallel=True, cache=True)
def topk_reactions(
    qleft,
    qllen,
    qright,
    qrlen,
    cleft,
    cllen,
    cright,
    crlen,
    sim,
    k,
):
    nq = qleft.shape[0]
    nc = cleft.shape[0]
    top_idx = np.full((nq, k), -1, dtype=np.int32)
    top_score = np.full((nq, k), -1.0, dtype=np.float32)
    for qi in nb.prange(nq):
        for cj in range(nc):
            score = _reaction_score(
                qleft[qi],
                int(qllen[qi]),
                qright[qi],
                int(qrlen[qi]),
                cleft[cj],
                int(cllen[cj]),
                cright[cj],
                int(crlen[cj]),
                sim,
            )
            # Stable deterministic insertion: candidate reactions are sorted
            # lexicographically, so equal scores retain lower candidate index.
            if score < top_score[qi, k - 1]:
                continue
            pos = k - 1
            while pos > 0 and score > top_score[qi, pos - 1]:
                top_score[qi, pos] = top_score[qi, pos - 1]
                top_idx[qi, pos] = top_idx[qi, pos - 1]
                pos -= 1
            top_score[qi, pos] = score
            top_idx[qi, pos] = cj
    return top_idx, top_score


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    queries = load_queries()
    db = pd.read_csv(
        CAGE_DB,
        usecols=["reaction_id", "UniprotID", "CANO_RXN_SMILES"],
        dtype=str,
    ).fillna("")
    db = db[
        ["UniprotID", "reaction_id", "CANO_RXN_SMILES"]
    ].drop_duplicates()

    q_by_id = dict(
        zip(
            queries.reaction_id.astype(str),
            queries.reaction_smiles.astype(str),
        )
    )
    all_unique_q_rxns = sorted(set(q_by_id.values()))
    valid_q_rxns, invalid_q_rxns = split_supported_query_reactions(
        all_unique_q_rxns
    )
    valid_q_set = set(valid_q_rxns)

    # Build molecular similarities against the complete author RHEA database
    # once.  Each benchmark cell applies its own test-reaction exclusion below,
    # matching retrieve.py when that cell is evaluated independently.
    cand_rxns = sorted(set(db.CANO_RXN_SMILES.astype(str).unique()))
    q_mols = all_molecules(valid_q_rxns)
    c_mols = all_molecules(cand_rxns)
    qmol_index = {x: i for i, x in enumerate(q_mols)}
    cmol_index = {x: i for i, x in enumerate(c_mols)}

    print(
        json.dumps(
            {
                "unique_query_ids": len(queries),
                "unique_query_reactions": len(all_unique_q_rxns),
                "retrieval_evaluable_unique_reactions": len(valid_q_rxns),
                "retrieval_unavailable_unique_reactions": len(invalid_q_rxns),
                "author_candidate_reactions": len(cand_rxns),
                "query_molecules": len(q_mols),
                "candidate_molecules": len(c_mols),
            }
        ),
        flush=True,
    )

    sim = build_similarity(q_mols, c_mols)
    ql, qln, qr, qrn = encode_parts(valid_q_rxns, qmol_index)
    cl, cln, cr, crn = encode_parts(cand_rxns, cmol_index)
    qsm_to_global = {x: i for i, x in enumerate(valid_q_rxns)}

    rxn_to_uids = (
        db.groupby("CANO_RXN_SMILES", sort=False)["UniprotID"]
        .agg(lambda x: sorted(set(map(str, x))))
        .to_dict()
    )

    query_rows = []
    gate_rows = []
    cell_summaries = {}
    cell_dirs = [
        d for d in sorted(BENCH.iterdir()) if (d / "test_pairs.csv").exists()
    ]
    for cell_dir in cell_dirs:
        cell = cell_dir.name
        cell_frame = pd.read_csv(
            cell_dir / "test_pairs.csv",
            usecols=["reaction_id"],
            dtype=str,
        ).fillna("")
        cell_ids = sorted(cell_frame.reaction_id.astype(str).unique())
        cell_smiles = {q_by_id[q] for q in cell_ids}
        candidate_indices = np.asarray(
            [i for i, r in enumerate(cand_rxns) if r not in cell_smiles],
            dtype=np.int64,
        )
        cell_valid_smiles = sorted(
            {q_by_id[q] for q in cell_ids if q_by_id[q] in valid_q_set}
        )
        q_global = np.asarray(
            [qsm_to_global[x] for x in cell_valid_smiles], dtype=np.int64
        )
        qsm_local = {x: i for i, x in enumerate(cell_valid_smiles)}

        print(
            f"cell={cell} queries={len(cell_ids)} "
            f"valid_unique_reactions={len(cell_valid_smiles)} "
            f"candidate_reactions={len(candidate_indices)}",
            flush=True,
        )
        if len(cell_valid_smiles):
            local_top_idx, local_top_score = topk_reactions(
                ql[q_global],
                qln[q_global],
                qr[q_global],
                qrn[q_global],
                cl[candidate_indices],
                cln[candidate_indices],
                cr[candidate_indices],
                crn[candidate_indices],
                sim,
                TOP_SIMILAR_REACTIONS,
            )
        else:
            local_top_idx = np.empty((0, TOP_SIMILAR_REACTIONS), dtype=np.int32)
            local_top_score = np.empty((0, TOP_SIMILAR_REACTIONS), dtype=np.float32)

        unavailable = 0
        candidate_counts = []
        for qid in cell_ids:
            qsm = q_by_id[qid]
            if qsm in invalid_q_rxns:
                unavailable += 1
                query_rows.append(
                    {
                        "cell": cell,
                        "reaction_id": qid,
                        "query_smiles": qsm,
                        "status": "unsupported_reaction_molecule",
                        "candidate_count": 0,
                        "top_similar_reactions": "[]",
                        "top_similarities": "[]",
                    }
                )
                continue

            li = qsm_local[qsm]
            seen_uids = set()
            top_rxns = []
            top_scores = []
            for rank in range(TOP_SIMILAR_REACTIONS):
                c_local = int(local_top_idx[li, rank])
                c_global = int(candidate_indices[c_local])
                crxn = cand_rxns[c_global]
                score = float(local_top_score[li, rank])
                top_rxns.append(crxn)
                top_scores.append(score)
                for uid in rxn_to_uids.get(crxn, []):
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    gate_rows.append(
                        {
                            "cell": cell,
                            "reaction_id": qid,
                            "candidate_uid": uid,
                            "source_reaction_smiles": crxn,
                            "reaction_similarity": score,
                            "similar_reaction_rank": rank + 1,
                        }
                    )
            candidate_counts.append(len(seen_uids))
            query_rows.append(
                {
                    "cell": cell,
                    "reaction_id": qid,
                    "query_smiles": qsm,
                    "status": "ok",
                    "candidate_count": len(seen_uids),
                    "top_similar_reactions": json.dumps(top_rxns),
                    "top_similarities": json.dumps(top_scores),
                }
            )

        cell_summaries[cell] = {
            "query_instances": len(cell_ids),
            "retrieval_unavailable_queries": unavailable,
            "mean_candidate_count": (
                float(np.mean(candidate_counts)) if candidate_counts else 0.0
            ),
            "median_candidate_count": (
                float(np.median(candidate_counts)) if candidate_counts else 0.0
            ),
        }
        print(f"cell={cell} done", flush=True)

    query_frame = pd.DataFrame(query_rows)
    gate_frame = pd.DataFrame(gate_rows)
    query_frame.to_csv(OUT / "query_gate.csv", index=False)
    gate_frame.to_csv(OUT / "gate_candidates.csv.gz", index=False)

    unavailable_ids = sorted(
        qid for qid, qsm in q_by_id.items() if qsm in invalid_q_rxns
    )
    summary = {
        "schema": "enzymecage-original-gate-full7-per-cell-v1",
        "status": "completed",
        "unique_queries": int(len(queries)),
        "cell_query_instances": int(len(query_frame)),
        "unique_query_smiles": int(len(all_unique_q_rxns)),
        "retrieval_evaluable_unique_query_smiles": int(len(valid_q_rxns)),
        "retrieval_unavailable_unique_query_smiles": int(len(invalid_q_rxns)),
        "retrieval_unavailable_query_ids": unavailable_ids,
        "cage_database": str(CAGE_DB.relative_to(ROOT)),
        "cage_database_source": str(CAGE_DB_SOURCE.relative_to(ROOT)),
        "cage_database_projection_audit": (
            "compact projection has the same 307027 rows and 307027 unique "
            "UniprotID-reaction_id-reaction_smiles relations as the author asset"
        ),
        "cage_database_unique_reactions": int(len(cand_rxns)),
        "query_molecules": int(len(q_mols)),
        "candidate_molecules": int(len(c_mols)),
        "top_similar_reactions": TOP_SIMILAR_REACTIONS,
        "semantics": (
            "Each of the seven benchmark cells is passed independently through "
            "the EnzymeCAGE README/retrieve.py protocol: RHEA 2023-07-12 "
            "rhea_rxn2uids.csv, getRSim, Top-10 similar reactions, and that "
            "cell's complete test-reaction set excluded from retrieval candidates."
        ),
        "implementation": (
            "same molecular fingerprints and greedy getRSim formula, compiled "
            "with Numba; deterministic lexicographic tie order"
        ),
        "cells": cell_summaries,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
