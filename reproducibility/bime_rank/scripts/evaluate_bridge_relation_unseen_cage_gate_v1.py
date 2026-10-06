from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs

from reproducibility.bime_rank.scripts import prepare_enzymecage_original_gate_full7_v1 as cage

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
OUT = ROOT / "results/bridge_relation_unseen_cage_gate_v1"
CAGE_DB = ROOT / "data/external/enzymecage_current/rhea_2023_compact.csv.gz"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
CACHE = ROOT / "results/enzymecage_original_gate_full7_v1"
TOPK = 10


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = pd.read_csv(TARGET, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    reactions = pd.read_csv(REACTIONS, dtype=str).fillna("")
    query_smiles = dict(zip(reactions.reaction_id.astype(str), reactions.reaction_smiles.astype(str)))
    query_ids = sorted(target.reaction_id.astype(str).unique())
    missing = [q for q in query_ids if q not in query_smiles]
    if missing:
        raise RuntimeError(f"missing query reaction metadata: {missing[:10]}")

    db = pd.read_csv(
        CAGE_DB,
        dtype=str,
        usecols=["reaction_id", "UniprotID", "CANO_RXN_SMILES"],
    ).fillna("").drop_duplicates()
    unique_query_rxns = sorted({query_smiles[q] for q in query_ids})
    valid, invalid = cage.split_supported_query_reactions(unique_query_rxns)
    candidate_rxns = sorted(set(db.CANO_RXN_SMILES.astype(str)))
    query_molecules = cage.all_molecules(valid)
    candidate_molecules = cage.all_molecules(candidate_rxns)

    sim = np.empty((len(query_molecules), len(candidate_molecules)), dtype=np.float32)
    old_sim_path = CACHE / "molecule_similarity.npy"
    reusable = False
    if old_sim_path.exists() and (CACHE / "query_molecules.txt").exists() and (CACHE / "candidate_molecules.txt").exists():
        old_q = (CACHE / "query_molecules.txt").read_text().splitlines()
        old_c = (CACHE / "candidate_molecules.txt").read_text().splitlines()
        reusable = old_c == candidate_molecules
        old_sim = np.load(old_sim_path, mmap_mode="r") if reusable else None
        old_index = {x: i for i, x in enumerate(old_q)} if reusable else {}
    else:
        old_sim = None
        old_index = {}

    cand_fp = None
    computed = 0
    for i, smi in enumerate(query_molecules):
        if reusable and smi in old_index:
            sim[i] = old_sim[old_index[smi]]
        else:
            if cand_fp is None:
                cand_fp = [cage.fp(x) for x in candidate_molecules]
            sim[i] = np.asarray(
                DataStructs.BulkTanimotoSimilarity(cage.fp(smi), cand_fp),
                dtype=np.float32,
            )
            computed += 1

    qmi = {x: i for i, x in enumerate(query_molecules)}
    cmi = {x: i for i, x in enumerate(candidate_molecules)}
    ql, qln, qr, qrn = cage.encode_parts(valid, qmi)
    cl, cln, cr, crn = cage.encode_parts(candidate_rxns, cmi)
    top_idx, top_score = cage.topk_reactions(
        ql, qln, qr, qrn, cl, cln, cr, crn, sim, TOPK
    )
    valid_index = {x: i for i, x in enumerate(valid)}
    rxn_to_uids = db.groupby("CANO_RXN_SMILES").UniprotID.agg(lambda x: set(map(str, x))).to_dict()

    meta = pd.read_csv(META, dtype=str).fillna("")
    aliases = {}
    for row in meta[["protein_id", "canonical_accession", "aliases"]].itertuples(index=False):
        vals = {str(row.protein_id), str(row.canonical_accession)}
        vals.update(x for x in re.split(r";+", str(row.aliases)) if x)
        aliases[str(row.protein_id)] = {x for x in vals if x}

    gate = {}
    query_rows = []
    candidate_rows = []
    invalid_set = set(invalid)
    for q in query_ids:
        smi = query_smiles[q]
        if smi in invalid_set:
            gate[q] = set()
            query_rows.append((q, "unsupported_reaction_molecule", 0))
            continue
        i = valid_index[smi]
        uids = set()
        for rank, j in enumerate(top_idx[i], 1):
            source_rxn = candidate_rxns[int(j)]
            for uid in rxn_to_uids.get(source_rxn, set()):
                if uid not in uids:
                    candidate_rows.append(
                        (q, uid, source_rxn, rank, float(top_score[i, rank - 1]))
                    )
                uids.add(uid)
        gate[q] = uids
        query_rows.append((q, "ok", len(uids)))

    target["cage_gate_hit"] = [
        bool(aliases.get(p, {p}) & gate.get(q, set()))
        for p, q in target[["protein_id", "reaction_id"]].itertuples(index=False)
    ]
    target.to_csv(OUT / "edge_gate_recall.csv.gz", index=False)
    qf = pd.DataFrame(query_rows, columns=["reaction_id", "status", "candidate_count"])
    qf.to_csv(OUT / "query_gate.csv", index=False)
    cf = pd.DataFrame(
        candidate_rows,
        columns=[
            "reaction_id",
            "candidate_uid",
            "source_reaction_smiles",
            "similar_reaction_rank",
            "reaction_similarity",
        ],
    )
    cf.to_csv(OUT / "gate_candidates.csv.gz", index=False)

    by_q = target.groupby("reaction_id").cage_gate_hit
    summary = {
        "schema": "bridge-relation-unseen-cage-gate-v1",
        "protocol": "published EnzymeCAGE Top-10 similar-reaction retrieval over frozen 2023 association graph",
        "queries": int(len(query_ids)),
        "edges": int(len(target)),
        "unsupported_queries": int((qf.status != "ok").sum()),
        "candidate_uid_union": int(cf.candidate_uid.nunique()),
        "candidate_rows": int(len(cf)),
        "mean_candidates_per_query": float(qf.candidate_count.mean()),
        "median_candidates_per_query": float(qf.candidate_count.median()),
        "edge_recall": float(target.cage_gate_hit.mean()),
        "query_hit": float(by_q.any().mean()),
        "macro_positive_recall": float(by_q.mean().mean()),
        "new_query_molecules_computed": int(computed),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
