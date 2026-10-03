from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from projects.active.bridge.evidence.experts import ReactionCenterPairEvidence
from projects.active.bridge.evidence.pair_scores import ClipzymePairEvidence, enzgfm_pair_evidence
from projects.active.bridge.kernel.evidence_fusion import query_standardize
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

OUT = ROOT / "results/fibre_dynamic_router_v4"
TOPK = 1000
TRAIN_FOLDS = (0, 1)
VAL_FOLD = 2

FEATURE_NAMES = [
    "expert_available_fraction",
    "expert_score_std",
    "expert_top1_margin",
    "expert_core_corr",
    "top20_overlap",
    "core_top1_margin",
    "core_score_std",
    "pocket_support_fraction",
]


def _z(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    return query_standardize(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(available, dtype=torch.bool),
    ).cpu().numpy()


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return 0.0
    x = a[mask]; y = b[mask]
    if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _top_margin(x: np.ndarray, mask: np.ndarray) -> float:
    v = x[mask]
    if len(v) < 2:
        return 0.0
    p = np.partition(v, -2)[-2:]
    return float(p.max() - p.min())


def _top_overlap(core: np.ndarray, expert: np.ndarray, mask: np.ndarray, k: int = 20) -> float:
    if not mask.any():
        return 0.0
    rows = np.flatnonzero(mask)
    kk = min(k, len(rows))
    a = set(rows[np.argsort(core[rows])[-kk:]].tolist())
    b = set(rows[np.argsort(expert[rows])[-kk:]].tolist())
    return float(len(a & b) / max(len(a | b), 1))


def _features(core: np.ndarray, expert: np.ndarray, available: np.ndarray, pocket: np.ndarray) -> np.ndarray:
    return np.asarray([
        float(available.mean()),
        float(expert[available].std()) if available.any() else 0.0,
        _top_margin(expert, available),
        _corr(core, expert, available),
        _top_overlap(core, expert, available),
        _top_margin(core, np.ones(len(core), dtype=bool)),
        float(core.std()),
        float(pocket.mean()),
    ], dtype=np.float64)


def _best_rank_after_prefix(
    core_order_rows: np.ndarray,
    expert_score: np.ndarray,
    available: np.ndarray,
    positives: set[int],
) -> tuple[int, int]:
    """Return Broad and expert-reranked best positive rank inside TopK.

    Candidates outside TopK preserve their Broad ordering; this utility target only
    rewards an expert when it can actually improve a positive already in the shortlist.
    """
    pos_local = [i for i, row in enumerate(core_order_rows) if int(row) in positives]
    base = min((i + 1 for i in pos_local), default=10**9)
    if not pos_local or not available.any():
        return base, base
    z = _z(expert_score, available)
    # Missing expert scores are neutral and retain Broad tie order.
    key = z.copy()
    key[~available] = -1e9
    order = np.lexsort((np.arange(len(key)), -key))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(1, len(order) + 1)
    routed = min(int(inverse[i]) for i in pos_local)
    return base, routed


def _pocket_support(index: FibreCandidateIndex) -> np.ndarray:
    import pickle as pkl

    gvp_path = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
    node_path = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
    mean_path = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
    gvp = torch.load(gvp_path, map_location="cpu", weights_only=False)
    node = torch.load(node_path, map_location="cpu", weights_only=False)
    mean = pkl.load(open(mean_path, "rb"))
    valid = {
        uid for uid in set(gvp) & set(node)
        if int(gvp[uid][0].shape[0]) == int(node[uid].shape[0])
    }
    metadata = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
        dtype=str,
    ).fillna("")
    supported: set[str] = set()
    for rec in metadata.to_dict("records"):
        pid = str(rec["protein_id"])
        aliases = [pid, str(rec.get("canonical_accession", ""))]
        aliases.extend(str(rec.get("aliases", "")).split(";"))
        if any(alias and alias in valid for alias in aliases):
            supported.add(pid)
    return np.asarray([pid in supported for pid in index.protein_ids], dtype=bool)


def _prepare_fold(fold: int, device: str) -> None:
    out = OUT / "prepared" / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(
        ROOT / f"results/cleanroom_internal_full_candidate_benchmarks_v1/clean2023_internal_double_cold_fold{fold}/test_pairs.csv",
        dtype=str,
    ).fillna("")
    positives = pairs.groupby("reaction_id")["protein_id"].apply(lambda s: set(map(str, s))).to_dict()
    queries = sorted(positives)

    index = FibreCandidateIndex(device=device)
    functional = enzgfm_pair_evidence("r2e", device=device)
    clip = ClipzymePairEvidence(device=device)
    mechanism = ReactionCenterPairEvidence(device=device)
    pocket_mask = _pocket_support(index)
    pid_to_row = index.protein_index

    records = []
    cache_rows = []
    batch = 64
    for start in range(0, len(queries), batch):
        batch_q = queries[start:start + batch]
        qrows = torch.as_tensor([index.reaction_index[q] for q in batch_q], device=index.device)
        with torch.no_grad():
            score_batch = (
                index.reaction_embeddings.index_select(0, qrows) @ index.protein_embeddings.T
            ).float().cpu().numpy()
        for j, q in enumerate(batch_q):
            core_full = score_batch[j]
            order = np.argsort(-core_full, kind="stable")[:TOPK]
            candidates = [index.protein_ids[int(i)] for i in order]
            core = core_full[order].astype(np.float64)
            pocket = pocket_mask[order]

            f = functional.score(direction="r2e", query_id=q, candidate_ids=candidates)
            c = clip.score(direction="r2e", query_id=q, candidate_ids=candidates)
            m = mechanism.score(direction="r2e", query_id=q, candidate_ids=candidates)

            fz = _z(np.asarray(f.score, float), np.asarray(f.available, bool))
            cz = _z(np.asarray(c.score, float), np.asarray(c.available, bool))
            mz = _z(np.asarray(m.score, float), np.asarray(m.available, bool))
            ga = np.asarray(c.available, bool) | np.asarray(m.available, bool)
            geom = np.zeros(TOPK, dtype=np.float64)
            counts = np.zeros(TOPK, dtype=np.int32)
            for z, a in [(cz, np.asarray(c.available, bool)), (mz, np.asarray(m.available, bool))]:
                geom[a] += z[a]; counts[a] += 1
            geom[ga] /= counts[ga]

            pos_rows = {pid_to_row[p] for p in positives[q] if p in pid_to_row}
            base_f, rank_f = _best_rank_after_prefix(order, np.asarray(f.score, float), np.asarray(f.available, bool), pos_rows)
            base_g, rank_g = _best_rank_after_prefix(order, geom, ga, pos_rows)
            base_rank = min(base_f, base_g)
            func_label = int(rank_f < base_rank)
            geom_label = int(rank_g < base_rank)

            func_feat = _features(core, fz, np.asarray(f.available, bool), pocket)
            geom_feat = _features(core, geom, ga, pocket)
            records.append({
                "fold": fold, "query_id": q, "base_best_rank_topk": int(base_rank if base_rank < 10**9 else 0),
                "functional_best_rank_topk": int(rank_f if rank_f < 10**9 else 0),
                "geometry_best_rank_topk": int(rank_g if rank_g < 10**9 else 0),
                "functional_useful": func_label, "geometry_useful": geom_label,
                **{f"functional_{n}": float(v) for n, v in zip(FEATURE_NAMES, func_feat, strict=True)},
                **{f"geometry_{n}": float(v) for n, v in zip(FEATURE_NAMES, geom_feat, strict=True)},
            })
            cache_rows.append({
                "query_id": q,
                "candidate_rows": order.astype(np.int32),
                "core": core.astype(np.float32),
                "functional": fz.astype(np.float32),
                "functional_available": np.asarray(f.available, bool),
                "geometry": geom.astype(np.float32),
                "geometry_available": ga,
                "pocket": pocket,
                "positive_rows": np.asarray(sorted(pos_rows), dtype=np.int32),
            })
        print(f"fold={fold} prepared {min(start+batch,len(queries))}/{len(queries)}", flush=True)

    pd.DataFrame(records).to_csv(out / "query_features.csv", index=False)
    np.save(out / "cache.npy", np.asarray(cache_rows, dtype=object), allow_pickle=True)
    (out / "summary.json").write_text(json.dumps({
        "fold": fold, "queries": len(records), "topk": TOPK,
        "functional_useful_fraction": float(np.mean([r["functional_useful"] for r in records])),
        "geometry_useful_fraction": float(np.mean([r["geometry_useful"] for r in records])),
        "pocket_support_mean": float(np.mean([r["geometry_pocket_support_fraction"] for r in records])),
    }, indent=2) + "\n")




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    _prepare_fold(args.fold, args.device)


if __name__ == "__main__":
    main()
