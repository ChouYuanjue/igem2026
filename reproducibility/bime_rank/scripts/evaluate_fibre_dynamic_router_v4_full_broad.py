from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.evidence.experts import ReactionCenterPairEvidence
from projects.active.bridge.evidence.pair_scores import ClipzymePairEvidence, enzgfm_pair_evidence
from projects.active.bridge.kernel.evidence_fusion import query_standardize
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

ROUTER = ROOT / "results/fibre_dynamic_router_v4/router.pkl"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
OUT = ROOT / "results/fibre_dynamic_router_v4/full_outer"
TOPK = 1000

BASE_FEATURE_NAMES = [
    "expert_available_fraction",
    "expert_score_std",
    "expert_top1_margin",
    "expert_core_corr",
    "top20_overlap",
    "core_top1_margin",
    "core_score_std",
    "pocket_support_fraction",
]


def z(values: np.ndarray, available: np.ndarray) -> np.ndarray:
    return query_standardize(
        torch.as_tensor(values, dtype=torch.float64),
        torch.as_tensor(available, dtype=torch.bool),
    ).cpu().numpy()


def corr(a, b, mask):
    if int(mask.sum()) < 3:
        return 0.0
    x, y = a[mask], b[mask]
    if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def margin(x, mask):
    v = x[mask]
    if len(v) < 2:
        return 0.0
    p = np.partition(v, -2)[-2:]
    return float(p.max() - p.min())


def overlap(core, expert, mask, k=20):
    if not mask.any():
        return 0.0
    rows = np.flatnonzero(mask)
    kk = min(k, len(rows))
    a = set(rows[np.argsort(core[rows])[-kk:]].tolist())
    b = set(rows[np.argsort(expert[rows])[-kk:]].tolist())
    return float(len(a & b) / max(len(a | b), 1))


def family_features(core, expert, available, pocket):
    return np.asarray([
        float(available.mean()),
        float(expert[available].std()) if available.any() else 0.0,
        margin(expert, available),
        corr(core, expert, available),
        overlap(core, expert, available),
        margin(core, np.ones(len(core), dtype=bool)),
        float(core.std()),
        float(pocket.mean()),
    ], dtype=np.float32)


def pocket_support(index: FibreCandidateIndex) -> np.ndarray:
    import pickle as pkl
    gvp = torch.load(
        ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt",
        map_location="cpu", weights_only=False,
    )
    node = torch.load(
        ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt",
        map_location="cpu", weights_only=False,
    )
    valid = {
        uid for uid in set(gvp) & set(node)
        if int(gvp[uid][0].shape[0]) == int(node[uid].shape[0])
    }
    meta = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
        dtype=str,
    ).fillna("")
    support = set()
    for rec in meta.to_dict("records"):
        pid = str(rec["protein_id"])
        aliases = [pid, str(rec.get("canonical_accession", ""))]
        aliases.extend(str(rec.get("aliases", "")).split(";"))
        if any(a and a in valid for a in aliases):
            support.add(pid)
    return np.asarray([pid in support for pid in index.protein_ids], dtype=bool)


def load_cells():
    positives = defaultdict(dict)
    cells = {}
    for d in sorted(BENCH.iterdir()):
        p = d / "test_pairs.csv"
        if not p.exists():
            continue
        frame = pd.read_csv(p, dtype=str).fillna("")
        frame = frame[["protein_id", "reaction_id"]].drop_duplicates()
        cells[d.name] = frame
        for q, g in frame.groupby("reaction_id"):
            positives[str(q)][d.name] = set(g["protein_id"].astype(str))
    return cells, positives


def predict_weight(bundle, X):
    pred = {}
    for fam in ("functional", "geometry"):
        cls, reg = bundle["models"][fam]
        p = cls.predict_proba(X)[:, 1]
        s = np.clip(reg.predict(X), 0.0, 0.75)
        w = np.where(
            p >= bundle["thresholds"][fam],
            s * bundle["scales"][fam],
            0.0,
        )
        pred[fam] = (p, np.clip(w, 0.0, 0.75))
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with open(ROUTER, "rb") as f:
        bundle = pickle.load(f)
    cells, positives = load_cells()
    queries = sorted(positives)

    index = FibreCandidateIndex(device=args.device)
    functional = enzgfm_pair_evidence("r2e", device=args.device)
    clip = ClipzymePairEvidence(device=args.device)
    mechanism = ReactionCenterPairEvidence(device=args.device)
    pocket_mask = pocket_support(index)
    pid_to_row = index.protein_index
    protein_rows = torch.arange(len(index.protein_ids), dtype=torch.long, device=index.device)

    query_records = []
    result_rows = []

    for start in range(0, len(queries), args.batch_size):
        batch_q = queries[start:start + args.batch_size]
        qrows = torch.as_tensor(
            [index.reaction_index[q] for q in batch_q],
            device=index.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            full_batch = (
                index.reaction_embeddings.index_select(0, qrows)
                @ index.protein_embeddings.T
            ).float()
            top_values_t, top_indices_t = torch.topk(
                full_batch, k=TOPK, dim=1, largest=True, sorted=True
            )
        top_values = top_values_t.cpu().numpy()
        top_indices = top_indices_t.cpu().numpy()

        for j, q in enumerate(batch_q):
            full_t = full_batch[j]
            top = top_indices[j].astype(np.int64, copy=False)
            candidates = [index.protein_ids[int(i)] for i in top]
            core = top_values[j].astype(np.float64, copy=False)
            core_z = (core - core.mean()) / max(core.std(), 1e-6)
            pocket = pocket_mask[top]

            fo = functional.score(direction="r2e", query_id=q, candidate_ids=candidates)
            co = clip.score(direction="r2e", query_id=q, candidate_ids=candidates)
            mo = mechanism.score(direction="r2e", query_id=q, candidate_ids=candidates)

            fa = np.asarray(fo.available, bool)
            fz = z(np.asarray(fo.score, float), fa)

            ca = np.asarray(co.available, bool)
            ma = np.asarray(mo.available, bool)
            cz = z(np.asarray(co.score, float), ca)
            mz = z(np.asarray(mo.score, float), ma)
            ga = ca | ma
            geom = np.zeros(TOPK, dtype=np.float64)
            count = np.zeros(TOPK, dtype=np.int32)
            for score, avail in ((cz, ca), (mz, ma)):
                geom[avail] += score[avail]
                count[avail] += 1
            geom[ga] /= count[ga]

            base_x = np.concatenate([
                family_features(core, fz, fa, pocket),
                family_features(core, geom, ga, pocket),
            ])
            extra = np.asarray([
                pocket[:20].mean(), pocket[:100].mean(),
                pocket[:500].mean(), pocket.mean(),
            ], dtype=np.float32)
            reaction = index.reaction_embeddings[index.reaction_index[q]].detach().cpu().numpy()[None, :]
            rpca = bundle["pca"].transform(reaction).astype(np.float32)[0]
            X = np.concatenate([base_x, extra, rpca])[None, :].astype(np.float32)
            pred = predict_weight(bundle, X)
            fp, fw = float(pred["functional"][0][0]), float(pred["functional"][1][0])
            gp, gw = float(pred["geometry"][0][0]), float(pred["geometry"][1][0])

            variant_scores = {
                "full": core_z + fw * fz + gw * geom,
                "minus_functional": core_z + gw * geom,
                "minus_structural_mechanistic": core_z + fw * fz,
            }
            variant_inverse = {}
            for variant, score in variant_scores.items():
                order = np.argsort(-score, kind="stable")
                inv = np.empty(TOPK, dtype=np.int32)
                inv[order] = np.arange(1, TOPK + 1)
                variant_inverse[variant] = inv

            query_records.append({
                "query_id": q,
                "functional_prob": fp, "functional_weight": fw,
                "geometry_prob": gp, "geometry_weight": gw,
                "functional_support": float(fa.mean()),
                "geometry_support": float(ga.mean()),
                "pocket_support": float(pocket.mean()),
            })

            top_row_to_local = {int(row): i for i, row in enumerate(top)}
            for cell, pos_ids in positives[q].items():
                pos_rows = [pid_to_row[p] for p in pos_ids if p in pid_to_row]
                if not pos_rows:
                    continue
                local = [top_row_to_local[r] for r in pos_rows if r in top_row_to_local]
                # Exact Broad best-positive rank from the full GPU score vector.
                pos_t = torch.as_tensor(pos_rows, dtype=torch.long, device=index.device)
                with torch.no_grad():
                    pos_scores = full_t.index_select(0, pos_t)
                    best_score_t = pos_scores.max()
                    best_rows_t = pos_t[pos_scores == best_score_t]
                    best_row_t = best_rows_t.min()
                    greater = (full_t > best_score_t).sum()
                    tied_before = ((full_t == best_score_t) & (protein_rows < best_row_t)).sum()
                    broad_rank = int((greater + tied_before + 1).item())
                variant_ranks = {}
                for variant, inv in variant_inverse.items():
                    variant_ranks[variant] = (
                        min(int(inv[i]) for i in local) if local else broad_rank
                    )
                result_rows.append({
                    "cell": cell, "query_id": q,
                    "positive_count": len(pos_rows),
                    "broad_best_rank": broad_rank,
                    "fused_best_rank": variant_ranks["full"],
                    "minus_functional_best_rank": variant_ranks["minus_functional"],
                    "minus_structural_mechanistic_best_rank": variant_ranks["minus_structural_mechanistic"],
                    "functional_weight": fw,
                    "geometry_weight": gw,
                })
        print(f"outer {min(start+args.batch_size,len(queries))}/{len(queries)}", flush=True)

    qframe = pd.DataFrame(query_records)
    rframe = pd.DataFrame(result_rows)
    qframe.to_csv(OUT / "query_router.csv", index=False)
    rframe.to_csv(OUT / "query_metrics.csv", index=False)

    def metrics(g, rank_col):
        rank = g[rank_col]
        return {
            "mrr": float((1.0 / rank).mean()),
            "hit10": float((rank <= 10).mean()),
            "hit100": float((rank <= 100).mean()),
            "hit1000": float((rank <= 1000).mean()),
        }

    per_cell = {}
    for cell, g in rframe.groupby("cell"):
        broad = metrics(g, "broad_best_rank")
        full = metrics(g, "fused_best_rank")
        no_func = metrics(g, "minus_functional_best_rank")
        no_geom = metrics(g, "minus_structural_mechanistic_best_rank")
        per_cell[cell] = {
            "queries": int(len(g)),
            "broad": broad,
            "full": full,
            "minus_functional": no_func,
            "minus_structural_mechanistic": no_geom,
            "full_delta_vs_broad": {k: float(full[k] - broad[k]) for k in broad},
            "functional_family_contribution": {k: float(full[k] - no_func[k]) for k in full},
            "structural_mechanistic_family_contribution": {k: float(full[k] - no_geom[k]) for k in full},
        }

    summary = {
        "schema": "fibre-dynamic-router-v4-full-broad-outer",
        "status": "completed",
        "unique_queries": int(len(qframe)),
        "cell_query_instances": int(len(rframe)),
        "topk_reranked": TOPK,
        "per_cell": per_cell,
        "gate": {
            "functional_active_fraction": float((qframe.functional_weight > 0).mean()),
            "functional_mean_active": float(qframe.loc[qframe.functional_weight > 0, "functional_weight"].mean()),
            "geometry_active_fraction": float((qframe.geometry_weight > 0).mean()),
            "geometry_mean_active": float(qframe.loc[qframe.geometry_weight > 0, "geometry_weight"].mean()),
            "both_active_fraction": float(((qframe.functional_weight > 0) & (qframe.geometry_weight > 0)).mean()),
            "neither_active_fraction": float(((qframe.functional_weight == 0) & (qframe.geometry_weight == 0)).mean()),
            "mean_pocket_support_top1000": float(qframe.pocket_support.mean()),
        },
        "test_labels_used_for_router_training_or_selection": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
