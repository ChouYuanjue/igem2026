from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from projects.active.fibre.evidence.experts import ReactionCenterPairEvidence
from projects.active.fibre.evidence.pair_scores import (
    ClipzymePairEvidence,
    enzgfm_pair_evidence,
)
from projects.active.fibre.kernel.evidence_fusion import query_standardize
from projects.active.fibre.model.assets import ROOT
from projects.active.fibre.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.fit_fibre_tps_specialist_gate_v2 import (
    FEATURES as TPS_FEATURES,
)

ROUTER = ROOT / "results/fibre_dynamic_router_v6_reaction_family_guidance/router.pkl"
REACTION_FAMILY = (
    ROOT / "results/enzymecage_reaction_family_response_v1/full_outer/query_features.csv"
)
TPS_ROOT = ROOT / "results/fibre_tps_specialist_response_v2"
TPS_GATE = ROOT / "results/fibre_tps_specialist_gate_v2/gate.pkl"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
OUT = ROOT / "results/fibre_dynamic_router_v6_full_outer"
TOPK = 1000
FAMILIES = ("p450", "phosphatase", "terpene")
RAW_SUFFIXES = (
    "reaction_pre_cosine_shift",
    "reaction_pre_norm_log_ratio",
    "reaction_post_cosine_shift",
    "reaction_post_norm_log_ratio",
    "reaction_attention_entropy_delta",
    "reaction_attention_peak_delta",
)


def z(values: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones(len(x), dtype=bool)
    return query_standardize(
        torch.as_tensor(x, dtype=torch.float64),
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
    return np.asarray(
        [
            float(available.mean()),
            float(expert[available].std()) if available.any() else 0.0,
            margin(expert, available),
            corr(core, expert, available),
            overlap(core, expert, available),
            margin(core, np.ones(len(core), dtype=bool)),
            float(core.std()),
            float(pocket.mean()),
        ],
        dtype=np.float32,
    )


def pocket_support(index: FibreCandidateIndex) -> np.ndarray:
    cache_root = ROOT / "results/fibre_expert_assets_v1/pocket_support"
    cache = cache_root / "broad_mask.npy"
    if not cache.exists():
        cache_root.mkdir(parents=True, exist_ok=True)
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
        supported = set()
        for rec in meta.to_dict("records"):
            pid = str(rec["protein_id"])
            aliases = [pid, str(rec.get("canonical_accession", ""))]
            aliases.extend(str(rec.get("aliases", "")).split(";"))
            if any(a and a in valid for a in aliases):
                supported.add(pid)
        mask = np.asarray([pid in supported for pid in index.protein_ids], dtype=bool)
        np.save(cache, mask)
        (cache_root / "manifest.json").write_text(
            json.dumps({
                "schema": "fibre-broad-pocket-support-mask-v1",
                "proteins": len(index.protein_ids),
                "supported": int(mask.sum()),
                "supported_fraction": float(mask.mean()),
                "cage_valid_structure_uids": len(valid),
                "labels_used": False,
            }, indent=2) + "\n"
        )
        return mask
    mask = np.load(cache, mmap_mode="r")
    if mask.shape != (len(index.protein_ids),):
        raise RuntimeError(
            f"pocket-support cache shape mismatch: {mask.shape} vs {len(index.protein_ids)}"
        )
    return np.asarray(mask, dtype=bool)


def load_cells():
    positives = defaultdict(dict)
    for d in sorted(BENCH.iterdir()):
        p = d / "test_pairs.csv"
        if not p.exists():
            continue
        frame = pd.read_csv(p, dtype=str).fillna("")
        frame = frame[["protein_id", "reaction_id"]].drop_duplicates()
        for q, g in frame.groupby("reaction_id"):
            positives[str(q)][d.name] = set(g["protein_id"].astype(str))
    return positives


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


def reaction_family_vector(row, bundle) -> np.ndarray:
    raw = []
    if row is None:
        raw = [0.0] * 24
    else:
        for fam in FAMILIES:
            for suffix in RAW_SUFFIXES:
                raw.append(float(row[f"{fam}_{suffix}"]))
        raw += [
            float(row["reaction_family_margin"]),
            float(row["reaction_family_entropy"]),
        ]
        winner = str(row["reaction_family_winner"])
        raw += [1.0 if winner == fam else 0.0 for fam in FAMILIES]
        raw += [1.0]
    x = np.asarray(raw, dtype=np.float32)
    continuous = 20
    x[:continuous] = (
        x[:continuous] - np.asarray(bundle["reaction_family_mean"]).reshape(-1)
    ) / np.asarray(bundle["reaction_family_std"]).reshape(-1)
    return x


def metrics(frame: pd.DataFrame, rank_col: str) -> dict:
    r = frame[rank_col]
    return {
        "mrr": float((1.0 / r).mean()),
        "hit10": float((r <= 10).mean()),
        "hit100": float((r <= 100).mean()),
        "hit1000": float((r <= 1000).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with open(ROUTER, "rb") as f:
        bundle = pickle.load(f)
    with open(TPS_GATE, "rb") as f:
        tps_gate = pickle.load(f)

    reaction_family = pd.read_csv(REACTION_FAMILY, dtype=str).fillna("")
    reaction_family = reaction_family.set_index("reaction_id", drop=False)

    tps_q = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str).fillna("")
    tps_q = tps_q.set_index("query_id", drop=False)
    tps_ids = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str)[
        "query_id"
    ].astype(str).tolist()
    tps_z = np.load(TPS_ROOT / "full_outer/reaction_tps.npy", mmap_mode="r")
    tps_index = {q: i for i, q in enumerate(tps_ids)}
    tps_protein = np.load(TPS_ROOT / "broad_protein_tps.npy", mmap_mode="r")

    positives = load_cells()
    queries = sorted(positives)

    index = FibreCandidateIndex(device=args.device)
    functional = enzgfm_pair_evidence("r2e", device=args.device)
    clip = ClipzymePairEvidence(device=args.device)
    mechanism = ReactionCenterPairEvidence(device=args.device)
    pocket_mask = pocket_support(index)
    pid_to_row = index.protein_index

    # One-time Broad-row alignment removes millions of repeated Python dict
    # lookups while preserving the exact expert score definitions.
    functional_p_row = np.asarray(
        [functional.p_index.get(pid, -1) for pid in index.protein_ids],
        dtype=np.int64,
    )
    clip_p_row = np.asarray(
        [clip.p_index.get(pid, -1) for pid in index.protein_ids],
        dtype=np.int64,
    )
    mechanism_p_row = np.asarray(
        [mechanism.p_index.get(pid, -1) for pid in index.protein_ids],
        dtype=np.int64,
    )
    protein_rows = torch.arange(
        len(index.protein_ids), dtype=torch.long, device=index.device
    )

    query_records = []
    result_rows = []

    for start in range(0, len(queries), args.batch_size):
        batch_q = queries[start : start + args.batch_size]
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

            # Functional/evolutionary expert: exact DualTowerPairEvidence score
            # with candidate rows pre-aligned once to the Broad universe.
            frows = functional_p_row[top]
            fr = functional.r_index.get(str(q), -1)
            fa = frows >= 0
            fraw = np.zeros(TOPK, dtype=np.float64)
            if fr < 0:
                fa[:] = False
            elif fa.any():
                idx_t = torch.as_tensor(
                    frows[fa], dtype=torch.long, device=index.device
                )
                fraw[fa] = (
                    functional.p.index_select(0, idx_t) @ functional.r[fr]
                ).float().cpu().numpy()
            fz = z(fraw, fa)

            # Structural CLIPZyme expert with native support semantics.
            crows = clip_p_row[top]
            cr = clip.r_index.get(str(q), -1)
            ca = crows >= 0
            if cr >= 0 and bool(clip.r_supported[cr]):
                ca &= clip.p_supported[np.maximum(crows, 0)]
            else:
                ca[:] = False
            craw = np.zeros(TOPK, dtype=np.float64)
            if ca.any():
                idx_t = torch.as_tensor(
                    crows[ca], dtype=torch.long, device=index.device
                )
                craw[ca] = (
                    clip.p_device.index_select(0, idx_t) @ clip.r_device[cr]
                ).float().cpu().numpy()
            cz = z(craw, ca)

            # Reaction-centre mechanism expert, again using frozen latents.
            mrows = mechanism_p_row[top]
            mr = mechanism.r_index.get(str(q), -1)
            ma = mrows >= 0
            mraw = np.zeros(TOPK, dtype=np.float64)
            if mr < 0:
                ma[:] = False
            elif ma.any():
                idx_t = torch.as_tensor(
                    mrows[ma], dtype=torch.long, device=index.device
                )
                mraw[ma] = (
                    mechanism.p.index_select(0, idx_t) @ mechanism.r[mr]
                ).float().cpu().numpy()
            mz = z(mraw, ma)
            ga = ca | ma
            geom = np.zeros(TOPK, dtype=np.float64)
            count = np.zeros(TOPK, dtype=np.int32)
            for score, avail in ((cz, ca), (mz, ma)):
                geom[avail] += score[avail]
                count[avail] += 1
            geom[ga] /= count[ga]

            base_x = np.concatenate(
                [
                    family_features(core, fz, fa, pocket),
                    family_features(core, geom, ga, pocket),
                ]
            )
            extra = np.asarray(
                [
                    pocket[:20].mean(),
                    pocket[:100].mean(),
                    pocket[:500].mean(),
                    pocket.mean(),
                ],
                dtype=np.float32,
            )
            reaction = (
                index.reaction_embeddings[index.reaction_index[q]]
                .detach()
                .cpu()
                .numpy()[None, :]
            )
            rpca = bundle["pca"].transform(reaction).astype(np.float32)[0]
            rr = None
            if q in reaction_family.index:
                rr = reaction_family.loc[q]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
            rf = reaction_family_vector(rr, bundle)
            X = np.concatenate([base_x, extra, rpca, rf])[None, :].astype(
                np.float32
            )
            pred = predict_weight(bundle, X)
            fp = float(pred["functional"][0][0])
            fw = float(pred["functional"][1][0])
            gp = float(pred["geometry"][0][0])
            gw = float(pred["geometry"][1][0])

            general = core_z + fw * fz + gw * geom
            no_func = core_z + gw * geom
            no_geom = core_z + fw * fz

            tps_active = False
            tps_weight = 0.0
            tps_score = np.zeros(TOPK, dtype=np.float64)
            if q in tps_q.index and q in tps_index:
                tq = tps_q.loc[q]
                if isinstance(tq, pd.DataFrame):
                    tq = tq.iloc[0]
                if float(tq["tps_ref_max_cosine"]) >= float(
                    tps_gate["semantic_threshold"]
                ):
                    qz = np.asarray(tps_z[tps_index[q]], dtype=np.float32)
                    raw_tps = (
                        np.asarray(tps_protein[top], dtype=np.float32) @ qz
                    ).astype(np.float64)
                    sorted_tps = np.sort(raw_tps)
                    values = {
                        "tps_ref_max_cosine": float(tq["tps_ref_max_cosine"]),
                        "tps_ref_top5_mean": float(tq["tps_ref_top5_mean"]),
                        "tps_ref_top20_mean": float(tq["tps_ref_top20_mean"]),
                        "tps_ref_margin_1_2": float(tq["tps_ref_margin_1_2"]),
                        "tps_ref_softmax_entropy": float(
                            tq["tps_ref_softmax_entropy"]
                        ),
                        "tps_score_mean": float(raw_tps.mean()),
                        "tps_score_std": float(raw_tps.std()),
                        "tps_score_top1_margin": float(
                            sorted_tps[-1] - sorted_tps[-2]
                        ),
                        "tps_score_top20_mean": float(sorted_tps[-20:].mean()),
                    }
                    feat = np.asarray(
                        [values[k] for k in TPS_FEATURES], dtype=float
                    )[None, :]
                    raw_w = float(
                        np.clip(
                            tps_gate["regressor"].predict(feat)[0],
                            0.0,
                            0.5,
                        )
                    )
                    tps_weight = raw_w * float(tps_gate["strength_scale"])
                    if tps_weight > 0:
                        tps_active = True
                        tps_score = z(raw_tps)

            final = general + tps_weight * tps_score
            variants = {
                "general": general,
                "final": final,
                "minus_functional": no_func + tps_weight * tps_score,
                "minus_structural_mechanistic": no_geom
                + tps_weight * tps_score,
                "minus_tps": general,
            }
            inverse = {}
            for name, score in variants.items():
                order = np.argsort(-score, kind="stable")
                inv = np.empty(TOPK, dtype=np.int32)
                inv[order] = np.arange(1, TOPK + 1)
                inverse[name] = inv

            query_records.append(
                {
                    "query_id": q,
                    "reaction_family_available": int(rr is not None),
                    "reaction_family_winner": (
                        str(rr["reaction_family_winner"]) if rr is not None else ""
                    ),
                    "functional_prob": fp,
                    "functional_weight": fw,
                    "geometry_prob": gp,
                    "geometry_weight": gw,
                    "tps_active": int(tps_active),
                    "tps_weight": tps_weight,
                    "functional_support": float(fa.mean()),
                    "geometry_support": float(ga.mean()),
                    "pocket_support": float(pocket.mean()),
                }
            )

            top_row_to_local = {
                int(row): i for i, row in enumerate(top)
            }
            for cell, pos_ids in positives[q].items():
                pos_rows = [
                    pid_to_row[p] for p in pos_ids if p in pid_to_row
                ]
                if not pos_rows:
                    continue
                local = [
                    top_row_to_local[r]
                    for r in pos_rows
                    if r in top_row_to_local
                ]
                pos_t = torch.as_tensor(
                    pos_rows, dtype=torch.long, device=index.device
                )
                with torch.no_grad():
                    pos_scores = full_t.index_select(0, pos_t)
                    best_score_t = pos_scores.max()
                    best_rows_t = pos_t[pos_scores == best_score_t]
                    best_row_t = best_rows_t.min()
                    greater = (full_t > best_score_t).sum()
                    tied_before = (
                        (full_t == best_score_t)
                        & (protein_rows < best_row_t)
                    ).sum()
                    broad_rank = int(
                        (greater + tied_before + 1).item()
                    )
                ranks = {}
                for name, inv in inverse.items():
                    ranks[name] = (
                        min(int(inv[i]) for i in local)
                        if local
                        else broad_rank
                    )
                result_rows.append(
                    {
                        "cell": cell,
                        "query_id": q,
                        "broad_best_rank": broad_rank,
                        "general_best_rank": ranks["general"],
                        "final_best_rank": ranks["final"],
                        "minus_functional_best_rank": ranks[
                            "minus_functional"
                        ],
                        "minus_structural_mechanistic_best_rank": ranks[
                            "minus_structural_mechanistic"
                        ],
                        "minus_tps_best_rank": ranks["minus_tps"],
                    }
                )
        print(
            f"outer {min(start+args.batch_size,len(queries))}/{len(queries)}",
            flush=True,
        )

    qframe = pd.DataFrame(query_records)
    rframe = pd.DataFrame(result_rows)
    qframe.to_csv(OUT / "query_router.csv", index=False)
    rframe.to_csv(OUT / "query_metrics.csv", index=False)

    rank_cols = {
        "broad": "broad_best_rank",
        "general_v6": "general_best_rank",
        "final_v6_plus_tps": "final_best_rank",
        "minus_functional": "minus_functional_best_rank",
        "minus_structural_mechanistic": (
            "minus_structural_mechanistic_best_rank"
        ),
        "minus_tps": "minus_tps_best_rank",
    }
    overall = {
        name: metrics(rframe, col) for name, col in rank_cols.items()
    }
    per_cell = {}
    for cell, g in rframe.groupby("cell"):
        per_cell[cell] = {
            name: metrics(g, col) for name, col in rank_cols.items()
        }

    rr_delta = (
        1.0 / rframe.final_best_rank
        - 1.0 / rframe.broad_best_rank
    )
    summary = {
        "schema": "fibre-dynamic-router-v6-full-outer",
        "status": "completed",
        "unique_queries": int(len(qframe)),
        "cell_query_instances": int(len(rframe)),
        "topk_reranked": TOPK,
        "overall": overall,
        "per_cell": per_cell,
        "gate": {
            "reaction_family_available_fraction": float(
                qframe.reaction_family_available.mean()
            ),
            "functional_active_fraction": float(
                (qframe.functional_weight > 0).mean()
            ),
            "geometry_active_fraction": float(
                (qframe.geometry_weight > 0).mean()
            ),
            "tps_active_fraction": float(qframe.tps_active.mean()),
            "tps_active_queries": int(qframe.tps_active.sum()),
        },
        "query_behavior_vs_broad": {
            "improved": int((rr_delta > 0).sum()),
            "tied": int((rr_delta == 0).sum()),
            "worsened": int((rr_delta < 0).sum()),
        },
        "scientific_boundary": {
            "cage_family_checkpoint_direct_ranking_authority": False,
            "cage_family_role": (
                "candidate-independent reaction-branch adaptation covariates "
                "for query-conditioned general expert routing"
            ),
            "tps_role": (
                "direct bounded correction only after frozen TPS reaction-"
                "manifold semantic gate"
            ),
            "outer_test_labels_used_for_training_or_selection": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
