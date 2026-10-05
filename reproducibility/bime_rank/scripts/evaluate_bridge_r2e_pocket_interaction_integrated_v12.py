from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.evidence.experts import ReactionCenterPairEvidence
from projects.active.bridge.evidence.pair_scores import (
    ClipzymePairEvidence,
    enzgfm_pair_evidence,
)
from projects.active.bridge.kernel.evidence_fusion import query_standardize
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.fit_fibre_tps_specialist_gate_v2 import (
    FEATURES as TPS_FEATURES,
)
from reproducibility.bime_rank.scripts import train_bridge_pocket_interaction_expert_v3 as pocket_v3
from reproducibility.bime_rank.scripts import train_bridge_pocket_loo_expert_v6 as pocket_v6

ROUTER = ROOT / "results/fibre_dynamic_router_v4/router.pkl"
TPS_ROOT = ROOT / "results/fibre_tps_specialist_response_v2"
TPS_GATE = ROOT / "results/fibre_tps_specialist_gate_v2/gate.pkl"
BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
OUT = ROOT / "results/bridge_r2e_pocket_interaction_integrated_v12"
POCKET_MODEL = ROOT / "results/bridge_pocket_loo_expert_v10/model.npz"
POCKET_META = ROOT / "results/bridge_pocket_interaction_repr_v1/outer_max/pairs.csv.gz"
POCKET_REPR = ROOT / "results/bridge_pocket_interaction_repr_v1/outer_max/interaction_fused_f32.npy"
POCKET_ALPHA = 0.35
POCKET_PREFIX = 20
TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
POCKET_DEV = ROOT / "results/bridge_pocket_query_reliability_v4/summary.json"
FAMILY_GATE = ROOT / "results/fibre_cage_family_specialist_gate_v1/gates.pkl"
FAMILY_APP = ROOT / "results/fibre_family_applicability_router_v1/full_outer_predictions.csv"
FAMILY_REACTION = ROOT / "results/enzymecage_reaction_family_response_v1/full_outer/query_features.csv"
FAMILY_PAIR = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/pair_features.csv.gz"
FAMILY_QUERY = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/query_features_calibrated.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
TOPK = 1000
TRAIN_RELATIONS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
FAMILIES = ("p450", "phosphatase", "terpene")
FAMILY_SCALE = {"p450": 0.5, "phosphatase": 0.25, "terpene": 0.75}
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


def load_targets():
    frame = pd.read_csv(TARGETS, dtype=str).fillna("")
    positives = frame.groupby("reaction_id")["protein_id"].apply(lambda x: set(x.astype(str))).to_dict()
    return positives


def weighted_geometry(cz, ca, mz, ma, pocket, supported_weight, unsupported_weight):
    cw = np.where(pocket, float(supported_weight), float(unsupported_weight))
    num = np.zeros(len(cz), dtype=np.float64)
    den = np.zeros(len(cz), dtype=np.float64)
    use_clip = ca & (cw > 0)
    num[use_clip] += cw[use_clip] * cz[use_clip]
    den[use_clip] += cw[use_clip]
    num[ma] += mz[ma]
    den[ma] += 1.0
    avail = den > 0
    out = np.zeros(len(cz), dtype=np.float64)
    out[avail] = num[avail] / den[avail]
    return out, avail


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


def member_routed_weights(bundle, core, functional_score, functional_available, geometry_score, geometry_available, pocket, rpca):
    base_x = np.concatenate([
        family_features(core, functional_score, functional_available, pocket),
        family_features(core, geometry_score, geometry_available, pocket),
    ])
    extra = np.asarray([
        pocket[:20].mean(),
        pocket[:100].mean(),
        pocket[:500].mean(),
        pocket.mean(),
    ], dtype=np.float32)
    X = np.concatenate([base_x, extra, rpca])[None, :].astype(np.float32)
    pred = predict_weight(bundle, X)
    return float(pred["functional"][1][0]), float(pred["geometry"][1][0])


def family_feature_columns(fam: str, suffixes) -> list[str]:
    return [f"{fam}_{suffix}" for suffix in suffixes] + [
        "family_calibrated_max_percentile",
        "family_calibrated_margin",
        "family_calibrated_entropy",
        "candidate_count",
    ]


def metrics(frame: pd.DataFrame, rank_col: str) -> dict:
    r = frame[rank_col]
    return {
        "mrr": float((1.0 / r).mean()),
        "hit10": float((r <= 10).mean()),
        "hit100": float((r <= 100).mean()),
        "hit1000": float((r <= 1000).mean()),
    }


def bounded_pocket_inverse(base_score, candidates, pocket_group):
    order = np.argsort(-base_score, kind="stable")
    inv = np.empty(len(order), dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1)
    if pocket_group is None or len(pocket_group) < 2:
        return inv, 0
    local_by_pid = {str(pid): i for i, pid in enumerate(candidates)}
    locs = []
    vals = []
    for pid, score in pocket_group[["protein_id", "pocket_interaction_score"]].itertuples(index=False):
        loc = local_by_pid.get(str(pid))
        if loc is None or int(inv[loc]) <= POCKET_PREFIX:
            continue
        locs.append(loc)
        vals.append(float(score))
    if len(locs) < 2:
        return inv, len(locs)
    locs = np.asarray(locs, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    mix = z(base_score[locs]) + POCKET_ALPHA * z(vals)
    slots = np.sort(inv[locs])
    ranked_locs = locs[np.argsort(-mix, kind="stable")]
    out = inv.copy()
    out[ranked_locs] = slots
    return out, len(locs)


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
    with open(FAMILY_GATE, "rb") as f:
        family_gate = pickle.load(f)

    family_app = pd.read_csv(FAMILY_APP, dtype={"reaction_id": str}).fillna("")
    family_app["confidence"] = pd.to_numeric(family_app["confidence"], errors="coerce").fillna(0.0)
    family_app = family_app.set_index("reaction_id", drop=False)
    family_reaction = pd.read_csv(FAMILY_REACTION, dtype={"reaction_id": str}).fillna("")
    family_reaction = family_reaction.set_index("reaction_id", drop=False)
    family_pair = pd.read_csv(FAMILY_PAIR, dtype=str).fillna("")
    family_pair_groups = {
        str(key): group for key, group in family_pair.groupby("CANO_RXN_SMILES", sort=False)
    }
    family_query = pd.read_csv(FAMILY_QUERY, dtype=str).fillna("")
    family_query = family_query.set_index("CANO_RXN_SMILES", drop=False)
    rxmeta = pd.read_csv(REACTIONS, dtype=str).fillna("")
    reaction_smiles = dict(zip(rxmeta["reaction_id"].astype(str), rxmeta["reaction_smiles"].astype(str)))

    tps_q = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str).fillna("")
    tps_q = tps_q.set_index("query_id", drop=False)
    tps_ids = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str)[
        "query_id"
    ].astype(str).tolist()
    tps_z = np.load(TPS_ROOT / "full_outer/reaction_tps.npy", mmap_mode="r")
    tps_index = {q: i for i, q in enumerate(tps_ids)}
    tps_protein = np.load(TPS_ROOT / "broad_protein_tps.npy", mmap_mode="r")

    positives = load_targets()
    pocket_npz = np.load(POCKET_MODEL)
    pocket_model = {"mean": pocket_npz["mean"], "std": pocket_npz["std"], "weight": pocket_npz["weight"]}
    pocket_pair = pd.read_csv(POCKET_META, dtype={"reaction_id": str, "protein_id": str}).fillna("")
    pocket_repr = np.load(POCKET_REPR, mmap_mode="r").astype(np.float32)
    pocket_score = pocket_v6.apply(pocket_model, pocket_v3.transform(pocket_repr, "directed_product"))
    pocket_pair = pocket_pair[["reaction_id", "protein_id"]].copy()
    pocket_pair["pocket_interaction_score"] = pocket_score
    pocket_groups = {str(q): g for q, g in pocket_pair.groupby("reaction_id", sort=False)}
    queries = sorted(positives)
    train_rel = pd.read_csv(TRAIN_RELATIONS, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    known_by_reaction = train_rel.groupby("reaction_id")["protein_id"].apply(lambda x: set(x.astype(str))).to_dict()

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

        for j, q in enumerate(batch_q):
            full_t = full_batch[j]
            known_rows = np.asarray(
                [pid_to_row[p] for p in known_by_reaction.get(q, set()) if p in pid_to_row],
                dtype=np.int64,
            )
            masked_full_t = full_t.clone()
            if len(known_rows):
                masked_full_t[torch.as_tensor(known_rows, dtype=torch.long, device=index.device)] = -torch.inf
            with torch.no_grad():
                top_values_t, top_indices_t = torch.topk(masked_full_t, k=TOPK, largest=True, sorted=True)
            top = top_indices_t.cpu().numpy().astype(np.int64, copy=False)
            candidates = [index.protein_ids[int(i)] for i in top]
            core = top_values_t.cpu().numpy().astype(np.float64, copy=False)
            core_z = (core - core.mean()) / max(core.std(), 1e-6)
            pocket = pocket_mask[top]
            pocket_feature = np.zeros(TOPK, dtype=np.float32)

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

            reaction = (
                index.reaction_embeddings[index.reaction_index[q]]
                .detach().cpu().numpy()[None, :]
            )
            rpca = bundle["pca"].transform(reaction).astype(np.float32)[0]
            zero_extra = np.zeros(4, dtype=np.float32)
            base_x = np.concatenate([
                family_features(core, fz, fa, pocket_feature),
                family_features(core, geom, ga, pocket_feature),
            ])
            X = np.concatenate([base_x, zero_extra, rpca])[None, :].astype(np.float32)
            pred = predict_weight(bundle, X)
            fp = float(pred["functional"][0][0]); fw = float(pred["functional"][1][0])
            gp = float(pred["geometry"][0][0]); gw0 = float(pred["geometry"][1][0])
            density = float(pocket[:20].mean())
            pocket_multiplier = 1.0
            gw = gw0
            general_no_pocket = core_z + fw * fz + gw0 * geom
            general = general_no_pocket
            no_func = core_z + gw * geom
            no_geom = core_z + fw * fz


            family_active = "none"
            family_weight = 0.0
            family_score = np.zeros(TOPK, dtype=np.float64)
            if q in family_app.index and q in family_reaction.index:
                ar = family_app.loc[q]
                if isinstance(ar, pd.DataFrame):
                    ar = ar.iloc[0]
                rr = family_reaction.loc[q]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
                fam = str(ar["pred"])
                confidence = float(ar["confidence"])
                rxn = reaction_smiles.get(q, "")
                if (
                    fam in FAMILIES
                    and confidence >= 0.50
                    and str(rr["reaction_family_winner"]) == fam
                    and rxn in family_query.index
                    and rxn in family_pair_groups
                ):
                    fq = family_query.loc[rxn]
                    if isinstance(fq, pd.DataFrame):
                        fq = fq.iloc[0]
                    if str(fq["family_calibrated_winner"]) == fam:
                        pg = family_pair_groups[rxn]
                        local_by_pid = {str(pid): i for i, pid in enumerate(candidates)}
                        residual = np.zeros(TOPK, dtype=np.float64)
                        avail = np.zeros(TOPK, dtype=bool)
                        col = f"{fam}_normalized_logit_response"
                        for pid, rawv in pg[["protein_id", col]].itertuples(index=False):
                            loc = local_by_pid.get(str(pid))
                            if loc is None:
                                continue
                            residual[loc] = float(rawv)
                            avail[loc] = True
                        residual = z(residual, avail)
                        cols = family_feature_columns(fam, family_gate["feature_suffixes"])
                        feat = [float(fq[c]) for c in cols] + [float(avail.mean())]
                        raw_w = float(
                            np.clip(
                                family_gate["models"][fam].predict(
                                    np.asarray(feat, dtype=float)[None, :]
                                )[0],
                                0.0,
                                0.50,
                            )
                        )
                        family_weight = raw_w * FAMILY_SCALE[fam]
                        if family_weight > 0:
                            family_active = fam
                            family_score = residual

            tps_active = False
            tps_weight = 0.0
            tps_score = np.zeros(TOPK, dtype=np.float64)
            if q in tps_q.index and q in tps_index:
                tq = tps_q.loc[q]
                if isinstance(tq, pd.DataFrame):
                    tq = tq.iloc[0]
                if (
                    float(tq["tps_ref_max_cosine"]) >= float(tps_gate["semantic_threshold"])
                    and family_active not in ("p450", "phosphatase")
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

            specialist_delta = (
                family_weight * family_score + tps_weight * tps_score
            )
            final_without_pocket = general_no_pocket + specialist_delta
            minus_func_without_pocket = no_func + specialist_delta
            minus_structural = no_geom + specialist_delta
            pocket_group = pocket_groups.get(str(q))
            final_pocket_inv, pocket_scoreable = bounded_pocket_inverse(
                final_without_pocket, candidates, pocket_group
            )
            minus_func_pocket_inv, _ = bounded_pocket_inverse(
                minus_func_without_pocket, candidates, pocket_group
            )
            variants = {
                "full_without_pocket_interaction": final_without_pocket,
                "minus_structural_mechanistic": minus_structural,
            }
            inverse = {}
            for name, score in variants.items():
                order = np.argsort(-score, kind="stable")
                inv = np.empty(TOPK, dtype=np.int32)
                inv[order] = np.arange(1, TOPK + 1)
                inverse[name] = inv
            inverse["full_with_pocket_interaction"] = final_pocket_inv
            inverse["minus_functional_homology"] = minus_func_pocket_inv

            query_records.append(
                {
                    "query_id": q,
                    "functional_prob": fp,
                    "functional_weight": fw,
                    "geometry_prob": gp,
                    "geometry_weight": gw,
                    "family_active": family_active,
                    "family_weight": family_weight,
                    "tps_active": int(tps_active),
                    "tps_weight": tps_weight,
                    "functional_support": float(fa.mean()),
                    "geometry_support": float(ga.mean()),
                    "pocket_support": float(pocket.mean()),
                    "pocket_density": density,
                    "pocket_multiplier": pocket_multiplier,
                    "pocket_interaction_scoreable": int(pocket_scoreable),
                    "known_relations_masked": int(len(known_rows)),
                }
            )

            top_row_to_local = {int(row): i for i, row in enumerate(top)}
            pos_ids = positives[q]
            pos_rows = [pid_to_row[p] for p in pos_ids if p in pid_to_row]
            if pos_rows:
                local = [top_row_to_local[r] for r in pos_rows if r in top_row_to_local]
                pos_t = torch.as_tensor(pos_rows, dtype=torch.long, device=index.device)
                with torch.no_grad():
                    pos_scores = masked_full_t.index_select(0, pos_t)
                    best_score_t = pos_scores.max(); best_rows_t = pos_t[pos_scores == best_score_t]; best_row_t = best_rows_t.min()
                    greater = (masked_full_t > best_score_t).sum()
                    tied_before = ((masked_full_t == best_score_t) & (protein_rows < best_row_t)).sum()
                    broad_rank = int((greater + tied_before + 1).item())
                ranks = {name:(min(int(inv[i]) for i in local) if local else broad_rank) for name,inv in inverse.items()}
                result_rows.append({
                    "query_id": q, "target_relations": len(pos_rows), "broad_best_rank": broad_rank,
                    "full_with_pocket_best_rank": ranks["full_with_pocket_interaction"],
                    "full_without_pocket_best_rank": ranks["full_without_pocket_interaction"],
                    "minus_functional_homology_best_rank": ranks["minus_functional_homology"],
                    "minus_structural_mechanistic_best_rank": ranks["minus_structural_mechanistic"],
                })
        print(
            f"outer {min(start+args.batch_size,len(queries))}/{len(queries)}",
            flush=True,
        )

    qframe = pd.DataFrame(query_records)
    rframe = pd.DataFrame(result_rows)
    qframe.to_csv(OUT / "query_router.csv", index=False)
    rframe.to_csv(OUT / "query_metrics.csv", index=False)

    rank_cols = {
        "full_integrated_system": "full_with_pocket_best_rank",
        "full_without_pocket_interaction": "full_without_pocket_best_rank",
        "minus_functional_homology": "minus_functional_homology_best_rank",
        "minus_structural_mechanistic": "minus_structural_mechanistic_best_rank",
    }
    overall = {
        name: metrics(rframe, col) for name, col in rank_cols.items()
    }
    rr_delta = (
        1.0 / rframe.full_with_pocket_best_rank
        - 1.0 / rframe.broad_best_rank
    )
    summary = {
        "schema": "bridge-r2e-pocket-interaction-integrated-v12",
        "status": "completed",
        "unique_queries": int(len(qframe)),
        "relation_unseen_queries_evaluated": int(len(rframe)),
        "topk_reranked": TOPK,
        "relation_target_policy": "all relations in maximal fair-pool union absent from clean2023; no enzyme-seen or reaction-seen restriction",
        "pocket_interaction": {"expert": "Pocket-Reaction Interaction Expert v10", "alpha": POCKET_ALPHA, "protected_prefix": POCKET_PREFIX, "model": str(POCKET_MODEL.relative_to(ROOT)), "representation": str(POCKET_REPR.relative_to(ROOT))},
        "overall": overall,
        "gate": {
            "functional_active_fraction": float(
                (qframe.functional_weight > 0).mean()
            ),
            "geometry_active_fraction": float(
                (qframe.geometry_weight > 0).mean()
            ),
            "cage_family_active_queries": int(qframe.family_active.ne("none").sum()),
            "cage_family_active_by_family": {
                fam: int(qframe.family_active.eq(fam).sum()) for fam in FAMILIES
            },
            "tps_active_fraction": float(qframe.tps_active.mean()),
            "tps_active_queries": int(qframe.tps_active.sum()),
            "any_domain_specialist_queries": int(
                (qframe.family_active.ne("none") | qframe.tps_active.astype(bool)).sum()
            ),
        },
        "query_behavior_vs_broad": {
            "improved": int((rr_delta > 0).sum()),
            "tied": int((rr_delta == 0).sum()),
            "worsened": int((rr_delta < 0).sum()),
        },
        "scientific_boundary": {
            "family_finetuned_cage_in_general_router": False,
            "family_specialists": (
                "P450/phosphatase/terpene experts are part of the integrated system; "
                "each receives direct residual ranking authority only when the frozen "
                "reaction applicability selector, reaction-family winner and pair-level "
                "family winner agree"
            ),
            "tps_role": (
                "TPS is the fourth domain specialist and receives bounded direct "
                "correction only after its frozen reaction-manifold gate"
            ),
            "category_ablation_scope": {
                "functional_homology": "zero-shot EnzGFM contribution; seed/homology is evaluated separately as relation context",
                "structural_mechanistic": "CLIPZyme plus reaction-center plus Pocket-Reaction Interaction Expert",
                "domain_specialists": "kept intact and not ablated",
            },
            "pocket_support": "independent pair-level Pocket-Reaction Interaction Expert; local correction only among scoreable candidates below protected Top20",
            "overall_test_labels_used_for_training_or_threshold_fitting": False,
        },
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
