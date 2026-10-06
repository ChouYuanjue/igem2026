from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUT = ROOT / "results/bridge_current_edgewise_e2r_v1"


def calibrated(raw: np.ndarray, available: np.ndarray, center: float, scale: float) -> np.ndarray:
    out = np.zeros(len(raw), dtype=np.float64)
    if available.any():
        out[available] = (raw[available] - float(center)) / max(float(scale), 1e-8)
    return out


def filtered_ranks(raw: list[int]) -> list[int]:
    arr = np.asarray(raw, dtype=np.int64)
    return [int(v - np.count_nonzero(arr < v)) for v in arr]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    positives = targets.groupby("protein_id")["reaction_id"].apply(lambda x: list(map(str, x))).to_dict()
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    known = train.groupby("protein_id")["reaction_id"].apply(lambda x: set(map(str, x))).to_dict()
    seen_p = set(train["protein_id"].astype(str))
    seen_r = set(train["reaction_id"].astype(str))

    rt = FinalBridgeRuntime(device=args.device)
    rids = rt.index.reaction_ids
    ridx = rt.index.reaction_index
    lex = rt._reaction_lex
    functional_member = rt.e2r_members["enzgfm_e2r"]
    clip_member = rt.e2r_members["clipzyme_structure"]
    frows = rt._e2r_functional_r_row
    crows = rt._clip_r_row

    rows = []
    qa = []
    queries = sorted(positives)

    for start in range(0, len(queries), args.batch_size):
        batch = queries[start:start + args.batch_size]
        valid = [q for q in batch if q in rt.index.protein_index]
        if not valid:
            continue
        prows = torch.as_tensor(
            [rt.index.protein_index[q] for q in valid],
            dtype=torch.long,
            device=rt.index.device,
        )
        with torch.no_grad():
            broad_batch = (
                rt.index.protein_embeddings.index_select(0, prows)
                @ rt.index.reaction_embeddings.T
            ).float().cpu().numpy().astype(np.float64, copy=False)

        for j, q in enumerate(valid):
            target_ids = [r for r in positives[q] if r in ridx]
            if not target_ids:
                continue
            broad = np.asarray(broad_batch[j], dtype=np.float64)
            bstd = max(float(broad.std()), 1e-8)
            bz = (broad - float(broad.mean())) / bstd
            ctx, ca0, cb0, av, both = rt._relation_components("e2r", q)
            rw, rp, rs = rt._relation_authority("e2r", q, bz, ctx, ca0, cb0, av, both)

            core_scale = float(rt.e2r_core_cal["scale"])
            core = (broad - float(rt.e2r_core_cal["center"])) / core_scale
            relation_core = rw * (bstd / core_scale) * ctx

            fp = rt.functional_e2r.p_index.get(q, -1)
            fa = frows >= 0
            fraw = np.zeros(len(rids), dtype=np.float64)
            if fp < 0:
                fa = np.zeros_like(fa)
            elif fa.any():
                rtrows = torch.as_tensor(frows[fa], dtype=torch.long, device=rt.index.device)
                with torch.no_grad():
                    fraw[fa] = (
                        rt.functional_e2r.r.index_select(0, rtrows)
                        @ rt.functional_e2r.p[fp]
                    ).float().cpu().numpy()
            fcal = calibrated(
                fraw,
                fa,
                functional_member["calibration"]["score_center"],
                functional_member["calibration"]["score_scale"],
            )

            cp = rt.clip.p_index.get(q, -1)
            cav = crows >= 0
            craw = np.zeros(len(rids), dtype=np.float64)
            if cp < 0 or not bool(rt.clip.p_supported[cp]):
                cav = np.zeros_like(cav)
            else:
                cav &= rt.clip.r_supported[np.maximum(crows, 0)]
            if cav.any():
                rtrows = torch.as_tensor(crows[cav], dtype=torch.long, device=rt.index.device)
                with torch.no_grad():
                    craw[cav] = (
                        rt.clip.r_device.index_select(0, rtrows)
                        @ rt.clip.p_device[cp]
                    ).float().cpu().numpy()
            ccal = calibrated(
                craw,
                cav,
                clip_member["calibration"]["score_center"],
                clip_member["calibration"]["score_scale"],
            )

            full = (
                core
                + relation_core
                + float(functional_member["strength"]) * fcal
                + float(clip_member["strength"]) * ccal
            )
            broad_eval = broad.copy()
            full_eval = full.copy()
            masked = [ridx[r] for r in known.get(q, set()) if r in ridx]
            if masked:
                m = np.asarray(masked, dtype=np.int64)
                broad_eval[m] = -np.inf
                full_eval[m] = -np.inf

            def rank_score(score: np.ndarray, rid: str) -> int:
                row = ridx[rid]
                val = float(score[row])
                return int(
                    np.count_nonzero(score > val)
                    + np.count_nonzero((score == val) & (lex < lex[row]))
                    + 1
                )

            broad_raw = [rank_score(broad_eval, r) for r in target_ids]
            full_raw = [rank_score(full_eval, r) for r in target_ids]
            broad_f = filtered_ranks(broad_raw)
            full_f = filtered_ranks(full_raw)

            for rid, br, fr in zip(target_ids, broad_f, full_f, strict=True):
                rows.append({
                    "protein_id": q,
                    "reaction_id": rid,
                    "broad_rank": br,
                    "full_rank": fr,
                    "protein_seen": q in seen_p,
                    "reaction_seen": rid in seen_r,
                    "candidate_count_filtered": len(rids) - len(masked) - len(target_ids) + 1,
                })
            qa.append({
                "query_id": q,
                "target_relations": len(target_ids),
                "train_memory_weight": rw,
                "functional_support": float(fa.mean()),
                "structural_support": float(cav.mean()),
            })
        done = min(start + args.batch_size, len(queries))
        if done % 640 == 0 or done == len(queries):
            print("e2r", done, "/", len(queries), flush=True)

    ef = pd.DataFrame(rows)
    qf = pd.DataFrame(qa)
    ef.to_csv(OUT / "edge_metrics.csv.gz", index=False)
    qf.to_csv(OUT / "query_audit.csv.gz", index=False)

    def met(df: pd.DataFrame, col: str) -> dict:
        r = df[col].to_numpy(np.int64)
        return {
            "edges": int(len(r)),
            "mrr": float((1.0 / r).mean()),
            "hit10": float((r <= 10).mean()),
            "hit100": float((r <= 100).mean()),
            "hit1000": float((r <= 1000).mean()),
            "median_rank": float(np.median(r)),
        }

    masks = {
        "both_seen_edge_unseen": ef.protein_seen & ef.reaction_seen,
        "protein_cold_only": (~ef.protein_seen) & ef.reaction_seen,
        "reaction_cold_only": ef.protein_seen & (~ef.reaction_seen),
        "double_cold": (~ef.protein_seen) & (~ef.reaction_seen),
    }
    summary = {
        "schema": "bridge-current-edgewise-e2r-v1",
        "protocol": {
            "unit": "one held-out relation edge",
            "current_runtime": rt.version,
            "episodic_support": "none",
            "long_term_memory": "enabled",
            "clean2023_known_relations_masked": True,
            "candidate_universe": len(rids),
            "edge_filtering": "all other held-out positives for the same query removed by rank correction",
        },
        "all": {"broad": met(ef, "broad_rank"), "full": met(ef, "full_rank")},
        "slices": {
            name: {"broad": met(ef[mask], "broad_rank"), "full": met(ef[mask], "full_rank")}
            for name, mask in masks.items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
