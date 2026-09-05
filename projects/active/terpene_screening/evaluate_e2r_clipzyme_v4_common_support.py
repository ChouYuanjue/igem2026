from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_unified_safe_system_e2r_anchored_lambdamart_v3 as v3
from projects.active.terpene_screening.e2r_anchored_lambdamart_runtime import AnchoredE2RRuntime
from projects.active.terpene_screening.evaluate_clipzyme_catalyst_common_support_v1 import _metrics, _norm
from projects.active.terpene_screening.run_e2r_clipzyme_anchored_lambdamart_v4 import anchored_order_v4

POSITIVE_SUPPORT = ROOT / "results/clipzyme_protein_support_v1/reactzyme_reaction_projected_double_cold_protein_support.csv"
CLIP_SCREEN = ROOT / "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p"
CLIP_REACTIONS = ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
V4_ROOT = ROOT / "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected"
OUT = ROOT / "results/clipzyme_native_extension_v1/e2r_common4222_current_v3_vs_clip_v4_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def projected_priority(full_order: np.ndarray, common_mask: np.ndarray, full_to_common: np.ndarray) -> np.ndarray:
    projected = full_order[common_mask[full_order]]
    rows = full_to_common[projected]
    if (rows < 0).any():
        raise AssertionError("projected order contains non-common row")
    score = np.empty(len(rows), dtype=np.float32)
    # Strictly decreasing unique priority; only ordering matters to _metrics.
    score[rows] = 1.0 - np.arange(len(rows), dtype=np.float32) / max(len(rows), 1)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pos = pd.read_csv(POSITIVE_SUPPORT, dtype=str).fillna("")
    pos = pos[pos["common_executable_positive"].str.lower().eq("true")][["reaction_id", "protein_id"]].drop_duplicates()
    if pos.reaction_id.nunique() != 109 or pos.protein_id.nunique() != 4161 or len(pos) != 4915:
        raise AssertionError("frozen common-support labels drifted")
    query_ids = sorted(pos.protein_id.unique())
    positives = pos.groupby("protein_id")["reaction_id"].agg(lambda s: set(s.astype(str))).to_dict()

    runtime = AnchoredE2RRuntime(device=args.device)
    full_ids = list(runtime.candidate_ids)
    full_idx = {r: i for i, r in enumerate(full_ids)}

    # The headline common support remains the original 4,222 author-executable reactions.
    old_entries = pd.read_csv(ROOT / "data/external/clipzyme_current/general_merged_reaction_embeddings_v1/entries.csv", dtype=str).fillna("")
    old_supported = old_entries[old_entries.clipzyme_supported.str.lower().eq("true")]
    common_ids = sorted(set(old_supported.reaction_id.astype(str)) & set(full_ids))
    if len(common_ids) != 4222:
        raise AssertionError(len(common_ids))
    common_set = set(common_ids)
    common_mask = np.asarray([r in common_set for r in full_ids], dtype=bool)
    full_to_common = np.full(len(full_ids), -1, dtype=np.int32)
    for i, r in enumerate(common_ids):
        full_to_common[full_idx[r]] = i

    # Official CLIPZyme query embeddings, matching the existing local common-support baseline.
    clip = pickle.load(CLIP_SCREEN.open("rb"))
    pids = [str(x) for x in clip["uniprots"]]
    pidx = {p: i for i, p in enumerate(pids)}
    h = clip["hiddens"]
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
    h = _norm(np.asarray(h, dtype=np.float32))
    if not set(query_ids) <= set(pidx):
        raise AssertionError("common-support query missing from official CLIPZyme screening embeddings")

    re = pd.read_csv(CLIP_REACTIONS / "entries.csv", dtype=str).fillna("")
    rmat = np.load(CLIP_REACTIONS / "embeddings.npy", mmap_mode="r")
    sup = re[re.clipzyme_supported.str.lower().eq("true")]
    ridx = {r: int(row) for r, row in sup[["reaction_id", "row"]].itertuples(index=False)}
    clip_support = np.asarray([r in ridx for r in full_ids], dtype=bool)
    clip_cand = np.zeros((len(full_ids), rmat.shape[1]), dtype=np.float32)
    clip_cand[clip_support] = _norm(np.asarray(rmat[[ridx[r] for r in full_ids if r in ridx]], dtype=np.float32))
    clip_ct = torch.as_tensor(clip_cand, dtype=torch.float32, device=runtime.device)

    cfg = json.loads((V4_ROOT / "config.json").read_text())
    selected = cfg["selected_config"]
    ranker = xgb.Booster()
    ranker.load_model(V4_ROOT / "ranker.json")
    if sha256_file(V4_ROOT / "ranker.json") != cfg["ranker_sha256"]:
        raise RuntimeError("V4 selected ranker hash mismatch")

    # Priority matrices are small enough on the 4,222 support and simplify exact metric reuse.
    score_v3 = np.empty((len(query_ids), len(common_ids)), dtype=np.float32)
    score_v4 = np.empty_like(score_v3)
    audit = []
    for qi, q in enumerate(query_ids):
        features = runtime.registered_query_features(q)
        S = runtime._expert_scores(features)
        ranks4 = np.stack([v3.full_ranks(S[e]) for e in range(4)], axis=1)
        union4 = np.asarray(sorted(set().union(*(set(map(int, v3.top_rows(S[e], v3.MAX_POOL))) for e in range(4)))), dtype=np.int32)
        X3 = v3.feature_matrix(S, union4, ranks4)
        pred3 = runtime.ranker.predict(xgb.DMatrix(X3))
        order3, _, selected3 = runtime.anchored_order(
            S, pred3, union4,
            protected_prefix=runtime.protected_prefix,
            pool_k=runtime.pool_k,
            prefix_k=runtime.prefix_k,
        )

        cq = torch.as_tensor(h[pidx[q]], dtype=torch.float32, device=runtime.device)
        with torch.no_grad():
            cscore = (clip_ct @ cq).detach().cpu().numpy().astype(np.float32)
        order4, local = anchored_order_v4(
            S, cscore, clip_support, True, ranker,
            protected_prefix=int(selected["protected_prefix"]),
            pool_k=int(selected["pool_k"]),
            prefix_k=int(selected["prefix_k"]),
        )
        score_v3[qi] = projected_priority(order3, common_mask, full_to_common)
        score_v4[qi] = projected_priority(order4, common_mask, full_to_common)
        audit.append({"query_id": q, "v3_selected": int(len(selected3)), **local})
        if (qi + 1) % 250 == 0 or qi + 1 == len(query_ids):
            print(f"common-support {qi+1}/{len(query_ids)}", flush=True)

    v3_sum, v3_q = _metrics(score_v3, query_ids, common_ids, positives)
    v4_sum, v4_q = _metrics(score_v4, query_ids, common_ids, positives)
    old = json.loads((ROOT / "results/clipzyme_catalyst_common_support_v1/summary.json").read_text())
    clip_sum = old["models"]["official_clipzyme"]["enzyme_to_reaction"]
    old_cat = old["models"]["catalyst_v3"]["enzyme_to_reaction"]

    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    v3_q.to_csv(out / "current_v3_query_metrics.csv", index=False)
    v4_q.to_csv(out / "clipzyme_v4_query_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "routing_audit.csv", index=False)
    metrics = ["mrr", "map", "ndcg_at_10", "hit_at_1", "hit_at_5", "hit_at_10", "hit_at_20", "hit_at_50"]
    result = {
        "protocol": "Frozen 4161-protein x 4222-reaction common support. Current V3 and selected V4 route on full 11081 reaction universe first, then filter to original common support. V4 selected solely on clean-dev before this run.",
        "selection_used_this_common_support": False,
        "queries": len(query_ids),
        "common_reaction_candidates": len(common_ids),
        "positive_pairs": len(pos),
        "full_route_candidate_count": len(full_ids),
        "models": {
            "official_clipzyme_historical_same_support": clip_sum,
            "historical_catalyst_v3_direct_same_support": old_cat,
            "current_four_expert_v3_projected": v3_sum,
            "clipzyme_five_expert_v4_projected": v4_sum,
        },
        "delta_v4_vs_current_v3": {k: float(v4_sum[k]) - float(v3_sum[k]) for k in metrics},
        "delta_v4_vs_historical_direct": {k: float(v4_sum[k]) - float(old_cat[k]) for k in metrics},
        "v4_ranker_sha256": cfg["ranker_sha256"],
        "v4_selected_config": selected,
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
