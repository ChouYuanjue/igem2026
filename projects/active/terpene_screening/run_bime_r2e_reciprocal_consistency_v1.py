from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as structural
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks

SOURCE = ROOT / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1"
OUT = ROOT / "results/bime_rank_unified_v1/r2e_reciprocal_consistency_v1"
FOLDS = (0, 1, 2)
FIXED_CONFIG = structural.FIXED_CONFIG
PREFIX_K = structural.PREFIX_K
BASE_FEATURE_NAMES = list(structural.FEATURE_NAMES)
RECIPROCAL_FEATURE_NAMES = [
    "primary_reverse_log_rank_fraction",
    "secondary_reverse_log_rank_fraction",
    "primary_reverse_reciprocal_rank",
    "secondary_reverse_reciprocal_rank",
    "primary_reverse_top10",
    "secondary_reverse_top10",
    "primary_bidirectional_mean_log_rank",
    "secondary_bidirectional_mean_log_rank",
    "primary_bidirectional_log_rank_gap",
    "secondary_bidirectional_log_rank_gap",
    "best_bidirectional_mean_log_rank",
    "best_reverse_log_rank",
]
FEATURE_NAMES = BASE_FEATURE_NAMES + RECIPROCAL_FEATURE_NAMES


def _load_source_cache(fold: int) -> dict[str, object]:
    p = SOURCE / "prepared" / f"fold{fold}"
    with np.load(p / "cache.npz") as z:
        arrays = {k: z[k] for k in z.files}
    q = pd.read_csv(p / "queries.csv", dtype=str)["query_id"].astype(str).tolist()
    return {"z": arrays, "queries": q}


def _reaction_lexical_rank(reaction_ids: list[str]) -> np.ndarray:
    order = np.argsort(np.asarray(reaction_ids, dtype=object), kind="stable")
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(order), dtype=np.int32)
    return rank


def _reverse_ranks_for_pairs(
    reaction_embeddings: torch.Tensor,
    protein_embeddings: torch.Tensor,
    candidate_rows: np.ndarray,
    query_reaction_rows: np.ndarray,
    reaction_lexical_rank: np.ndarray,
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Exact ordinal reverse rank for each cached (query reaction, candidate protein) pair.

    For a pair (r,p), rank r among every reaction for p. Equal scores use lexical
    reaction ID as the deterministic stable tie-break, matching project ranking policy.
    Only protein candidates that actually occur in the forward shortlist are scored.
    """
    unique_rows, inverse = np.unique(candidate_rows.astype(np.int64), return_inverse=True)
    out = np.empty(len(candidate_rows), dtype=np.int32)
    lex_t = torch.as_tensor(reaction_lexical_rank, dtype=torch.int64, device=reaction_embeddings.device)
    qlex_all = reaction_lexical_rank[query_reaction_rows]
    for st in range(0, len(unique_rows), chunk_size):
        en = min(st + chunk_size, len(unique_rows))
        prows = torch.as_tensor(unique_rows[st:en], dtype=torch.long, device=reaction_embeddings.device)
        with torch.no_grad():
            score = (reaction_embeddings @ protein_embeddings[prows].T).float()
        pair_idx = np.flatnonzero((inverse >= st) & (inverse < en))
        local_col_np = inverse[pair_idx] - st
        qrow_np = query_reaction_rows[pair_idx]
        local_col = torch.as_tensor(local_col_np, dtype=torch.long, device=score.device)
        qrow = torch.as_tensor(qrow_np, dtype=torch.long, device=score.device)
        target = score[qrow, local_col]
        selected = score[:, local_col]
        greater = (selected > target.unsqueeze(0)).sum(dim=0, dtype=torch.int32)
        # Stable lexical tie-break among exact equal float scores.
        qlex = torch.as_tensor(qlex_all[pair_idx], dtype=torch.int64, device=score.device)
        tied_before = ((selected == target.unsqueeze(0)) & (lex_t.unsqueeze(1) < qlex.unsqueeze(0))).sum(dim=0, dtype=torch.int32)
        rank = 1 + greater + tied_before
        out[pair_idx] = rank.cpu().numpy().astype(np.int32, copy=False)
        if (en % (chunk_size * 8) == 0) or en == len(unique_rows):
            print(f"reverse-rank {en}/{len(unique_rows)} unique proteins", flush=True)
        del score, selected, target, greater, tied_before, rank
    return out


def append_reciprocal_features(
    base_features: np.ndarray,
    primary_reverse_ranks: np.ndarray,
    secondary_reverse_ranks: np.ndarray,
    reaction_count: int,
) -> np.ndarray:
    X0 = np.asarray(base_features, dtype=np.float32)
    rr0 = np.asarray(primary_reverse_ranks, dtype=np.int32)
    rr1 = np.asarray(secondary_reverse_ranks, dtype=np.int32)
    if len(X0) != len(rr0) or len(X0) != len(rr1):
        raise ValueError("reciprocal feature rows are not aligned")
    denom = math.log1p(int(reaction_count))
    rlog0 = (np.log1p(rr0.astype(np.float64)) / denom).astype(np.float32)
    rlog1 = (np.log1p(rr1.astype(np.float64)) / denom).astype(np.float32)
    rrr0 = (1.0 / rr0.astype(np.float64)).astype(np.float32)
    rrr1 = (1.0 / rr1.astype(np.float64)).astype(np.float32)
    rt10_0 = (rr0 <= 10).astype(np.float32); rt10_1 = (rr1 <= 10).astype(np.float32)
    p_log = X0[:, BASE_FEATURE_NAMES.index("primary_log_rank_fraction")]
    s_log = X0[:, BASE_FEATURE_NAMES.index("secondary_log_rank_fraction")]
    mean0 = 0.5 * (p_log + rlog0); mean1 = 0.5 * (s_log + rlog1)
    gap0 = np.abs(p_log - rlog0); gap1 = np.abs(s_log - rlog1)
    extra = np.column_stack([
        rlog0, rlog1, rrr0, rrr1, rt10_0, rt10_1,
        mean0, mean1, gap0, gap1, np.minimum(mean0, mean1), np.minimum(rlog0, rlog1),
    ]).astype(np.float32, copy=False)
    out = np.concatenate([X0, extra], axis=1).astype(np.float32, copy=False)
    if out.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((out.shape, len(FEATURE_NAMES)))
    return out


def prepare_fold(fold: int, device_name: str, chunk_size: int) -> None:
    device = torch.device(device_name)
    src = _load_source_cache(fold)
    z = src["z"]
    X0 = np.asarray(z["X"], dtype=np.float32)
    candidate_rows = np.asarray(z["candidate_rows"], dtype=np.int32)
    ptr = np.asarray(z["query_ptr"], dtype=np.int64)
    query_ids = list(src["queries"])

    ids, _, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    ridx = {r: i for i, r in enumerate(reaction_ids)}
    qrows = np.asarray([ridx[q] for q in query_ids], dtype=np.int32)
    pair_qrows = np.empty(len(candidate_rows), dtype=np.int32)
    for qi, rr in enumerate(qrows):
        pair_qrows[int(ptr[qi]):int(ptr[qi + 1])] = rr
    rlex = _reaction_lexical_rank(reaction_ids)

    print(f"fold={fold} primary reverse ranks", flush=True)
    rr0 = _reverse_ranks_for_pairs(re0, pe0, candidate_rows, pair_qrows, rlex, chunk_size=chunk_size)
    print(f"fold={fold} secondary reverse ranks", flush=True)
    rr1 = _reverse_ranks_for_pairs(re1, pe1, candidate_rows, pair_qrows, rlex, chunk_size=chunk_size)

    X = append_reciprocal_features(X0, rr0, rr1, len(reaction_ids))

    out = OUT / "prepared" / f"fold{fold}"; out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "cache.npz",
        X=X,
        labels=z["labels"], candidate_rows=candidate_rows, fallback_ranks=z["fallback_ranks"],
        query_ptr=ptr, positive_rows=z["positive_rows"], positive_fallback_ranks=z["positive_fallback_ranks"],
        pos_ptr=z["pos_ptr"], lexical_rank=z["lexical_rank"],
        primary_reverse_ranks=rr0, secondary_reverse_ranks=rr1,
    )
    pd.DataFrame({"query_id": query_ids}).to_csv(out / "queries.csv", index=False)
    audit = pd.DataFrame({
        "fold": fold,
        "query_id": np.repeat(np.asarray(query_ids, dtype=object), np.diff(ptr)),
        "candidate_row": candidate_rows,
        "primary_reverse_rank": rr0,
        "secondary_reverse_rank": rr1,
    })
    audit.to_csv(out / "pair_reverse_ranks.csv.gz", index=False, compression="gzip")
    summary = {
        "fold": fold, "queries": len(query_ids), "cache_pairs": len(candidate_rows),
        "unique_shortlist_proteins": int(len(np.unique(candidate_rows))), "reaction_candidates": len(reaction_ids),
        "feature_count": len(FEATURE_NAMES),
        "primary_reverse_top10_fraction": float((rr0 <= 10).mean()),
        "secondary_reverse_top10_fraction": float((rr1 <= 10).mean()),
        "median_primary_reverse_rank": float(np.median(rr0)),
        "median_secondary_reverse_rank": float(np.median(rr1)),
        "external_metrics_used": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def _load_cache(fold: int) -> dict[str, object]:
    p = OUT / "prepared" / f"fold{fold}"
    with np.load(p / "cache.npz") as z:
        arrays = {k: z[k] for k in z.files}
    return {"z": arrays, "queries": pd.read_csv(p / "queries.csv", dtype=str)["query_id"].astype(str).tolist()}


def _training_matrix(caches: list[dict[str, object]]):
    xs=[]; ys=[]; groups=[]
    hard_col = FEATURE_NAMES.index("best3_log_rank")
    for c in caches:
        z=c["z"]; ptr=z["query_ptr"]
        for qi,q in enumerate(c["queries"]):
            a,b=int(ptr[qi]),int(ptr[qi+1]); X=z["X"][a:b]; y=z["labels"][a:b]; rows=z["candidate_rows"][a:b]
            pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
            if not len(pos): continue
            hard=neg[np.lexsort((rows[neg],X[neg,hard_col]))][:128]
            rem=np.setdiff1d(neg,hard,assume_unique=False)
            rng=np.random.default_rng(base._stable_seed(f"bime-r2e-recip|{q}"))
            rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64)
            keep=np.concatenate([pos,hard,rnd])
            xs.append(X[keep]); ys.append(y[keep].astype(np.float32)); groups.append(len(keep))
    return np.concatenate(xs),np.concatenate(ys),groups


def _train(caches: list[dict[str, object]], seed: int) -> xgb.Booster:
    X,y,groups=_training_matrix(caches); dm=xgb.DMatrix(X,label=y); dm.set_group(groups)
    cfg=FIXED_CONFIG
    params={"objective":"rank:ndcg","eval_metric":"ndcg@10","tree_method":"hist","device":"cuda" if torch.cuda.is_available() else "cpu","max_depth":cfg.max_depth,"eta":cfg.learning_rate,"min_child_weight":cfg.min_child_weight,"lambda":cfg.reg_lambda,"subsample":0.9,"colsample_bytree":0.9,"lambdarank_pair_method":"topk","lambdarank_num_pair_per_sample":cfg.lambdarank_pairs,"seed":seed,"verbosity":0}
    return xgb.train(params,dm,num_boost_round=cfg.rounds)


def _evaluate(model: xgb.Booster, cache: dict[str, object], fold: int) -> pd.DataFrame:
    z=cache["z"]; ptr=z["query_ptr"]; pred=model.predict(xgb.DMatrix(z["X"])); lex=z["lexical_rank"]; pp=z["pos_ptr"]
    out=[]
    for qi,q in enumerate(cache["queries"]):
        a,b=int(ptr[qi]),int(ptr[qi+1]); rows=z["candidate_rows"][a:b]; local=pred[a:b]; fb=z["fallback_ranks"][a:b]
        order=np.lexsort((lex[rows],-local)); take=order[:min(PREFIX_K,len(order))]
        selected_rows=rows[take]; selected_fb=fb[take]; selected_pos={int(r):i+1 for i,r in enumerate(selected_rows)}
        pa,pb=int(pp[qi]),int(pp[qi+1]); prows=z["positive_rows"][pa:pb]; pfb=z["positive_fallback_ranks"][pa:pb]
        new=[]
        for r,fr in zip(prows,pfb,strict=True):
            r=int(r); fr=int(fr)
            if r in selected_pos: nr=selected_pos[r]
            else: nr=len(selected_rows)+fr-int(np.count_nonzero(selected_fb<fr))
            new.append(nr)
        out.append({"fold":fold,"query_id":q,**evaluate_full_candidate_ranks(np.asarray(new,dtype=np.int64),len(lex))})
    return pd.DataFrame(out)


def crossfit() -> None:
    caches={f:_load_cache(f) for f in FOLDS}; frames=[]
    for holdout in FOLDS:
        m=_train([caches[f] for f in FOLDS if f!=holdout],base._stable_seed(f"bime-r2e-recip|fold{holdout}"))
        q=_evaluate(m,caches[holdout],holdout); frames.append(q); print(f"crossfit fold={holdout} done",flush=True)
    new=pd.concat(frames,ignore_index=True)
    old=pd.read_csv(SOURCE/"development_oof_query_metrics.csv",dtype={"query_id":str})
    old_metrics=base._metric_map(old); new_metrics=base._metric_map(new); keys=base.ALL_GATE_METRICS
    delta={k:new_metrics[k]-old_metrics[k] for k in keys}; fold_delta={}
    for f in FOLDS:
        om=base._metric_map(old[old.fold.eq(f)]); nm=base._metric_map(new[new.fold.eq(f)])
        fold_delta[str(f)]={k:nm[k]-om[k] for k in keys}
    primary=("mrr","map","hit_at_20","hit_at_50")
    gm=float(math.exp(sum(math.log(max(new_metrics[k]/old_metrics[k],1e-12)) for k in primary)/len(primary)))
    fold_safe=all(v["mrr"]>=-0.005 and v["map"]>=-0.005 for v in fold_delta.values())
    pooled_safe=delta["mrr"]>=-0.002 and delta["map"]>=-0.002 and delta["hit_at_20"]>=-0.005 and delta["hit_at_50"]>=-0.005
    material=gm>=1.005 and (delta["mrr"]>=0.003 or delta["map"]>=0.003 or delta["hit_at_20"]>=0.01 or delta["hit_at_50"]>=0.01)
    admitted=bool(fold_safe and pooled_safe and material)
    OUT.mkdir(parents=True,exist_ok=True); new.to_csv(OUT/"development_oof_query_metrics.csv",index=False)
    payload={"status":"admitted" if admitted else "not_admitted","comparison":"same structural BiME-Rank folds/candidate union/fixed ranker capacity; only reverse-rank bidirectional-consistency features added","fixed_config":FIXED_CONFIG.__dict__,"old":old_metrics,"new":new_metrics,"delta":delta,"fold_delta":fold_delta,"primary_geomean_ratio":gm,"fold_safe":fold_safe,"pooled_safe":pooled_safe,"material":material,"admitted":admitted,"external_metrics_used":False,"admission_rule":"internal clean-dev only; fold MRR/MAP >= -0.005; pooled MRR/MAP >= -0.002 and Hit20/50 >= -0.005; primary geomean >=1.005; material gain in MRR/MAP >=.003 or Hit20/50 >=.01"}
    (OUT/"development_result.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


def fit_final() -> None:
    result=json.loads((OUT/"development_result.json").read_text())
    if not result.get("admitted"): raise RuntimeError("reciprocal consistency did not pass admission")
    caches=[_load_cache(f) for f in FOLDS]; m=_train(caches,base._stable_seed("bime-r2e-recip-final")); out=OUT/"selected"; out.mkdir(parents=True,exist_ok=True); m.save_model(out/"ranker.json")
    cfg={"feature_names":FEATURE_NAMES,"fixed_config":FIXED_CONFIG.__dict__,"training_folds":list(FOLDS),"external_metrics_used":False,"source_structural_ranker":str((SOURCE/'selected/ranker.json').relative_to(ROOT))}
    import hashlib
    cfg["ranker_sha256"]=hashlib.sha256((out/"ranker.json").read_bytes()).hexdigest(); (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n"); print(json.dumps(cfg,indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","crossfit","fit-final"]); ap.add_argument("--fold",type=int,choices=FOLDS); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--chunk-size",type=int,default=512); a=ap.parse_args()
    if a.stage=="prepare":
        if a.fold is None: raise ValueError("--fold required")
        prepare_fold(a.fold,a.device,a.chunk_size)
    elif a.stage=="crossfit": crossfit()
    else: fit_final()

if __name__ == "__main__":
    main()
