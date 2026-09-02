from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.model_selection import ParameterSampler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.broad_rhea_metrics import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_TOP_PERCENTS,
    evaluate_full_candidate_ranks,
    summarize_query_metrics,
)
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks  # noqa: E402
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
)

DEV_ROOT = ROOT / "results/comprehensive_enzgfm_center_top1_v1/dev"
OUT = ROOT / "results/r2e_lambdarank_fusion_v1"
PRIMARY_MODELS = DEV_ROOT / "baseline_center"
SECONDARY_MODELS = DEV_ROOT / "candidate_center"
PRIMARY_PROTEINS = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
SECONDARY_PROTEINS = ROOT / "data/external/enzgfm_current/general_merged_650m_mean_v1"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1"
MAX_POOL_K = 500
ROUTER_THRESHOLD = 0.9
SEARCH_SEED = 20260902
SEARCH_COUNT = 18
FOLDS = (0, 1, 2)

FEATURE_NAMES = [
    "primary_raw_score", "secondary_raw_score",
    "primary_query_zscore", "secondary_query_zscore",
    "primary_log_rank_fraction", "secondary_log_rank_fraction",
    "primary_reciprocal_rank", "secondary_reciprocal_rank",
    "zscore_difference", "log_rank_difference", "best_log_rank", "worst_log_rank",
    "fallback_log_rank_fraction", "alternate_log_rank_fraction",
    "fallback_zscore", "alternate_zscore",
    "primary_top10", "secondary_top10", "primary_top50", "secondary_top50",
    "primary_top200", "secondary_top200",
    "max_train_binary_drfp_tanimoto", "low_similarity_router_flag",
]

PRIMARY_METRICS = ("mrr", "map", "ndcg_at_10", "hit_at_10")
ALL_GATE_METRICS = ("mrr", "map", "macro_roc_auc", "ndcg_at_10", "hit_at_10", "hit_at_20", "hit_at_50")
# A learned prefix followed by an exact fallback ranking has a deterministic total
# order but no single globally comparable continuous score.  Therefore AUROC is
# derived from exact positive ranks for BOTH the candidate and its current-router
# baseline inside this family.  This avoids mixing rank-AUROC with legacy average-tie
# score-AUROC (whose fold0 difference is only ~2e-7 but is a different convention).


@dataclass(frozen=True)
class Config:
    config_id: str
    pool_k: int
    prefix_k: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    reg_lambda: float
    rounds: int
    lambdarank_pairs: int


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big") % (2**31 - 1)


def _metric_map(frame: pd.DataFrame) -> dict[str, float]:
    summary = summarize_query_metrics(frame, budgets=DEFAULT_BUDGETS, top_percents=DEFAULT_TOP_PERCENTS)
    return {
        "mrr": float(summary["mrr"]),
        "map": float(summary["map"]),
        "macro_roc_auc": float(summary["macro_roc_auc"]),
        "ndcg_at_10": float(summary["ndcg_at_10"]),
        "hit_at_10": float(summary["hit_at_10"]),
        "hit_at_20": float(summary["hit_at_20"]),
        "hit_at_50": float(summary["hit_at_50"]),
        "median_best_positive_rank": float(summary["median_best_positive_rank"]),
    }


def _lexical_rank(ids: list[str]) -> np.ndarray:
    order = np.argsort(np.asarray(ids, dtype=object), kind="stable")
    ranks = np.empty(len(ids), dtype=np.int32)
    ranks[order] = np.arange(len(ids), dtype=np.int32)
    return ranks


def _full_order(scores: np.ndarray, lexical_rank: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((lexical_rank, -np.asarray(scores, dtype=np.float64))).astype(np.int32)
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, inverse


def _build_features(
    primary_scores: np.ndarray,
    secondary_scores: np.ndarray,
    rows: np.ndarray,
    primary_ranks: np.ndarray,
    secondary_ranks: np.ndarray,
    fallback_is_secondary: bool,
    similarity: float,
) -> np.ndarray:
    n = len(primary_scores)
    p_mean = float(primary_scores.mean()); p_std = max(float(primary_scores.std()), 1e-6)
    s_mean = float(secondary_scores.mean()); s_std = max(float(secondary_scores.std()), 1e-6)
    pz = (primary_scores[rows] - p_mean) / p_std
    sz = (secondary_scores[rows] - s_mean) / s_std
    pr = primary_ranks[rows].astype(np.float64)
    sr = secondary_ranks[rows].astype(np.float64)
    denom = math.log1p(n)
    plog = np.log1p(pr) / denom
    slog = np.log1p(sr) / denom
    if fallback_is_secondary:
        flog, alog = slog, plog
        fz, az = sz, pz
    else:
        flog, alog = plog, slog
        fz, az = pz, sz
    cols = [
        primary_scores[rows], secondary_scores[rows], pz, sz,
        plog, slog, 1.0 / pr, 1.0 / sr,
        pz - sz, plog - slog, np.minimum(plog, slog), np.maximum(plog, slog),
        flog, alog, fz, az,
        (pr <= 10).astype(np.float32), (sr <= 10).astype(np.float32),
        (pr <= 50).astype(np.float32), (sr <= 50).astype(np.float32),
        (pr <= 200).astype(np.float32), (sr <= 200).astype(np.float32),
        np.full(len(rows), float(similarity), dtype=np.float32),
        np.full(len(rows), float(fallback_is_secondary), dtype=np.float32),
    ]
    out = np.column_stack(cols).astype(np.float32, copy=False)
    if out.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((out.shape, len(FEATURE_NAMES)))
    return out


def _load_fold_embeddings(fold: int, device: torch.device):
    p0, ids0 = load_protein_library(PRIMARY_PROTEINS)
    p1, ids1 = load_protein_library(SECONDARY_PROTEINS)
    if ids0 != ids1:
        raise ValueError("Primary/secondary protein candidate order differs")
    d0 = PRIMARY_MODELS / f"fold{fold}"
    d1 = SECONDARY_MODELS / f"fold{fold}"
    s0 = load_feature_schema(d0); s1 = load_feature_schema(d1)
    reaction0, rids0 = load_registered_reaction_feature_library(REACTIONS, s0)
    reaction1, rids1 = load_registered_reaction_feature_library(REACTIONS, s1)
    if rids0 != rids1 or not np.array_equal(reaction0, reaction1):
        raise ValueError("Primary/secondary reaction feature libraries differ")
    m0 = load_models(d0 / "models", "production", device)
    m1 = load_models(d1 / "models", "production", device)
    if len(m0) != 1 or len(m1) != 1:
        raise ValueError("V1 expects one production checkpoint per fold/source")
    pe0 = encode_chunks(m0[0], p0, kind="protein", device=device, chunk_size=8192)
    pe1 = encode_chunks(m1[0], p1, kind="protein", device=device, chunk_size=8192)
    # Match evaluate_broad_rhea_benchmark exactly: encode the complete registered
    # reaction library before selecting query rows.  Encoding only the dev subset
    # is mathematically equivalent, but changing GEMM shapes can change the last
    # float32 bits and therefore a score tie by one rank.
    re0 = encode_chunks(m0[0], reaction0, kind="reaction", device=device, chunk_size=8192)
    re1 = encode_chunks(m1[0], reaction1, kind="reaction", device=device, chunk_size=8192)
    return ids0, reaction0, rids0, pe0, pe1, re0, re1


def prepare_fold(fold: int, device_name: str) -> None:
    device = torch.device(device_name)
    ids, reaction_features, reaction_ids, pe0, pe1, re0, re1 = _load_fold_embeddings(fold, device)
    candidate_index = {value: i for i, value in enumerate(ids)}
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}
    lex_rank = _lexical_rank(ids)
    dev_pairs = pd.read_csv(DEV_ROOT / "baseline_base" / f"fold{fold}" / "dev_pairs.csv", dtype=str).fillna("")
    query_ids = sorted(dev_pairs["reaction_id"].unique())
    positives = dev_pairs.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    cell = f"clean2023_internal_double_cold_salted_comprehensive_enzgfm_center_top1_v1_dev_20260901_fold{fold}"
    difficulty = pd.read_csv(DEV_ROOT / "difficulty" / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    sim_map = dict(zip(difficulty["reaction_id"].astype(str), difficulty["max_train_drfp_tanimoto"].astype(float)))
    qrows = [reaction_index[q] for q in query_ids]

    query_ptr = [0]; pos_ptr = [0]
    all_x: list[np.ndarray] = []; all_rows: list[np.ndarray] = []; all_y: list[np.ndarray] = []
    all_r0: list[np.ndarray] = []; all_r1: list[np.ndarray] = []; all_fb: list[np.ndarray] = []
    pos_rows_all: list[np.ndarray] = []; pos_fb_all: list[np.ndarray] = []
    baseline_records: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    # Keep the exact query-batch shape used by the full-candidate evaluator.
    batch = 32
    for start in range(0, len(query_ids), batch):
        stop = min(start + batch, len(query_ids))
        rows = torch.as_tensor(qrows[start:stop], dtype=torch.long, device=device)
        with torch.no_grad():
            s0b = (re0[rows] @ pe0.T).float().cpu().numpy()
            s1b = (re1[rows] @ pe1.T).float().cpu().numpy()
        for local, q in enumerate(query_ids[start:stop]):
            s0 = s0b[local].astype(np.float32, copy=False)
            s1 = s1b[local].astype(np.float32, copy=False)
            o0, inv0 = _full_order(s0, lex_rank)
            o1, inv1 = _full_order(s1, lex_rank)
            sim = float(sim_map[q]); use_secondary = sim < ROUTER_THRESHOLD
            fb_inv = inv1 if use_secondary else inv0
            union = np.unique(np.concatenate([o0[:MAX_POOL_K], o1[:MAX_POOL_K]])).astype(np.int32)
            r0 = inv0[union]; r1 = inv1[union]; fb = fb_inv[union]
            x = _build_features(s0, s1, union, inv0, inv1, use_secondary, sim)
            positive_rows = np.asarray(sorted(candidate_index[p] for p in positives[q]), dtype=np.int32)
            y = np.isin(union, positive_rows).astype(np.uint8)
            pfb = fb_inv[positive_rows].astype(np.int32)
            metrics = evaluate_full_candidate_ranks(pfb, len(ids))
            baseline_records.append({"fold": fold, "query_id": q, "similarity": sim, "used_secondary": use_secondary, **metrics})
            all_x.append(x); all_rows.append(union); all_y.append(y); all_r0.append(r0.astype(np.int32)); all_r1.append(r1.astype(np.int32)); all_fb.append(fb.astype(np.int32))
            pos_rows_all.append(positive_rows); pos_fb_all.append(pfb)
            query_ptr.append(query_ptr[-1] + len(union)); pos_ptr.append(pos_ptr[-1] + len(positive_rows))
            audit.append({"fold": fold, "query_id": q, "similarity": sim, "use_secondary": use_secondary, "union_max500": len(union), "positives": len(positive_rows), "positive_in_union": int(y.sum())})
        print(f"prepare fold={fold} {stop}/{len(query_ids)}", flush=True)
    out = OUT / "prepared" / f"fold{fold}"; out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "cache.npz",
        X=np.concatenate(all_x), candidate_rows=np.concatenate(all_rows), labels=np.concatenate(all_y),
        primary_ranks=np.concatenate(all_r0), secondary_ranks=np.concatenate(all_r1), fallback_ranks=np.concatenate(all_fb),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        positive_rows=np.concatenate(pos_rows_all), positive_fallback_ranks=np.concatenate(pos_fb_all), pos_ptr=np.asarray(pos_ptr, dtype=np.int64),
        lexical_rank=lex_rank,
    )
    pd.DataFrame({"query_id": query_ids}).to_csv(out / "queries.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "audit.csv", index=False)
    base = pd.DataFrame(baseline_records); base.to_csv(out / "current_router_query_metrics.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"fold": fold, "queries": len(query_ids), "candidate_count": len(ids), "max_pool_k": MAX_POOL_K, "rows": int(query_ptr[-1]), "positive_in_union_queries": int(sum(x["positive_in_union"] > 0 for x in audit)), "baseline": _metric_map(base)}, indent=2) + "\n")


def _load_cache(fold: int) -> dict[str, object]:
    p = OUT / "prepared" / f"fold{fold}"
    z = np.load(p / "cache.npz")
    return {"z": z, "queries": pd.read_csv(p / "queries.csv", dtype=str)["query_id"].astype(str).tolist(), "baseline": pd.read_csv(p / "current_router_query_metrics.csv", dtype={"query_id": str})}


def _filtered(cache: dict[str, object], pool_k: int) -> dict[str, object]:
    z = cache["z"]; ptr = z["query_ptr"]
    xs=[]; ys=[]; rows=[]; fbr=[]; qptr=[0]
    for qi in range(len(ptr)-1):
        a,b=int(ptr[qi]),int(ptr[qi+1]); keep=(z["primary_ranks"][a:b] <= pool_k) | (z["secondary_ranks"][a:b] <= pool_k)
        xs.append(z["X"][a:b][keep]); ys.append(z["labels"][a:b][keep]); rows.append(z["candidate_rows"][a:b][keep]); fbr.append(z["fallback_ranks"][a:b][keep]); qptr.append(qptr[-1]+int(keep.sum()))
    return {"X":np.concatenate(xs),"y":np.concatenate(ys),"rows":np.concatenate(rows),"fallback_ranks":np.concatenate(fbr),"query_ptr":np.asarray(qptr,dtype=np.int64),"positive_rows":z["positive_rows"],"positive_fallback_ranks":z["positive_fallback_ranks"],"pos_ptr":z["pos_ptr"],"lexical_rank":z["lexical_rank"],"queries":cache["queries"],"baseline":cache["baseline"]}


def _sample_training_group(
    X: np.ndarray,
    y: np.ndarray,
    rows: np.ndarray,
    *,
    query_id: str,
    pool_k: int,
    hard_negatives: int = 128,
    random_negatives: int = 32,
) -> np.ndarray:
    """Keep all positives plus deterministic source-hard and random negatives.

    Candidate identity is never a model feature.  ``rows`` is used only as a stable
    tie-break for equal hardness during training-set construction.
    """
    positives = np.flatnonzero(y > 0)
    negatives = np.flatnonzero(y <= 0)
    if len(positives) == 0:
        return np.empty(0, dtype=np.int64)
    best_rank_column = FEATURE_NAMES.index("best_log_rank")
    hard_order = negatives[
        np.lexsort((rows[negatives], X[negatives, best_rank_column]))
    ]
    hard = hard_order[: min(hard_negatives, len(hard_order))]
    remaining = hard_order[len(hard) :]
    if len(remaining) and random_negatives > 0:
        rng = np.random.default_rng(
            _stable_seed(f"trainneg|{SEARCH_SEED}|{pool_k}|{query_id}")
        )
        random = rng.choice(
            remaining, size=min(random_negatives, len(remaining)), replace=False
        ).astype(np.int64)
    else:
        random = np.empty(0, dtype=np.int64)
    return np.concatenate([positives, hard, random]).astype(np.int64, copy=False)


def _training_matrix(caches: list[dict[str, object]], pool_k: int):
    xs=[]; ys=[]; groups=[]
    for c in caches:
        f=_filtered(c,pool_k); p=f["query_ptr"]
        for qi in range(len(p)-1):
            a,b=int(p[qi]),int(p[qi+1]); y=f["y"][a:b]
            keep=_sample_training_group(
                f["X"][a:b], y, f["rows"][a:b],
                query_id=str(f["queries"][qi]), pool_k=pool_k,
            )
            if len(keep) == 0:
                continue
            xs.append(f["X"][a:b][keep]); ys.append(y[keep].astype(np.float32)); groups.append(len(keep))
    if not xs: raise ValueError("No positive-containing LambdaRank training groups")
    return np.concatenate(xs),np.concatenate(ys),groups


def _train(caches: list[dict[str, object]], cfg: Config, seed: int) -> xgb.Booster:
    X,y,groups=_training_matrix(caches,cfg.pool_k)
    dm=xgb.DMatrix(X,label=y); dm.set_group(groups)
    params={
        "objective":"rank:ndcg","eval_metric":"ndcg@10","tree_method":"hist","device":"cuda" if torch.cuda.is_available() else "cpu",
        "max_depth":cfg.max_depth,"eta":cfg.learning_rate,"min_child_weight":cfg.min_child_weight,"lambda":cfg.reg_lambda,
        "subsample":0.9,"colsample_bytree":0.9,"lambdarank_pair_method":"topk","lambdarank_num_pair_per_sample":cfg.lambdarank_pairs,
        "seed":seed,"verbosity":0,
    }
    return xgb.train(params,dm,num_boost_round=cfg.rounds)


def _evaluate(booster: xgb.Booster, cache: dict[str, object], cfg: Config, fold: int) -> pd.DataFrame:
    f=_filtered(cache,cfg.pool_k); ptr=f["query_ptr"]
    pred=booster.predict(xgb.DMatrix(f["X"]))
    lex=f["lexical_rank"]; posptr=f["pos_ptr"]; out=[]
    for qi,q in enumerate(f["queries"]):
        a,b=int(ptr[qi]),int(ptr[qi+1]); rows=f["rows"][a:b]; local_pred=pred[a:b]; fb=f["fallback_ranks"][a:b]
        order=np.lexsort((lex[rows],-local_pred)); take=order[:min(cfg.prefix_k,len(order))]
        selected_rows=rows[take]; selected_fb=fb[take]; selected_pos={int(r):i+1 for i,r in enumerate(selected_rows)}
        pa,pb=int(posptr[qi]),int(posptr[qi+1]); prows=f["positive_rows"][pa:pb]; pfb=f["positive_fallback_ranks"][pa:pb]
        new=[]
        for r,fr in zip(prows,pfb,strict=True):
            r=int(r); fr=int(fr)
            if r in selected_pos:
                nr=selected_pos[r]
            else:
                before=int(np.count_nonzero(selected_fb < fr))
                nr=len(selected_rows)+fr-before
            new.append(nr)
        metrics=evaluate_full_candidate_ranks(np.asarray(new,dtype=np.int64),len(lex))
        out.append({"fold":fold,"query_id":q,"pool_k":cfg.pool_k,"prefix_k":cfg.prefix_k,**metrics})
    return pd.DataFrame(out)


def _configs() -> list[Config]:
    space={
        "pool_k":[50,100,200,500],"prefix_k":[10,20,50,100],"max_depth":[2,3,4],"learning_rate":[0.03,0.05,0.08,0.12],
        "min_child_weight":[1.0,5.0,10.0],"reg_lambda":[1.0,5.0,10.0,20.0],"rounds":[50,80,120,180],"lambdarank_pairs":[8,16,20,32],
    }
    sampled=list(ParameterSampler(space,n_iter=SEARCH_COUNT,random_state=SEARCH_SEED))
    sampled.append({"pool_k":50,"prefix_k":50,"max_depth":3,"learning_rate":0.05,"min_child_weight":1.0,"reg_lambda":10.0,"rounds":80,"lambdarank_pairs":20})
    unique={}
    for d in sampled:
        if int(d["prefix_k"]) > 2*int(d["pool_k"]): continue
        key=json.dumps(d,sort_keys=True); unique[key]=d
    out=[]
    for i,(key,d) in enumerate(sorted(unique.items())):
        cid=f"cfg_{i:02d}_"+hashlib.blake2b(key.encode(),digest_size=4).hexdigest()
        out.append(Config(cid,int(d["pool_k"]),int(d["prefix_k"]),int(d["max_depth"]),float(d["learning_rate"]),float(d["min_child_weight"]),float(d["reg_lambda"]),int(d["rounds"]),int(d["lambdarank_pairs"])))
    return out


def _geomean_ratio(candidate: dict[str,float], baseline: dict[str,float]) -> float:
    vals=[]
    for k in PRIMARY_METRICS:
        if baseline[k] <= 0: return 0.0
        vals.append(candidate[k]/baseline[k])
    return float(math.exp(sum(math.log(max(v,1e-12)) for v in vals)/len(vals)))


def search() -> None:
    caches={f:_load_cache(f) for f in FOLDS}; configs=_configs(); OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"search_configurations.json").write_text(json.dumps([c.__dict__ for c in configs],indent=2)+"\n")
    rows=[]; query_frames=[]
    baseline_all=pd.concat([caches[f]["baseline"] for f in FOLDS],ignore_index=True); baseline_metrics=_metric_map(baseline_all)
    for ci,cfg in enumerate(configs):
        fold_frames=[]; fold_deltas={}
        for holdout in FOLDS:
            trainc=[caches[f] for f in FOLDS if f!=holdout]
            booster=_train(trainc,cfg,_stable_seed(f"{cfg.config_id}|{holdout}"))
            q=_evaluate(booster,caches[holdout],cfg,holdout); fold_frames.append(q)
            bm=_metric_map(caches[holdout]["baseline"]); cm=_metric_map(q); fold_deltas[str(holdout)]={k:cm[k]-bm[k] for k in ALL_GATE_METRICS}
        qa=pd.concat(fold_frames,ignore_index=True); cm=_metric_map(qa); delta={k:cm[k]-baseline_metrics[k] for k in ALL_GATE_METRICS}
        no_reg=all(delta[k]>=-1e-12 for k in ALL_GATE_METRICS)
        fold_safe=all(v["mrr"]>=-0.005 and v["map"]>=-0.005 for v in fold_deltas.values())
        gm=_geomean_ratio(cm,baseline_metrics); material=(gm>=1.005 and (delta["mrr"]>=0.003 or delta["map"]>=0.003 or delta["hit_at_10"]>=0.01))
        rows.append({**cfg.__dict__,**{f"baseline_{k}":baseline_metrics[k] for k in ALL_GATE_METRICS},**{f"candidate_{k}":cm[k] for k in ALL_GATE_METRICS},**{f"delta_{k}":delta[k] for k in ALL_GATE_METRICS},"primary_geomean_ratio":gm,"pooled_no_regression":no_reg,"fold_safe":fold_safe,"material_gate":material,"fold_deltas_json":json.dumps(fold_deltas,sort_keys=True)})
        qa.insert(0,"config_id",cfg.config_id); query_frames.append(qa)
        print(json.dumps({"progress":f"{ci+1}/{len(configs)}","config":cfg.config_id,"gm":gm,"material":material,"delta":delta}),flush=True)
    result=pd.DataFrame(rows); admissible=result[result["pooled_no_regression"] & result["fold_safe"]].copy()
    if len(admissible):
        admissible=admissible.sort_values(["primary_geomean_ratio","candidate_hit_at_50","config_id"],ascending=[False,False,True]); selected=admissible.iloc[0].to_dict()
    else: selected=None
    result.to_csv(OUT/"development_search.csv",index=False); pd.concat(query_frames,ignore_index=True).to_csv(OUT/"development_oof_query_metrics.csv",index=False)
    payload={"status":"development_selected_material" if selected and bool(selected["material_gate"]) else "development_no_material_winner" if selected else "development_no_admissible_config","baseline":baseline_metrics,"selected":selected,"configuration_count":len(configs),"selection_rule":"constraint-satisfying maximum primary geometric-mean relative gain; Hit@50 then config id tie break","confirmation_allowed":bool(selected and selected["material_gate"]),"external_or_outer_metrics_used":False}
    (OUT/"development_result.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


def fit_selected() -> None:
    result=json.loads((OUT/"development_result.json").read_text())
    if not result.get("confirmation_allowed"): raise RuntimeError("Development material gate did not pass")
    s=result["selected"]; cfg=Config(s["config_id"],int(s["pool_k"]),int(s["prefix_k"]),int(s["max_depth"]),float(s["learning_rate"]),float(s["min_child_weight"]),float(s["reg_lambda"]),int(s["rounds"]),int(s["lambdarank_pairs"]))
    caches=[_load_cache(f) for f in FOLDS]; booster=_train(caches,cfg,_stable_seed("final_dev_ranker")); out=OUT/"selected"; out.mkdir(parents=True,exist_ok=True); booster.save_model(out/"ranker.json"); (out/"config.json").write_text(json.dumps(cfg.__dict__,indent=2)+"\n"); print(json.dumps(cfg.__dict__,indent=2))


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","search","fit-selected"]); ap.add_argument("--fold",type=int,choices=FOLDS); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=ap.parse_args()
    if a.stage=="prepare":
        if a.fold is None: raise ValueError("--fold required for prepare")
        prepare_fold(a.fold,a.device)
    elif a.stage=="search": search()
    else: fit_selected()

if __name__=="__main__": main()
