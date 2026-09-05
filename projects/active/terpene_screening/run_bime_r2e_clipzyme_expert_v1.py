from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening import run_r2e_lambdarank_fusion_v1 as base
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks

OUT = ROOT / "results/bime_rank_unified_v1/r2e_clipzyme_expert_v1"
CLIP_PROTEIN_ROOT = ROOT / "results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1"
CLIP_REACTION_ROOT = ROOT / "results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1"
OLD_OOF = ROOT / "results/r2e_lambdarank_fusion_v1/development_oof_query_metrics.csv"
FOLDS = (0, 1, 2)
POOL_K = 100
PREFIX_K = 100

EXTRA_FEATURE_NAMES = [
    "clip_raw_score",
    "clip_query_zscore",
    "clip_log_rank_fraction",
    "clip_reciprocal_rank",
    "clip_candidate_supported",
    "clip_query_supported",
    "clip_top10",
    "clip_top50",
    "clip_top100",
    "clip_z_minus_fallback",
    "clip_logrank_minus_fallback",
    "top10_votes3",
    "top50_votes3",
    "top100_votes3",
    "best3_log_rank",
    "best3_zscore",
]
FEATURE_NAMES = [*base.FEATURE_NAMES, *EXTRA_FEATURE_NAMES]
FIXED_CONFIG = base.Config(
    config_id="bime_r2e_clipzyme_fixed_from_cfg_07_392fe119",
    pool_k=100,
    prefix_k=100,
    max_depth=2,
    learning_rate=0.12,
    min_child_weight=5.0,
    reg_lambda=1.0,
    rounds=80,
    lambdarank_pairs=20,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_clip_assets(candidate_ids: list[str], device: torch.device):
    pe = pd.read_csv(CLIP_PROTEIN_ROOT / "entries.csv", dtype=str).fillna("")
    pmat = np.load(CLIP_PROTEIN_ROOT / "embeddings.npy", mmap_mode="r")
    prows = pe["candidate_row"].astype(int).to_numpy(np.int32)
    if len(pmat) != len(prows):
        raise RuntimeError("CLIP protein entries/embedding mismatch")
    if not np.array_equal(np.asarray([candidate_ids[r] for r in prows], dtype=object), pe["protein_id"].astype(str).to_numpy(object)):
        raise RuntimeError("CLIP protein candidate-row alignment drifted")
    support_lookup = np.full(len(candidate_ids), -1, dtype=np.int32)
    support_lookup[prows] = np.arange(len(prows), dtype=np.int32)
    ptorch = torch.as_tensor(np.asarray(pmat), dtype=torch.float32, device=device)

    re = pd.read_csv(CLIP_REACTION_ROOT / "entries.csv", dtype=str).fillna("")
    rmat = np.load(CLIP_REACTION_ROOT / "embeddings.npy", mmap_mode="r")
    supported = re[re["clipzyme_supported"].str.lower().eq("true")].copy()
    ridx = {str(r): int(row) for r, row in supported[["reaction_id", "row"]].itertuples(index=False)}
    return ptorch, prows, support_lookup, rmat, ridx


def _clip_features(
    base_x: np.ndarray,
    rows: np.ndarray,
    clip_scores_supported: np.ndarray | None,
    clip_inverse_supported: np.ndarray | None,
    clip_support_lookup: np.ndarray,
    query_supported: bool,
    candidate_count: int,
) -> np.ndarray:
    pos = clip_support_lookup[rows]
    candidate_supported = pos >= 0
    effective = candidate_supported & bool(query_supported)
    raw = np.zeros(len(rows), dtype=np.float32)
    z = np.zeros(len(rows), dtype=np.float32)
    cr = np.full(len(rows), candidate_count + 1, dtype=np.float64)
    if query_supported and clip_scores_supported is not None and clip_inverse_supported is not None:
        vals = clip_scores_supported
        mean = float(vals.mean())
        std = max(float(vals.std()), 1e-6)
        raw[effective] = vals[pos[effective]]
        z[effective] = (vals[pos[effective]] - mean) / std
        cr[effective] = clip_inverse_supported[pos[effective]].astype(np.float64)
    clip_denom = math.log1p(max(int(np.count_nonzero(clip_support_lookup >= 0)), 1))
    clog = np.ones(len(rows), dtype=np.float32)
    rr = np.zeros(len(rows), dtype=np.float32)
    clog[effective] = (np.log1p(cr[effective]) / clip_denom).astype(np.float32)
    rr[effective] = (1.0 / cr[effective]).astype(np.float32)
    c10 = (effective & (cr <= 10)).astype(np.float32)
    c50 = (effective & (cr <= 50)).astype(np.float32)
    c100 = (effective & (cr <= 100)).astype(np.float32)

    idx = {name: base.FEATURE_NAMES.index(name) for name in [
        "primary_query_zscore", "secondary_query_zscore",
        "primary_log_rank_fraction", "secondary_log_rank_fraction",
        "fallback_log_rank_fraction", "fallback_zscore",
        "primary_top10", "secondary_top10",
        "primary_top50", "secondary_top50",
    ]}
    fallback_z = base_x[:, idx["fallback_zscore"]]
    fallback_log = base_x[:, idx["fallback_log_rank_fraction"]]
    pz = base_x[:, idx["primary_query_zscore"]]
    sz = base_x[:, idx["secondary_query_zscore"]]
    plog = base_x[:, idx["primary_log_rank_fraction"]]
    slog = base_x[:, idx["secondary_log_rank_fraction"]]
    p10 = base_x[:, idx["primary_top10"]]
    s10 = base_x[:, idx["secondary_top10"]]
    p50 = base_x[:, idx["primary_top50"]]
    s50 = base_x[:, idx["secondary_top50"]]

    # Existing base features expose top200 rather than top100. Reconstruct base top100
    # from normalized log-ranks without introducing candidate identity.
    nlog100 = math.log1p(100.0) / math.log1p(candidate_count)
    p100 = (plog <= nlog100 + 1e-12).astype(np.float32)
    s100 = (slog <= nlog100 + 1e-12).astype(np.float32)
    best3_log = np.minimum(plog, slog)
    best3_log[effective] = np.minimum(best3_log[effective], clog[effective])
    best3_z = np.maximum(pz, sz)
    best3_z[effective] = np.maximum(best3_z[effective], z[effective])

    extra = np.column_stack([
        raw,
        z,
        clog,
        rr,
        candidate_supported.astype(np.float32),
        np.full(len(rows), float(bool(query_supported)), dtype=np.float32),
        c10,
        c50,
        c100,
        z - fallback_z,
        clog - fallback_log,
        p10 + s10 + c10,
        p50 + s50 + c50,
        p100 + s100 + c100,
        best3_log,
        best3_z,
    ]).astype(np.float32, copy=False)
    out = np.concatenate([base_x, extra], axis=1).astype(np.float32, copy=False)
    if out.shape[1] != len(FEATURE_NAMES):
        raise AssertionError((out.shape, len(FEATURE_NAMES)))
    return out


def prepare_fold(fold: int, device_name: str) -> None:
    device = torch.device(device_name)
    ids, _, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    candidate_index = {p: i for i, p in enumerate(ids)}
    reaction_index = {r: i for i, r in enumerate(reaction_ids)}
    lex = base._lexical_rank(ids)
    clip_pt, clip_candidate_rows, clip_lookup, clip_rmat, clip_ridx = _load_clip_assets(ids, device)
    clip_lex = lex[clip_candidate_rows]

    dev_pairs = pd.read_csv(base.DEV_ROOT / "baseline_base" / f"fold{fold}" / "dev_pairs.csv", dtype=str).fillna("")
    query_ids = sorted(dev_pairs["reaction_id"].astype(str).unique())
    positives = dev_pairs.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    cell = f"clean2023_internal_double_cold_salted_comprehensive_enzgfm_center_top1_v1_dev_20260901_fold{fold}"
    difficulty = pd.read_csv(base.DEV_ROOT / "difficulty" / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    sim_map = dict(zip(difficulty["reaction_id"].astype(str), difficulty["max_train_drfp_tanimoto"].astype(float)))
    qrows = [reaction_index[q] for q in query_ids]

    Xs=[]; ys=[]; candidate_rows_all=[]; fallback_all=[]; query_ptr=[0]
    pos_rows_all=[]; pos_fb_all=[]; pos_ptr=[0]
    audits=[]
    batch=32
    for st in range(0, len(query_ids), batch):
        stop=min(st+batch,len(query_ids))
        rows_t=torch.as_tensor(qrows[st:stop],dtype=torch.long,device=device)
        with torch.no_grad():
            s0b=(re0[rows_t] @ pe0.T).float().cpu().numpy()
            s1b=(re1[rows_t] @ pe1.T).float().cpu().numpy()
        # Score CLIP only for supported reaction queries in this batch.
        clip_q = []
        clip_local = []
        for j,q in enumerate(query_ids[st:stop]):
            if q in clip_ridx:
                clip_q.append(np.asarray(clip_rmat[clip_ridx[q]],dtype=np.float32))
                clip_local.append(j)
        clip_scores_by_local={}
        if clip_q:
            qt=torch.as_tensor(np.stack(clip_q),dtype=torch.float32,device=device)
            with torch.no_grad():
                cb=(qt @ clip_pt.T).float().cpu().numpy()
            for k,j in enumerate(clip_local):
                clip_scores_by_local[j]=cb[k].astype(np.float32,copy=False)

        for j,q in enumerate(query_ids[st:stop]):
            s0=s0b[j].astype(np.float32,copy=False); s1=s1b[j].astype(np.float32,copy=False)
            o0,inv0=base._full_order(s0,lex); o1,inv1=base._full_order(s1,lex)
            sim=float(sim_map[q]); use_secondary=sim < base.ROUTER_THRESHOLD
            fb_inv=inv1 if use_secondary else inv0
            query_supported=j in clip_scores_by_local
            clip_scores=clip_scores_by_local.get(j)
            clip_inverse=None; clip_top=np.empty(0,dtype=np.int32)
            if query_supported:
                corder=np.lexsort((clip_lex,-clip_scores)).astype(np.int32)
                clip_inverse=np.empty(len(corder),dtype=np.int32)
                clip_inverse[corder]=np.arange(1,len(corder)+1,dtype=np.int32)
                clip_top=clip_candidate_rows[corder[:POOL_K]]
            union=np.unique(np.concatenate([o0[:POOL_K],o1[:POOL_K],clip_top])).astype(np.int32)
            bx=base._build_features(s0,s1,union,inv0,inv1,use_secondary,sim)
            x=_clip_features(bx,union,clip_scores,clip_inverse,clip_lookup,query_supported,len(ids))
            positive_rows=np.asarray(sorted(candidate_index[p] for p in positives[q]),dtype=np.int32)
            y=np.isin(union,positive_rows).astype(np.uint8)
            pfb=fb_inv[positive_rows].astype(np.int32)
            Xs.append(x); ys.append(y); candidate_rows_all.append(union); fallback_all.append(fb_inv[union].astype(np.int32))
            pos_rows_all.append(positive_rows); pos_fb_all.append(pfb)
            query_ptr.append(query_ptr[-1]+len(union)); pos_ptr.append(pos_ptr[-1]+len(positive_rows))
            pos_clip_supported=int(np.count_nonzero(clip_lookup[positive_rows]>=0))
            pos_clip_top100=0
            if query_supported and clip_inverse is not None:
                pp=clip_lookup[positive_rows]; ok=pp>=0
                pos_clip_top100=int(np.count_nonzero(clip_inverse[pp[ok]]<=100))
            audits.append({
                "fold":fold,"query_id":q,"similarity":sim,"use_secondary":use_secondary,
                "union_size":len(union),"positives":len(positive_rows),"positive_in_union":int(y.sum()),
                "clip_query_supported":bool(query_supported),"clip_positive_supported":pos_clip_supported,
                "clip_positive_top100":pos_clip_top100,
            })
        print(f"prepare-r2e-clip fold={fold} {stop}/{len(query_ids)}",flush=True)

    out=OUT/"prepared"/f"fold{fold}"; out.mkdir(parents=True,exist_ok=True)
    np.savez(out/"cache.npz",X=np.concatenate(Xs),labels=np.concatenate(ys),candidate_rows=np.concatenate(candidate_rows_all),fallback_ranks=np.concatenate(fallback_all),query_ptr=np.asarray(query_ptr,dtype=np.int64),positive_rows=np.concatenate(pos_rows_all),positive_fallback_ranks=np.concatenate(pos_fb_all),pos_ptr=np.asarray(pos_ptr,dtype=np.int64),lexical_rank=lex)
    pd.DataFrame({"query_id":query_ids}).to_csv(out/"queries.csv",index=False)
    adf=pd.DataFrame(audits); adf.to_csv(out/"audit.csv",index=False)
    summary={
        "fold":fold,"queries":len(query_ids),"candidate_count":len(ids),"feature_count":len(FEATURE_NAMES),
        "clip_query_supported":int(adf.clip_query_supported.sum()),
        "clip_query_supported_fraction":float(adf.clip_query_supported.mean()),
        "queries_positive_in_union":int((adf.positive_in_union>0).sum()),
        "mean_union_size":float(adf.union_size.mean()),
        "clip_candidate_supported":int(np.count_nonzero(clip_lookup>=0)),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


def _load_cache(fold:int):
    p=OUT/"prepared"/f"fold{fold}"
    with np.load(p/"cache.npz") as z:
        arrays={k:z[k] for k in z.files}
    return {"z":arrays,"queries":pd.read_csv(p/"queries.csv",dtype=str).query_id.astype(str).tolist()}


def _training_matrix(caches:list[dict]):
    xs=[]; ys=[]; groups=[]
    hard_col=FEATURE_NAMES.index("best3_log_rank")
    for c in caches:
        z=c["z"]; ptr=z["query_ptr"]
        for qi,q in enumerate(c["queries"]):
            a,b=int(ptr[qi]),int(ptr[qi+1]); X=z["X"][a:b]; y=z["labels"][a:b]; rows=z["candidate_rows"][a:b]
            pos=np.flatnonzero(y>0); neg=np.flatnonzero(y==0)
            if not len(pos): continue
            hard=neg[np.lexsort((rows[neg],X[neg,hard_col]))][:128]
            rem=np.setdiff1d(neg,hard,assume_unique=False)
            rng=np.random.default_rng(base._stable_seed(f"bime-r2e-clip|{q}"))
            rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64)
            keep=np.concatenate([pos,hard,rnd])
            xs.append(X[keep]); ys.append(y[keep].astype(np.float32)); groups.append(len(keep))
    return np.concatenate(xs),np.concatenate(ys),groups


def _train(caches:list[dict],seed:int):
    X,y,groups=_training_matrix(caches)
    dm=xgb.DMatrix(X,label=y); dm.set_group(groups)
    cfg=FIXED_CONFIG
    params={"objective":"rank:ndcg","eval_metric":"ndcg@10","tree_method":"hist","device":"cuda" if torch.cuda.is_available() else "cpu","max_depth":cfg.max_depth,"eta":cfg.learning_rate,"min_child_weight":cfg.min_child_weight,"lambda":cfg.reg_lambda,"subsample":0.9,"colsample_bytree":0.9,"lambdarank_pair_method":"topk","lambdarank_num_pair_per_sample":cfg.lambdarank_pairs,"seed":seed,"verbosity":0}
    return xgb.train(params,dm,num_boost_round=cfg.rounds)


def _evaluate(model:xgb.Booster,cache:dict,fold:int)->pd.DataFrame:
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
    caches={f:_load_cache(f) for f in FOLDS}
    frames=[]
    for holdout in FOLDS:
        model=_train([caches[f] for f in FOLDS if f!=holdout],base._stable_seed(f"bime-r2e-clip|fold{holdout}"))
        q=_evaluate(model,caches[holdout],holdout); frames.append(q)
        print(f"crossfit fold={holdout} done",flush=True)
    new=pd.concat(frames,ignore_index=True)
    old=pd.read_csv(OLD_OOF,dtype={"query_id":str})
    old=old[old.config_id.eq("cfg_07_392fe119")].copy()
    if len(old)!=len(new): raise RuntimeError((len(old),len(new)))
    old_metrics=base._metric_map(old); new_metrics=base._metric_map(new)
    keys=base.ALL_GATE_METRICS
    delta={k:new_metrics[k]-old_metrics[k] for k in keys}
    fold_delta={}
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
    payload={"status":"admitted" if admitted else "not_admitted","comparison":"same folds, same fixed ranker hyperparameters; only CLIPZyme expert/features/candidates added","old_selected_config":"cfg_07_392fe119","fixed_config":FIXED_CONFIG.__dict__,"old":old_metrics,"new":new_metrics,"delta":delta,"fold_delta":fold_delta,"primary_geomean_ratio":gm,"fold_safe":fold_safe,"pooled_safe":pooled_safe,"material":material,"admitted":admitted,"external_metrics_used":False,"admission_rule":"internal clean-dev only; fold MRR/MAP >= -0.005; pooled MRR/MAP >= -0.002 and Hit20/50 >= -0.005; primary geomean >=1.005; material gain in MRR/MAP >=.003 or Hit20/50 >=.01"}
    (OUT/"development_result.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


def fit_final() -> None:
    result=json.loads((OUT/"development_result.json").read_text())
    if not result.get("admitted"): raise RuntimeError("CLIPZyme expert did not pass internal admission gate")
    caches=[_load_cache(f) for f in FOLDS]; model=_train(caches,base._stable_seed("bime-r2e-clip-final"))
    out=OUT/"selected"; out.mkdir(parents=True,exist_ok=True); model.save_model(out/"ranker.json")
    cfg={"feature_names":FEATURE_NAMES,"fixed_config":FIXED_CONFIG.__dict__,"training_folds":list(FOLDS),"external_metrics_used":False,"clip_protein_manifest_sha256":_sha256(CLIP_PROTEIN_ROOT/"manifest.json"),"clip_reaction_manifest_sha256":_sha256(CLIP_REACTION_ROOT/"manifest.json")}
    cfg["ranker_sha256"]=_sha256(out/"ranker.json"); (out/"config.json").write_text(json.dumps(cfg,indent=2)+"\n"); print(json.dumps(cfg,indent=2))


def main()->None:
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","crossfit","fit-final"]); ap.add_argument("--fold",type=int,choices=FOLDS); ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=ap.parse_args()
    if a.stage=="prepare":
        if a.fold is None: raise ValueError("--fold required")
        prepare_fold(a.fold,a.device)
    elif a.stage=="crossfit": crossfit()
    else: fit_final()


if __name__=="__main__":
    main()
