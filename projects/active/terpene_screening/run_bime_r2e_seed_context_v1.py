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
from projects.active.terpene_screening import run_bime_r2e_clipzyme_expert_v1 as clipmod
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.rank_open_world import load_protein_library, tied_rank_percentile

OUT = ROOT / "results/bime_rank_unified_v1/r2e_seed_context_v1"
PROTEIN_ROOT = ROOT / "data/catalyst_candidate_universes/general_merged/proteins"
FOLDS = (0, 1, 2)
POOL_K = 100
PREFIX_K = 100
SEED_REPS = 3
LEGACY_DIRECT_WEIGHT = 0.99
FEATURE_NAMES = [
    "base_log_rank_fraction", "base_reciprocal_rank", "base_top10", "base_top50", "base_top100",
    "seed_raw_score", "seed_query_zscore", "seed_log_rank_fraction", "seed_reciprocal_rank",
    "seed_top10", "seed_top50", "seed_top100", "best_log_rank", "rank_gap_seed_minus_base",
    "both_top10", "both_top50", "both_top100",
]
FIXED_CONFIG = {
    "pool_k": POOL_K, "prefix_k": PREFIX_K, "max_depth": 2, "eta": 0.12,
    "min_child_weight": 5.0, "reg_lambda": 1.0, "rounds": 80, "lambdarank_pairs": 20,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def masked_order(scores: np.ndarray, lex: np.ndarray, masked_row: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((lex, -scores)).astype(np.int32)
    order = order[order != int(masked_row)]
    inv = np.full(len(scores), len(order) + 1, dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return order, inv


def choose_seed_rows(q: str, positives: np.ndarray) -> list[int]:
    values = list(map(int, positives))
    if len(values) < 2:
        return []
    out: list[int] = []
    for rep in range(min(SEED_REPS, len(values))):
        rng = np.random.default_rng(base._stable_seed(f"bime-r2e-seed-context|{q}|{rep}"))
        candidates = [v for v in values if v not in out]
        if not candidates:
            break
        out.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    return out


def load_structural_model() -> xgb.Booster:
    p = clipmod.OUT / "selected/ranker.json"
    cfg = json.loads((clipmod.OUT / "selected/config.json").read_text())
    if sha(p) != cfg["ranker_sha256"]:
        raise RuntimeError("frozen structural BiME ranker hash mismatch")
    model = xgb.Booster(); model.load_model(p)
    return model


def structural_order(
    model: xgb.Booster,
    *,
    s0: np.ndarray, s1: np.ndarray, inv0: np.ndarray, inv1: np.ndarray,
    use_secondary: bool, similarity: float, lex: np.ndarray,
    clip_scores: np.ndarray | None, clip_inv: np.ndarray | None,
    clip_lookup: np.ndarray, query_supported: bool, clip_top: np.ndarray,
    masked_row: int,
) -> np.ndarray:
    o0 = np.argsort(inv0, kind="stable").astype(np.int32)
    o1 = np.argsort(inv1, kind="stable").astype(np.int32)
    union = np.unique(np.concatenate([o0[:POOL_K], o1[:POOL_K], clip_top])).astype(np.int32)
    union = union[union != masked_row]
    bx = base._build_features(s0, s1, union, inv0, inv1, use_secondary, similarity)
    x = clipmod._clip_features(bx, union, clip_scores, clip_inv, clip_lookup, query_supported, len(lex))
    pred = model.predict(xgb.DMatrix(x))
    local = np.lexsort((lex[union], -pred))[:min(PREFIX_K, len(union))]
    selected = union[local]
    fallback = o1 if use_secondary else o0
    chosen = set(map(int, selected))
    rest = np.asarray([r for r in fallback if int(r) not in chosen and int(r) != masked_row], dtype=np.int32)
    return np.concatenate([selected, rest]).astype(np.int32, copy=False)


def context_features(rows: np.ndarray, base_inv: np.ndarray, seed_scores: np.ndarray, seed_inv: np.ndarray, n: int) -> np.ndarray:
    br = base_inv[rows].astype(np.float64)
    sr = seed_inv[rows].astype(np.float64)
    denom = math.log1p(n)
    blog = (np.log1p(br) / denom).astype(np.float32)
    slog = (np.log1p(sr) / denom).astype(np.float32)
    brr = (1.0 / br).astype(np.float32)
    srr = (1.0 / sr).astype(np.float32)
    mean = float(seed_scores.mean()); std = max(float(seed_scores.std()), 1e-6)
    sz = ((seed_scores[rows] - mean) / std).astype(np.float32)
    b10 = (br <= 10).astype(np.float32); b50 = (br <= 50).astype(np.float32); b100 = (br <= 100).astype(np.float32)
    s10 = (sr <= 10).astype(np.float32); s50 = (sr <= 50).astype(np.float32); s100 = (sr <= 100).astype(np.float32)
    x = np.column_stack([
        blog, brr, b10, b50, b100,
        seed_scores[rows].astype(np.float32), sz, slog, srr, s10, s50, s100,
        np.minimum(blog, slog), (slog - blog).astype(np.float32),
        b10 * s10, b50 * s50, b100 * s100,
    ]).astype(np.float32)
    assert x.shape[1] == len(FEATURE_NAMES)
    return x


def ranks_of_hidden(order: np.ndarray, hidden_rows: np.ndarray) -> np.ndarray:
    inv = np.empty(int(order.max()) + 1 if len(order) else 0, dtype=np.int64)
    # candidate IDs are dense 0..N-1; allocate with hidden maximum safety.
    n = max(int(order.max(initial=0)), int(hidden_rows.max(initial=0))) + 1
    inv = np.full(n, len(order) + 1, dtype=np.int64)
    inv[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return inv[hidden_rows]


def prepare_fold(fold: int, device_name: str) -> None:
    device = torch.device(device_name)
    ids, _, reaction_ids, pe0, pe1, re0, re1 = base._load_fold_embeddings(fold, device)
    raw_features, raw_ids = load_protein_library(PROTEIN_ROOT)
    if raw_ids != ids:
        raise RuntimeError("general protein feature order drifted")
    raw_features = np.asarray(raw_features, dtype=np.float32)
    raw_features_t = torch.as_tensor(raw_features, dtype=torch.float32, device=device)
    candidate_index = {p: i for i, p in enumerate(ids)}
    reaction_index = {r: i for i, r in enumerate(reaction_ids)}
    lex = base._lexical_rank(ids)
    clip_pt, clip_candidate_rows, clip_lookup, clip_rmat, clip_ridx = clipmod._load_clip_assets(ids, device)
    clip_lex = lex[clip_candidate_rows]
    structural_model = load_structural_model()

    dev_pairs = pd.read_csv(base.DEV_ROOT / "baseline_base" / f"fold{fold}" / "dev_pairs.csv", dtype=str).fillna("")
    positives = dev_pairs.groupby("reaction_id")["protein_id"].apply(lambda x: sorted(set(map(str, x)))).to_dict()
    query_ids = sorted(q for q, ps in positives.items() if len(ps) >= 2)
    cell = f"clean2023_internal_double_cold_salted_comprehensive_enzgfm_center_top1_v1_dev_20260901_fold{fold}"
    difficulty = pd.read_csv(base.DEV_ROOT / "difficulty" / cell / "reaction_slices.csv", dtype={"reaction_id": str})
    sim_map = dict(zip(difficulty["reaction_id"].astype(str), difficulty["max_train_drfp_tanimoto"].astype(float)))

    Xs=[]; ys=[]; rows_all=[]; fallback_all=[]; ptr=[0]; pos_all=[]; posfb_all=[]; pptr=[0]; trials=[]; audits=[]
    legacy_metrics=[]; base_metrics=[]; seed_metrics=[]
    for qi, q in enumerate(query_ids):
        rr = reaction_index[q]
        with torch.no_grad():
            s0 = (re0[rr] @ pe0.T).float().cpu().numpy().astype(np.float32)
            s1 = (re1[rr] @ pe1.T).float().cpu().numpy().astype(np.float32)
        sim = float(sim_map[q]); use_secondary = sim < base.ROUTER_THRESHOLD
        cscore = None; c_inv_raw = None; c_top_raw = np.empty(0, dtype=np.int32); query_supported = q in clip_ridx
        if query_supported:
            qt = torch.as_tensor(np.asarray(clip_rmat[clip_ridx[q]], dtype=np.float32), device=device)
            with torch.no_grad(): cscore = (clip_pt @ qt).float().cpu().numpy().astype(np.float32)
        # Production known-mask semantics are score/rank first, mask seed only at output.
        # Therefore the frozen zero-shot BiME order is computed exactly once per query.
        _o0_full, inv0_full = masked_order(s0, lex, -1)
        _o1_full, inv1_full = masked_order(s1, lex, -1)
        clip_inv_full = None; clip_top_full = np.empty(0, dtype=np.int32)
        if query_supported and cscore is not None:
            cord = np.lexsort((clip_lex, -cscore)).astype(np.int32)
            clip_inv_full = np.full(len(cscore), len(cord)+1, dtype=np.int32)
            clip_inv_full[cord] = np.arange(1, len(cord)+1, dtype=np.int32)
            clip_top_full = clip_candidate_rows[cord[:POOL_K]]
        base_order_full = structural_order(
            structural_model, s0=s0, s1=s1, inv0=inv0_full, inv1=inv1_full, use_secondary=use_secondary,
            similarity=sim, lex=lex, clip_scores=cscore, clip_inv=clip_inv_full, clip_lookup=clip_lookup,
            query_supported=query_supported, clip_top=clip_top_full, masked_row=-1,
        )
        prows = np.asarray([candidate_index[p] for p in positives[q]], dtype=np.int32)
        for rep, seed_row in enumerate(choose_seed_rows(q, prows)):
            hidden = prows[prows != seed_row]
            base_order = base_order_full[base_order_full != seed_row]
            base_inv = np.full(len(ids), len(base_order)+1, dtype=np.int32); base_inv[base_order] = np.arange(1, len(base_order)+1, dtype=np.int32)
            with torch.no_grad():
                seed_scores = (raw_features_t @ raw_features_t[seed_row]).float().cpu().numpy().astype(np.float32, copy=False)
            seed_order, seed_inv = masked_order(seed_scores, lex, seed_row)
            union = np.unique(np.concatenate([base_order[:POOL_K], seed_order[:POOL_K]])).astype(np.int32)
            union = union[union != seed_row]
            x = context_features(union, base_inv, seed_scores, seed_inv, len(ids)-1)
            y = np.isin(union, hidden).astype(np.uint8)
            Xs.append(x); ys.append(y); rows_all.append(union); fallback_all.append(base_inv[union]); ptr.append(ptr[-1]+len(union))
            pos_all.append(hidden); posfb_all.append(base_inv[hidden]); pptr.append(pptr[-1]+len(hidden))
            trial_id=f"{q}|seed={ids[seed_row]}|rep={rep}"; trials.append(trial_id)
            # Comparator metrics: current legacy 0.99 direct-rank hybrid, zero-shot BiME, and seed-only.
            direct = s1 if use_secondary else s0
            legacy_score = LEGACY_DIRECT_WEIGHT*tied_rank_percentile(direct, ids) + (1-LEGACY_DIRECT_WEIGHT)*tied_rank_percentile(seed_scores, ids)
            lo,_ = masked_order(legacy_score, lex, seed_row)
            for name,order,store in [('legacy',lo,legacy_metrics),('base',base_order,base_metrics),('seed',seed_order,seed_metrics)]:
                m=evaluate_full_candidate_ranks(ranks_of_hidden(order,hidden),len(ids)-1); store.append({'fold':fold,'trial_id':trial_id,'query_id':q,'seed_id':ids[seed_row],**m})
            audits.append({'fold':fold,'trial_id':trial_id,'query_id':q,'seed_id':ids[seed_row],'hidden_positives':len(hidden),'union_size':len(union),'hidden_in_union':int(y.sum()),'clip_query_supported':query_supported,'use_secondary':use_secondary,'similarity':sim})
        if (qi+1)%50==0: print(f"prepare seed-context fold={fold} {qi+1}/{len(query_ids)}",flush=True)

    out=OUT/'prepared'/f'fold{fold}'; out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'cache.npz',X=np.concatenate(Xs),labels=np.concatenate(ys),candidate_rows=np.concatenate(rows_all),fallback_ranks=np.concatenate(fallback_all),query_ptr=np.asarray(ptr,dtype=np.int64),positive_rows=np.concatenate(pos_all),positive_fallback_ranks=np.concatenate(posfb_all),pos_ptr=np.asarray(pptr,dtype=np.int64),lexical_rank=lex)
    pd.DataFrame({'trial_id':trials}).to_csv(out/'trials.csv',index=False); pd.DataFrame(audits).to_csv(out/'audit.csv',index=False)
    pd.DataFrame(legacy_metrics).to_csv(out/'legacy_query_metrics.csv',index=False); pd.DataFrame(base_metrics).to_csv(out/'base_query_metrics.csv',index=False); pd.DataFrame(seed_metrics).to_csv(out/'seed_query_metrics.csv',index=False)
    summary={'fold':fold,'eligible_queries':len(query_ids),'trials':len(trials),'candidate_count':len(ids),'mean_union_size':float(pd.DataFrame(audits).union_size.mean()),'queries_hidden_in_union_fraction':float((pd.DataFrame(audits).hidden_in_union>0).mean()),'external_metrics_used':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))


def load_cache(fold:int):
    p=OUT/'prepared'/f'fold{fold}'
    with np.load(p/'cache.npz') as z: data={k:z[k] for k in z.files}
    return {'z':data,'trials':pd.read_csv(p/'trials.csv',dtype=str).trial_id.astype(str).tolist()}


def training_matrix(caches:list[dict]):
    xs=[];ys=[];groups=[]
    for c in caches:
        z=c['z'];ptr=z['query_ptr']
        for i,t in enumerate(c['trials']):
            a,b=int(ptr[i]),int(ptr[i+1]);X=z['X'][a:b];y=z['labels'][a:b];rows=z['candidate_rows'][a:b]
            pos=np.flatnonzero(y>0);neg=np.flatnonzero(y==0)
            if not len(pos): continue
            hardness=np.minimum(X[neg,0],X[neg,7])
            hard=neg[np.lexsort((rows[neg],hardness))][:128]
            rem=np.setdiff1d(neg,hard,assume_unique=False);rng=np.random.default_rng(base._stable_seed('seedctx|'+t))
            rnd=rng.choice(rem,size=min(32,len(rem)),replace=False) if len(rem) else np.empty(0,dtype=np.int64)
            keep=np.concatenate([pos,hard,rnd]);xs.append(X[keep]);ys.append(y[keep].astype(np.float32));groups.append(len(keep))
    return np.concatenate(xs),np.concatenate(ys),groups


def train(caches:list[dict],seed:int):
    X,y,g=training_matrix(caches);dm=xgb.DMatrix(X,label=y);dm.set_group(g)
    p={'objective':'rank:ndcg','eval_metric':'ndcg@10','tree_method':'hist','device':'cuda' if torch.cuda.is_available() else 'cpu','max_depth':2,'eta':0.12,'min_child_weight':5.0,'lambda':1.0,'subsample':0.9,'colsample_bytree':0.9,'lambdarank_pair_method':'topk','lambdarank_num_pair_per_sample':20,'seed':seed,'verbosity':0}
    return xgb.train(p,dm,num_boost_round=80)


def evaluate(model,cache,fold):
    z=cache['z'];ptr=z['query_ptr'];pp=z['pos_ptr'];pred=model.predict(xgb.DMatrix(z['X']));lex=z['lexical_rank'];out=[]
    for i,t in enumerate(cache['trials']):
        a,b=int(ptr[i]),int(ptr[i+1]);rows=z['candidate_rows'][a:b];local=pred[a:b];fb=z['fallback_ranks'][a:b]
        take=np.lexsort((lex[rows],-local))[:min(PREFIX_K,len(rows))];selected=rows[take];selected_fb=fb[take];sp={int(r):j+1 for j,r in enumerate(selected)}
        pa,pb=int(pp[i]),int(pp[i+1]);prows=z['positive_rows'][pa:pb];pfb=z['positive_fallback_ranks'][pa:pb];ranks=[]
        for r,fr in zip(prows,pfb,strict=True):
            r=int(r);fr=int(fr);ranks.append(sp[r] if r in sp else len(selected)+fr-int(np.count_nonzero(selected_fb<fr)))
        q=t.split('|seed=',1)[0];sid=t.split('|seed=',1)[1].split('|rep=',1)[0]
        out.append({'fold':fold,'trial_id':t,'query_id':q,'seed_id':sid,**evaluate_full_candidate_ranks(np.asarray(ranks),len(lex)-1)})
    return pd.DataFrame(out)


def metric_map(df): return base._metric_map(df)


def crossfit():
    caches={f:load_cache(f) for f in FOLDS};frames=[]
    for h in FOLDS:
        m=train([caches[f] for f in FOLDS if f!=h],base._stable_seed(f'seedctx-fold{h}'));frames.append(evaluate(m,caches[h],h));print('crossfit',h,flush=True)
    new=pd.concat(frames,ignore_index=True);new.to_csv(OUT/'development_oof_query_metrics.csv',index=False)
    legacy=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'legacy_query_metrics.csv') for f in FOLDS],ignore_index=True)
    zero=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'base_query_metrics.csv') for f in FOLDS],ignore_index=True)
    seed=pd.concat([pd.read_csv(OUT/'prepared'/f'fold{f}'/'seed_query_metrics.csv') for f in FOLDS],ignore_index=True)
    old=metric_map(legacy);base_m=metric_map(zero);seed_m=metric_map(seed);new_m=metric_map(new)
    keys=base.ALL_GATE_METRICS;dlegacy={k:new_m[k]-old[k] for k in keys};dbase={k:new_m[k]-base_m[k] for k in keys}
    folds={}
    for f in FOLDS:
        nm=metric_map(new[new.fold.eq(f)]);lm=metric_map(legacy[legacy.fold.eq(f)]);folds[str(f)]={k:nm[k]-lm[k] for k in keys}
    primary=('mrr','map','hit_at_20','hit_at_50');gm=float(math.exp(sum(math.log(max(new_m[k]/old[k],1e-12)) for k in primary)/len(primary)))
    fold_safe=all(v['mrr']>=-0.005 and v['map']>=-0.005 for v in folds.values()); pooled_safe=dlegacy['mrr']>=-0.002 and dlegacy['map']>=-0.002 and dlegacy['hit_at_20']>=-0.005 and dlegacy['hit_at_50']>=-0.005
    material=gm>=1.01 and (dlegacy['mrr']>=0.003 or dlegacy['map']>=0.003 or dlegacy['hit_at_20']>=0.01 or dlegacy['hit_at_50']>=0.01);admit=bool(fold_safe and pooled_safe and material)
    payload={'status':'admitted_internal' if admit else 'not_admitted','scenario':'one known positive enzyme; seed itself masked; hidden positives evaluated in 185918-protein general universe','trials':len(new),'eligible_queries':int(new.query_id.nunique()),'seed_reps':SEED_REPS,'legacy_current_fewshot':old,'zero_shot_bime':base_m,'seed_only':seed_m,'context_bime':new_m,'delta_vs_legacy':dlegacy,'delta_vs_zero_shot':dbase,'fold_delta_vs_legacy':folds,'primary_geomean_ratio_vs_legacy':gm,'fold_safe':fold_safe,'pooled_safe':pooled_safe,'material':material,'admitted':admit,'external_metrics_used':False,'next_if_admitted':'fit final on all three internal folds, then freeze before any strict temporal retention'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'development_result.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))


def fit_final():
    r=json.loads((OUT/'development_result.json').read_text());assert r['admitted']
    m=train([load_cache(f) for f in FOLDS],base._stable_seed('seedctx-final'));o=OUT/'selected';o.mkdir(parents=True,exist_ok=True);m.save_model(o/'ranker.json')
    cfg={'feature_names':FEATURE_NAMES,'fixed_config':FIXED_CONFIG,'training_folds':list(FOLDS),'seed_reps':SEED_REPS,'external_metrics_used':False,'frozen_structural_bime_ranker_sha256':sha(clipmod.OUT/'selected/ranker.json'),'protein_feature_source':str(PROTEIN_ROOT),'protein_feature_manifest_sha256':sha(PROTEIN_ROOT/'manifest.json') if (PROTEIN_ROOT/'manifest.json').exists() else None};cfg['ranker_sha256']=sha(o/'ranker.json');(o/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(cfg,indent=2))


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','crossfit','fit-final']);ap.add_argument('--fold',type=int,choices=FOLDS);ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=ap.parse_args()
    if a.stage=='prepare': assert a.fold is not None;prepare_fold(a.fold,a.device)
    elif a.stage=='crossfit':crossfit()
    else:fit_final()
if __name__=='__main__':main()
