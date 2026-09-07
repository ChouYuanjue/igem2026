from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.bime_rank_r2e_runtime import fuse_bime_r2e_scores
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.rank_open_world import (
    load_feature_schema,
    load_models,
    load_protein_library,
    load_registered_reaction_feature_library,
    tied_rank_percentile,
)
from projects.active.terpene_screening.run_bime_r2e_seed_context_v1 import (
    context_features,
    choose_seed_rows,
    masked_order,
)

PRIMARY = ROOT / 'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'
SECONDARY = ROOT / 'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1'
PP = ROOT / 'data/catalyst_candidate_universes/general_merged/proteins'
SP = ROOT / 'data/external/enzgfm_current/general_merged_650m_mean_v1'
RF = ROOT / 'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
BASE_BUNDLE = ROOT / 'results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1'
BASE_SHA = '86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'
STRUCT_BUNDLE = ROOT / 'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected'
STRUCT_SHA = '383dca7a176c47f3b0e431bb4b33492ae2f954c417f37f873969887b2d6f03e5'
P_ASSET = ROOT / 'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1'
R_ASSET = ROOT / 'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
SEED_BUNDLE = ROOT / 'results/bime_rank_unified_v1/r2e_seed_context_v1/selected'
PAIR = ROOT / 'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv'
QFILE = ROOT / 'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt'
DIFF = ROOT / 'results/rhea128_to141_external_v2/posthoc_difficulty/rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv'
OUT = ROOT / 'results/bime_rank_unified_v1/r2e_seed_context_retention_v1'
LEGACY_DIRECT_WEIGHT = 0.99
POOL_K = 100
PREFIX_K = 100
BOOT = 50000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_map(df: pd.DataFrame) -> dict[str, float]:
    cols = {
        'mrr': 'reciprocal_rank', 'map': 'average_precision', 'macro_roc_auc': 'roc_auc',
        'ndcg_at_10': 'ndcg_at_10', 'hit_at_10': 'hit_at_10', 'hit_at_20': 'hit_at_20', 'hit_at_50': 'hit_at_50',
    }
    return {k: float(df[v].mean()) for k, v in cols.items()} | {
        'median_best_positive_rank': float(df.best_positive_rank.median())
    }


def hidden_ranks(order: np.ndarray, hidden: np.ndarray, n: int) -> np.ndarray:
    inv = np.full(n, len(order) + 1, dtype=np.int32)
    inv[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return inv[hidden]


def context_order(
    ranker: xgb.Booster,
    base_order_full: np.ndarray,
    seed_scores: np.ndarray,
    seed_row: int,
    lex: np.ndarray,
) -> np.ndarray:
    base_order = base_order_full[base_order_full != seed_row]
    base_inv = np.full(len(seed_scores), len(base_order)+1, dtype=np.int32)
    base_inv[base_order] = np.arange(1, len(base_order)+1, dtype=np.int32)
    seed_order, seed_inv = masked_order(seed_scores, lex, seed_row)
    union = np.unique(np.concatenate([base_order[:POOL_K], seed_order[:POOL_K]])).astype(np.int32)
    union = union[union != seed_row]
    X = context_features(union, base_inv, seed_scores, seed_inv, len(seed_scores)-1)
    pred = ranker.predict(xgb.DMatrix(X))
    take = np.lexsort((lex[union], -pred))[:min(PREFIX_K, len(union))]
    selected = union[take]
    chosen = np.zeros(len(seed_scores), dtype=bool); chosen[selected] = True
    tail = base_order[~chosen[base_order]]
    return np.concatenate([selected, tail]).astype(np.int32, copy=False)


def paired_bootstrap(new: pd.DataFrame, base: pd.DataFrame, seed: int = 20260905) -> dict[str, object]:
    merged = new.merge(base, on='trial_id', suffixes=('_new', '_base'), validate='one_to_one')
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ['reciprocal_rank','average_precision','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']:
        d = merged[f'{metric}_new'].to_numpy(float) - merged[f'{metric}_base'].to_numpy(float)
        vals = np.empty(BOOT, dtype=np.float32)
        for st in range(0, BOOT, 1000):
            m = min(1000, BOOT-st)
            idx = rng.integers(0, len(d), size=(m, len(d)))
            vals[st:st+m] = d[idx].mean(axis=1)
        out[metric] = {
            'delta': float(d.mean()),
            'ci95': [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SEED_BUNDLE/'config.json').read_text())
    if sha(SEED_BUNDLE/'ranker.json') != cfg['ranker_sha256']:
        raise RuntimeError('frozen seed-context ranker hash mismatch')
    if cfg.get('external_metrics_used') is not False:
        raise RuntimeError('seed ranker was not selected internally only')
    ranker = xgb.Booster(); ranker.load_model(SEED_BUNDLE/'ranker.json')
    pcfg = json.loads((STRUCT_BUNDLE/'config.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pf,pids = load_protein_library(PP); sf,sids = load_protein_library(SP)
    assert pids == sids and len(pids) == 185918
    # Production ESM-C features are normalized and are exactly the seed-similarity representation.
    norms = np.linalg.norm(pf, axis=1)
    if float(np.max(np.abs(norms - 1.0))) > 2e-5:
        raise RuntimeError('production seed protein features are not normalized as expected')
    ps=load_feature_schema(PRIMARY); ss=load_feature_schema(SECONDARY)
    rf,rids=load_registered_reaction_feature_library(RF,ps); rf2,rids2=load_registered_reaction_feature_library(RF,ss)
    assert rids==rids2 and np.array_equal(rf,rf2)
    pm=load_models(PRIMARY/'models','production',device); sm=load_models(SECONDARY/'models','production',device)
    assert len(pm)==len(sm)==1
    pe=encode_chunks(pm[0],pf,kind='protein',device=device,chunk_size=8192)
    se=encode_chunks(sm[0],sf,kind='protein',device=device,chunk_size=8192)
    pre=encode_chunks(pm[0],rf,kind='reaction',device=device,chunk_size=8192)
    sre=encode_chunks(sm[0],rf,kind='reaction',device=device,chunk_size=8192)

    qids=[x.strip() for x in open(QFILE) if x.strip()]
    ridx={r:i for i,r in enumerate(rids)}; pidx={p:i for i,p in enumerate(pids)}
    pairs=pd.read_csv(PAIR,dtype=str).fillna(''); pos=pairs.groupby('reaction_id').protein_id.apply(lambda x:sorted(set(map(str,x)))).to_dict()
    diff=pd.read_csv(DIFF,dtype={'reaction_id':str}); sims=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)))
    lex=np.empty(len(pids),dtype=np.int32); o=np.argsort(np.asarray(pids),kind='stable'); lex[o]=np.arange(len(pids),dtype=np.int32)
    records={'context':[],'legacy':[],'zero_shot':[],'seed_only':[]}; audits=[]
    eligible=0
    for st in range(0,len(qids),16):
        qs=qids[st:st+16]; qr=torch.tensor([ridx[q] for q in qs],dtype=torch.long,device=device)
        with torch.no_grad():
            psc=(pre[qr]@pe.T).float().cpu().numpy(); ssc=(sre[qr]@se.T).float().cpu().numpy()
        for j,q in enumerate(qs):
            prows=np.asarray([pidx[p] for p in pos[q]],dtype=np.int32)
            seeds=choose_seed_rows(q,prows)
            if not seeds: continue
            eligible += 1
            fused=fuse_bime_r2e_scores(
                psc[j],ssc[j],pids,reaction_id=q,similarity=float(sims[q]),threshold=0.9,
                base_ranker_bundle=BASE_BUNDLE,base_ranker_sha256=BASE_SHA,
                structural_ranker_bundle=STRUCT_BUNDLE,structural_ranker_sha256=STRUCT_SHA,
                clip_protein_asset=P_ASSET,clip_reaction_asset=R_ASSET,
                clip_protein_manifest_sha256=pcfg['clip_protein_manifest_sha256'],clip_reaction_manifest_sha256=pcfg['clip_reaction_manifest_sha256'],
                device=str(device),expected_pool_k=100,expected_prefix_k=100,
            )
            if not fused.structure_expert_applied: raise RuntimeError(f'structural base unexpectedly unavailable for {q}')
            direct = ssc[j] if float(sims[q]) < 0.9 else psc[j]
            for rep,seed_row in enumerate(seeds):
                hidden=prows[prows!=seed_row]
                seed_scores=(pf @ pf[seed_row]).astype(np.float32,copy=False)
                seed_order,_=masked_order(seed_scores,lex,seed_row)
                base_order=fused.full_order[fused.full_order!=seed_row]
                legacy_score=(LEGACY_DIRECT_WEIGHT*tied_rank_percentile(direct,pids)+(1-LEGACY_DIRECT_WEIGHT)*tied_rank_percentile(seed_scores,pids))
                legacy_order,_=masked_order(legacy_score,lex,seed_row)
                ctx_order=context_order(ranker,fused.full_order,seed_scores,seed_row,lex)
                trial=f'{q}|seed={pids[seed_row]}|rep={rep}'
                for name,order in [('context',ctx_order),('legacy',legacy_order),('zero_shot',base_order),('seed_only',seed_order)]:
                    m=evaluate_full_candidate_ranks(hidden_ranks(order,hidden,len(pids)),len(pids)-1)
                    records[name].append({'trial_id':trial,'query_id':q,'seed_id':pids[seed_row],**m})
                audits.append({'trial_id':trial,'query_id':q,'seed_id':pids[seed_row],'hidden_positives':len(hidden),'similarity':float(sims[q]),'fallback_secondary':float(sims[q])<0.9})
        print('retention',min(st+16,len(qids)),'/',len(qids),flush=True)

    frames={k:pd.DataFrame(v) for k,v in records.items()}
    for k,v in frames.items(): v.to_csv(OUT/f'{k}_query_metrics.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'audit.csv',index=False)
    metrics={k:metric_map(v) for k,v in frames.items()}
    delta_legacy={k:metrics['context'][k]-metrics['legacy'][k] for k in metrics['context'] if k!='median_best_positive_rank'}
    delta_zero={k:metrics['context'][k]-metrics['zero_shot'][k] for k in metrics['context'] if k!='median_best_positive_rank'}
    boots={
        'context_minus_legacy': paired_bootstrap(frames['context'],frames['legacy'],20260905),
        'context_minus_zero_shot': paired_bootstrap(frames['context'],frames['zero_shot'],20260906),
    }
    # External data is veto-only. No parameter is changed based on these values.
    retain = bool(
        delta_legacy['mrr'] >= -0.002 and delta_legacy['map'] >= -0.002 and
        delta_legacy['hit_at_20'] >= -0.01 and delta_legacy['hit_at_50'] >= -0.01 and
        (delta_legacy['mrr'] > 0 or delta_legacy['map'] > 0 or delta_legacy['hit_at_20'] > 0 or delta_legacy['hit_at_50'] > 0)
    )
    summary={
        'status':'retention_passed' if retain else 'retention_failed',
        'protocol':'Frozen seed-context ranker evaluated once after internal admission on pre-existing Rhea128->141 strict mutual-train-cold labels; one positive seed is masked and remaining positives are hidden targets in full 185918-protein production universe',
        'external_metrics_used_for_selection':False,'external_metrics_used_for_retuning':False,'retention_is_veto_only':True,
        'eligible_queries':eligible,'trials':len(frames['context']),'candidate_count':len(pids),'positive_pairs_source':str(PAIR),
        'seed_ranker_sha256':cfg['ranker_sha256'],'structural_bime_ranker_sha256':STRUCT_SHA,
        'metrics':metrics,'delta_context_vs_legacy':delta_legacy,'delta_context_vs_zero_shot':delta_zero,
        'paired_bootstrap_50000':boots,'retention_gate_passed':retain,
    }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not retain: raise SystemExit(2)

if __name__=='__main__': main()
