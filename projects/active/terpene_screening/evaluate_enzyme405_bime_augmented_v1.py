from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.terpene_screening.evaluate_enzyme405_fixed_support import prepare_fixed_support
from projects.active.terpene_screening.evaluate_enzymecage_405_cleanroom import load_query_reaction_library, official_query_id
from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import evaluate_scores
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.fair_benchmark import sha256_file
from projects.active.terpene_screening.rank_open_world import (
    exact_max_train_binary_drfp_tanimoto,
    load_feature_schema,
    load_models,
    load_protein_library,
)
from projects.active.terpene_screening.r2e_lambdarank_runtime import fuse_r2e_scores

PAIRS=ROOT/'data/external/enzymecage_current/enzyme405_max_common_support_v1/complete_candidate_226.csv'
OLD_AUDIT=ROOT/'results/enzyme405_current_mainline_v1/complete226_fixed_support/protein_feature_audit.csv'
PRIMARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'
SECONDARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1'
BASE_ESMC=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
BASE_ENZGFM=ROOT/'data/external/enzgfm_current/general_merged_650m_mean_v1'
EXT_ESMC=ROOT/'data/external/enzymecage_current/catalyst_features/new_protein_esmc'
EXT_ENZGFM=ROOT/'results/bime_rank_unified_v1/enzyme405_extension_v1/enzgfm'
EXT_SCOPE=ROOT/'results/bime_rank_unified_v1/enzyme405_extension_v1/protein_ids.csv'
QUERY_FEATURES=ROOT/'data/external/enzymecage_current/catalyst_features/query_reaction_rdkitplus_center_v1'
SIM_FEATURES=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'
TRAINING_PAIRS=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
REGISTERED_REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reactions.csv'
RANKER=ROOT/'results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1'
RANKER_SHA='86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'
OUT=ROOT/'results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1'
CAGE_ROOT=ROOT/'results/enzymecage_local_maxsupport_v1'
BOOT=50000


def file_sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def load_rows(root:Path, ids:list[str])->np.ndarray:
    x,all_ids=load_protein_library(root); idx={v:i for i,v in enumerate(all_ids)}; missing=[v for v in ids if v not in idx]
    if missing: raise RuntimeError(f'{root} misses extension IDs: {missing[:10]}')
    return np.asarray(x[np.asarray([idx[v] for v in ids],dtype=np.int64)],dtype=np.float32)

def build_augmented():
    ep,eids=load_protein_library(BASE_ESMC); zp,zids=load_protein_library(BASE_ENZGFM)
    if eids!=zids or len(eids)!=185918: raise RuntimeError('base BiME candidate universes drifted')
    ext=pd.read_csv(EXT_SCOPE,dtype=str).fillna('').protein_id.astype(str).tolist(); ext=sorted(dict.fromkeys(ext))
    if len(ext)!=252 or set(ext)&set(eids): raise RuntimeError('extension scope drifted or overlaps base')
    ex=load_rows(EXT_ESMC,ext); zx=load_rows(EXT_ENZGFM,ext)
    return np.concatenate([np.asarray(ep,dtype=np.float32),ex]),np.concatenate([np.asarray(zp,dtype=np.float32),zx]),eids+ext,ext

def official_seed_query_metrics(frame:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for seed in range(40,45):
        p=CAGE_ROOT/f'complete226_seed{seed}'/'complete_candidate_226_epoch_19.csv'
        d=pd.read_csv(p,dtype=str).fillna('');
        if len(d)!=len(frame): raise RuntimeError(f'official seed{seed} support row count drift')
        # Bind scores by the immutable original support row keys rather than row position alone.
        test=prepare_fixed_support(d); base=frame[['_support_row','reaction_id','protein_id','label']].copy()
        if not np.array_equal(test['_support_row'].to_numpy(),frame['_support_row'].to_numpy()): raise RuntimeError(f'seed{seed} support order drift')
        if not np.array_equal(test['protein_id'].astype(str).to_numpy(),frame['protein_id'].astype(str).to_numpy()): raise RuntimeError(f'seed{seed} protein support drift')
        test['cage_score']=pd.to_numeric(d['pred'],errors='raise')
        _,qm=evaluate_scores(test,'cage_score'); q=qm[qm.direction.eq('reaction_to_enzyme')].copy();q['seed']=seed;rows.append(q)
    return pd.concat(rows,ignore_index=True)

def bootstrap(catalyst_q:pd.DataFrame,cage_q:pd.DataFrame)->dict[str,object]:
    cols={'hit_at_10':'hit_at_10','reciprocal_rank':'reciprocal_rank','average_precision':'average_precision','ndcg_at_10':'ndcg_at_10','dcg_at_10':'dcg_at_10','roc_auc':'roc_auc'}
    mean=cage_q.groupby('query_id',sort=True)[list(cols.values())].mean().reset_index()
    cat=catalyst_q[catalyst_q.direction.eq('reaction_to_enzyme')].copy()
    j=cat.merge(mean,on='query_id',suffixes=('_catalyst','_cage'),validate='one_to_one'); rng=np.random.default_rng(20260905);out={}
    for name,col in cols.items():
        a=j[f'{col}_catalyst'].to_numpy(float);b=j[f'{col}_cage'].to_numpy(float);delta=a-b;vals=np.empty(BOOT,dtype=np.float32)
        for st in range(0,BOOT,1000):
            m=min(1000,BOOT-st);idx=rng.integers(0,len(delta),size=(m,len(delta)));vals[st:st+m]=delta[idx].mean(axis=1)
        out[name]={'enzymecage_mean':float(b.mean()),'catalyst':float(a.mean()),'delta_catalyst_minus_enzymecage':float(delta.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],'prob_delta_gt_0':float((vals>0).mean())}
    return {'protocol':'Enzyme-405 complete226 paired query bootstrap; frozen augmented-support BiME-Rank vs querywise mean of five fixed official EnzymeCAGE checkpoints seeds40-44','queries':len(j),'bootstrap_replicates':BOOT,'seed':20260905,'metrics':out}

def main():
    OUT.mkdir(parents=True,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); started=time.time()
    raw=pd.read_csv(PAIRS,dtype=str).fillna('');frame=prepare_fixed_support(raw); audit=pd.read_csv(OLD_AUDIT,dtype=str).fillna(''); alias=dict(zip(audit.protein_id,audit.feature_entity))
    ef,zf,cids,ext=build_augmented(); cset=set(cids)
    frame['model_entity']=frame.protein_id.map(lambda x: x if x in cset else alias.get(x,'')); missing=frame.loc[~frame.model_entity.isin(cset),'protein_id'].unique().tolist()
    if missing: raise RuntimeError(f'immutable support still has unmapped proteins after extension: {missing[:10]}')
    # Mapping is label-independent and must not create label conflicts for a reaction/entity pair.
    conflicts=frame.groupby(['reaction_id','model_entity']).label.agg(lambda x:len(set(map(int,x)))).gt(1).sum()
    if conflicts: raise RuntimeError(f'label conflicts after alias mapping: {conflicts}')
    qf,qids=load_query_reaction_library(QUERY_FEATURES);qidx={q:i for i,q in enumerate(qids)}; queries=sorted(frame.reaction_id.unique());qrows=np.asarray([qidx[q] for q in queries],dtype=np.int64); qmat=np.asarray(qf[qrows],dtype=np.float32)
    pm=load_models(PRIMARY/'models','production',device);sm=load_models(SECONDARY/'models','production',device);assert len(pm)==len(sm)==1
    print('encoding augmented proteins',len(cids),flush=True)
    pe=encode_chunks(pm[0],ef,kind='protein',device=device,chunk_size=8192);se=encode_chunks(sm[0],zf,kind='protein',device=device,chunk_size=8192)
    print('encoding 226 query reactions',flush=True)
    pre=encode_chunks(pm[0],qmat,kind='reaction',device=device,chunk_size=512);sre=encode_chunks(sm[0],qmat,kind='reaction',device=device,chunk_size=512)
    smiles_by_q=frame.drop_duplicates('reaction_id').set_index('reaction_id').CANO_RXN_SMILES.to_dict(); entity_index={v:i for i,v in enumerate(cids)}
    score_map={};aud=[]
    for st in range(0,len(queries),16):
        qs=queries[st:st+16]
        with torch.no_grad(): psc=(pre[st:st+len(qs)]@pe.T).float().cpu().numpy();ssc=(sre[st:st+len(qs)]@se.T).float().cpu().numpy()
        for j,q in enumerate(qs):
            nearest,sim=exact_max_train_binary_drfp_tanimoto(reaction_id=None,reaction_smiles=smiles_by_q[q],feature_dir=SIM_FEATURES,training_pairs=TRAINING_PAIRS,registered_reactions_csv=REGISTERED_REACTIONS,feature_cache_dir=None,failure_policy='warn')
            fused=fuse_r2e_scores(psc[j],ssc[j],cids,similarity=float(sim),threshold=.9,ranker_bundle=RANKER,ranker_sha256=RANKER_SHA,expected_pool_k=100,expected_prefix_k=100)
            # Priority is a monotone encoding of the complete 186170-candidate frozen order.
            score_map[q]=fused.priority_scores
            aud.append({'reaction_id':q,'similarity':float(sim),'nearest_train_reaction_id':nearest,'fallback_secondary':bool(fused.fallback_is_secondary),'union_size':int(fused.union_size),'prefix_size':int(fused.prefix_size)})
        print('ranked',min(st+16,len(queries)),'/',len(queries),flush=True)
    frame['bime_rank_score']=[float(score_map[q][entity_index[e]]) for q,e in frame[['reaction_id','model_entity']].itertuples(index=False)]
    metrics,qm=evaluate_scores(frame,'bime_rank_score');frame.to_csv(OUT/'pair_scores.csv',index=False);qm.to_csv(OUT/'query_metrics.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'runtime_audit.csv',index=False)
    cage_q=official_seed_query_metrics(frame);cage_q.to_csv(OUT/'enzymecage_seed_query_metrics.csv',index=False);boot=bootstrap(qm,cage_q);(OUT/'paired_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
    summary={'status':'completed','protocol':'Frozen BiME-Rank base LambdaRank on label-independent augmented 186170-protein universe, then projected to immutable Enzyme-405 complete226 support; CLIP structural expert is unavailable for raw non-registered reaction queries exactly as in production','support_selection_allowed_here':False,'target_labels_used_for_model_or_routing':False,'target_labels_used_for_extension_selection':False,'pairs':str(PAIRS),'pairs_sha256':sha256_file(PAIRS),'queries':int(frame.reaction_id.nunique()),'rows_raw':len(frame),'rows_canonical':len(frame.drop_duplicates(['reaction_id','protein_id'])),'candidate_uids':int(frame.protein_id.nunique()),'positive_rows_raw':int(frame.label.sum()),'base_candidate_count':185918,'extension_candidate_count':len(ext),'augmented_candidate_count':len(cids),'extension_ids_sha256':json.loads((ROOT/'results/bime_rank_unified_v1/enzyme405_extension_v1/scope_manifest.json').read_text())['ids_sha256'],'primary_model_dir':str(PRIMARY),'secondary_model_dir':str(SECONDARY),'ranker_sha256':RANKER_SHA,'structural_expert_applied':False,'structural_unavailability_reason':'raw Enzyme-405 reaction queries have no registered Rhea ID / native CLIP reaction identity','metrics':metrics,'paired_bootstrap_50000_vs_enzymecage':boot,'runtime_seconds':time.time()-started,'fairness_boundary':'No benchmark row or positive is removed. The 252 benchmark-only proteins are selected solely by immutable-support ID/sequence coverage and receive frozen ESM-C/EnzGFM representations; no downstream parameter is trained or tuned. 685 additional UID aliases reuse their pre-existing exact-sequence canonical production entity. Full augmented-universe ranking happens before fixed-support projection.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n');print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
