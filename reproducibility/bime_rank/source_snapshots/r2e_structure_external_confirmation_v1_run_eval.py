from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch
ROOT=Path('/home/s241850073/igem2026'); sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.bime_rank_r2e_runtime import fuse_bime_r2e_scores
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks,summarize_query_metrics,DEFAULT_BUDGETS,DEFAULT_TOP_PERCENTS
PRIMARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'
SECONDARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1'
PP=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
SP=ROOT/'data/external/enzgfm_current/general_merged_650m_mean_v1'
RF=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
BASE_BUNDLE=ROOT/'results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1'
BASE_SHA='86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'
STRUCT_BUNDLE=ROOT/'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected'
STRUCT_SHA='383dca7a176c47f3b0e431bb4b33492ae2f954c417f37f873969887b2d6f03e5'
P_ASSET=ROOT/'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1'
R_ASSET=ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
PCFG=json.loads((STRUCT_BUNDLE/'config.json').read_text())
QFILE=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt'
PAIR=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv'
SUP=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt'
DIFF=ROOT/'results/rhea128_to141_external_v2/posthoc_difficulty/rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv'
OLD=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_current_system_fair_v1/query_metrics.csv'
CLIP=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_clipzyme_fair_v1/clipzyme_query_metrics.csv'
OUT=ROOT/'results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1'; OUT.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda')
pf,pids=load_protein_library(PP); sf,sids=load_protein_library(SP); assert pids==sids and len(pids)==185918
ps=load_feature_schema(PRIMARY); ss=load_feature_schema(SECONDARY)
rf,rids=load_registered_reaction_feature_library(RF,ps); rf2,rids2=load_registered_reaction_feature_library(RF,ss); assert rids==rids2 and np.array_equal(rf,rf2)
pm=load_models(PRIMARY/'models','production',device); sm=load_models(SECONDARY/'models','production',device); assert len(pm)==len(sm)==1
print('encoding proteins/reactions',flush=True)
pe=encode_chunks(pm[0],pf,kind='protein',device=device,chunk_size=8192); se=encode_chunks(sm[0],sf,kind='protein',device=device,chunk_size=8192)
pre=encode_chunks(pm[0],rf,kind='reaction',device=device,chunk_size=8192); sre=encode_chunks(sm[0],rf,kind='reaction',device=device,chunk_size=8192)
qids=[x.strip() for x in open(QFILE) if x.strip()]; assert len(qids)==144; ridx={r:i for i,r in enumerate(rids)}
pairs=pd.read_csv(PAIR,dtype=str); pos=pairs.groupby('reaction_id')['protein_id'].apply(lambda x:set(map(str,x))).to_dict()
diff=pd.read_csv(DIFF,dtype={'reaction_id':str}); sims=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)))
support=[x.strip() for x in open(SUP) if x.strip()]; support_set=set(support); assert len(support)==166202 and support==sorted(support)
full_index={p:i for i,p in enumerate(pids)}; sup_full=np.array([full_index[p] for p in support],dtype=np.int32); sup_mask=np.zeros(len(pids),dtype=bool); sup_mask[sup_full]=True; sup_index={p:i for i,p in enumerate(support)}
rows=[]; audits=[]
for st in range(0,len(qids),16):
  qs=qids[st:st+16]; qr=torch.tensor([ridx[q] for q in qs],dtype=torch.long,device=device)
  with torch.no_grad():
    psc=(pre[qr]@pe.T).float().cpu().numpy(); ssc=(sre[qr]@se.T).float().cpu().numpy()
  for j,q in enumerate(qs):
    fused=fuse_bime_r2e_scores(psc[j],ssc[j],pids,reaction_id=q,similarity=float(sims[q]),threshold=0.9,base_ranker_bundle=BASE_BUNDLE,base_ranker_sha256=BASE_SHA,structural_ranker_bundle=STRUCT_BUNDLE,structural_ranker_sha256=STRUCT_SHA,clip_protein_asset=P_ASSET,clip_reaction_asset=R_ASSET,clip_protein_manifest_sha256=PCFG['clip_protein_manifest_sha256'],clip_reaction_manifest_sha256=PCFG['clip_reaction_manifest_sha256'],device='cuda',expected_pool_k=100,expected_prefix_k=100)
    assert fused.structure_expert_applied, q
    projected=fused.full_order[sup_mask[fused.full_order]]
    inv=np.empty(len(support),dtype=np.int32)
    for rank,fr in enumerate(projected,1): inv[sup_index[pids[int(fr)]]]=rank
    pr=np.array([inv[sup_index[p]] for p in pos[q]],dtype=np.int32)
    rows.append({'query_id':q,**evaluate_full_candidate_ranks(pr,len(support))})
    audits.append({'query_id':q,'similarity':float(sims[q]),'fallback_secondary':bool(fused.fallback_is_secondary),'union_size':int(fused.union_size),'prefix_size':int(fused.prefix_size),'structure_expert_applied':bool(fused.structure_expert_applied),'positives':len(pr)})
  print('done',min(st+16,len(qids)),'/',len(qids),flush=True)
frame=pd.DataFrame(rows); frame.to_csv(OUT/'query_metrics.csv',index=False); pd.DataFrame(audits).to_csv(OUT/'audit.csv',index=False)
metrics=summarize_query_metrics(frame,budgets=DEFAULT_BUDGETS,top_percents=DEFAULT_TOP_PERCENTS)
old=pd.read_csv(OLD,dtype={'query_id':str}); clip=pd.read_csv(CLIP,dtype={'query_id':str})
def suffix_metrics(df,suffix):
  return df.rename(columns={c:f'{c}_{suffix}' for c in df.columns if c!='query_id'})
joined=(suffix_metrics(frame,'new')
        .merge(suffix_metrics(old,'old'),on='query_id',validate='one_to_one')
        .merge(suffix_metrics(clip,'clip'),on='query_id',validate='one_to_one'))
per_query_columns={
  'mrr':'reciprocal_rank', 'map':'average_precision', 'macro_roc_auc':'roc_auc',
  'ndcg_at_10':'ndcg_at_10', 'hit_at_10':'hit_at_10', 'hit_at_20':'hit_at_20', 'hit_at_50':'hit_at_50',
}
rng=np.random.default_rng(20260905); B=50000; paired={}
for metric,col in per_query_columns.items():
  nval=joined[f'{col}_new'].to_numpy(float); oval=joined[f'{col}_old'].to_numpy(float); cval=joined[f'{col}_clip'].to_numpy(float)
  rec={}
  for label,basev in [('old_bime',oval),('clipzyme',cval)]:
    delta=nval-basev; vals=np.empty(B,dtype=np.float32)
    for s in range(0,B,1000):
      idx=rng.integers(0,len(delta),size=(min(1000,B-s),len(delta))); vals[s:s+len(idx)]=delta[idx].mean(axis=1)
    rec[label]={'baseline':float(basev.mean()),'new':float(nval.mean()),'delta':float(delta.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
  paired[metric]=rec
summary={'protocol':'Frozen BiME-Rank R2E CLIPZyme expert, selected solely on internal clean-dev, evaluated once on pre-existing Rhea128->141 strict double-cold mutual-train-cold 144-query x 166202 common protein support','external_metrics_used_for_selection':False,'queries':len(qids),'full_candidates_ranked':len(pids),'common_candidates':len(support),'positive_pairs':len(pairs),'metrics':metrics,'paired_bootstrap_50000':paired,'structural_ranker_sha256':STRUCT_SHA,'base_ranker_sha256':BASE_SHA,'all_queries_structure_supported':bool(pd.DataFrame(audits).structure_expert_applied.all())}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)
