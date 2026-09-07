from pathlib import Path
import sys,json
import numpy as np,pandas as pd,torch,xgboost as xgb
ROOT=Path('/home/s241850073/igem2026'); sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.r2e_lambdarank_runtime import build_features,full_order,lexical_rank
from projects.active.terpene_screening.bime_rank_r2e_runtime import _structural_features
from projects.active.terpene_screening.run_bime_r2e_reciprocal_consistency_v1 import _reverse_ranks_for_pairs,append_reciprocal_features
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks,summarize_query_metrics,DEFAULT_BUDGETS,DEFAULT_TOP_PERCENTS
PRIMARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'
SECONDARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1'
PP=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
SP=ROOT/'data/external/enzgfm_current/general_merged_650m_mean_v1'
RF=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
STRUCT=ROOT/'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected'
RECIP=ROOT/'results/bime_rank_unified_v1/r2e_reciprocal_consistency_v1/selected'
RECIP_SHA='db065ab755eac60e01617d8b93474f7f830022bfb7eadf11527330a69b6e4dd0'
P_ASSET=ROOT/'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1'
R_ASSET=ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
QFILE=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt'
PAIR=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv'
SUP=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt'
DIFF=ROOT/'results/rhea128_to141_external_v2/posthoc_difficulty/rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv'
STRUCT_MET=ROOT/'results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/query_metrics.csv'
OUT=ROOT/'results/bime_rank_unified_v1/r2e_reciprocal_external_confirmation_v1'; OUT.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda')
# Frozen universal experts and full candidate axes.
pf,pids=load_protein_library(PP); sf,sids=load_protein_library(SP); assert pids==sids and len(pids)==185918
ps=load_feature_schema(PRIMARY); ss=load_feature_schema(SECONDARY)
rf,rids=load_registered_reaction_feature_library(RF,ps); rf2,rids2=load_registered_reaction_feature_library(RF,ss); assert rids==rids2 and np.array_equal(rf,rf2)
pm=load_models(PRIMARY/'models','production',device); sm=load_models(SECONDARY/'models','production',device); assert len(pm)==len(sm)==1
print('encoding production protein/reaction towers',flush=True)
pe=encode_chunks(pm[0],pf,kind='protein',device=device,chunk_size=8192); se=encode_chunks(sm[0],sf,kind='protein',device=device,chunk_size=8192)
pre=encode_chunks(pm[0],rf,kind='reaction',device=device,chunk_size=8192); sre=encode_chunks(sm[0],rf,kind='reaction',device=device,chunk_size=8192)
# Frozen CLIPZyme assets.
ce=pd.read_csv(P_ASSET/'entries.csv',dtype=str).fillna(''); crows=ce.candidate_row.astype(int).to_numpy(np.int32); cids=ce.protein_id.astype(str).tolist(); assert [pids[int(r)] for r in crows]==cids
cmat=torch.as_tensor(np.array(np.load(P_ASSET/'embeddings.npy',mmap_mode='r'),dtype=np.float32,copy=True),device=device)
cre=pd.read_csv(R_ASSET/'entries.csv',dtype=str).fillna(''); crm={str(r):int(row) for r,row,s in cre[['reaction_id','row','clipzyme_supported']].itertuples(index=False) if str(s).lower()=='true'}
crmat=np.load(R_ASSET/'embeddings.npy',mmap_mode='r')
clip_lookup=np.full(len(pids),-1,dtype=np.int32); clip_lookup[crows]=np.arange(len(crows),dtype=np.int32)
# Evaluation protocol.
qids=[x.strip() for x in open(QFILE) if x.strip()]; assert len(qids)==144; ridx={r:i for i,r in enumerate(rids)}; assert all(q in ridx and q in crm for q in qids)
pairs=pd.read_csv(PAIR,dtype=str); positives=pairs.groupby('reaction_id')['protein_id'].apply(lambda x:set(map(str,x))).to_dict()
diff=pd.read_csv(DIFF,dtype={'reaction_id':str}); sims=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)))
support=[x.strip() for x in open(SUP) if x.strip()]; assert len(support)==166202 and support==sorted(support)
full_index={p:i for i,p in enumerate(pids)}; sup_full=np.asarray([full_index[p] for p in support],dtype=np.int32); sup_mask=np.zeros(len(pids),dtype=bool); sup_mask[sup_full]=True; sup_index={p:i for i,p in enumerate(support)}
plex=lexical_rank(pids); rlex=np.empty(len(rids),dtype=np.int32); ro=np.argsort(np.asarray(rids,dtype=object),kind='stable'); rlex[ro]=np.arange(len(ro),dtype=np.int32)
# Build exactly the admitted structural shortlist/features first, storing fallback order.
Xs=[]; unions=[]; fallback_orders=[]; qrow_pairs=[]; ptr=[0]; audit=[]
for st in range(0,len(qids),16):
  qs=qids[st:st+16]; qr=torch.tensor([ridx[q] for q in qs],dtype=torch.long,device=device)
  with torch.no_grad():
    psc=(pre[qr]@pe.T).float().cpu().numpy(); ssc=(sre[qr]@se.T).float().cpu().numpy()
    cq=torch.as_tensor(np.stack([np.array(crmat[crm[q]],dtype=np.float32,copy=True) for q in qs]),device=device)
    csc=(cq@cmat.T).float().cpu().numpy()
  for j,q in enumerate(qs):
    s0=psc[j].astype(np.float32,copy=False); s1=ssc[j].astype(np.float32,copy=False)
    po,pinv=full_order(s0,plex); so,sinv=full_order(s1,plex); use_secondary=float(sims[q])<0.9; fb=so if use_secondary else po
    cs=csc[j].astype(np.float32,copy=False); clex=plex[crows]; co=np.lexsort((clex,-cs)).astype(np.int32); cinv=np.empty(len(co),dtype=np.int32); cinv[co]=np.arange(1,len(co)+1,dtype=np.int32)
    union=np.unique(np.concatenate([po[:100],so[:100],crows[co[:100]]])).astype(np.int32)
    bx=build_features(s0,s1,union,pinv,sinv,use_secondary,float(sims[q])); sx=_structural_features(bx,union,cs,cinv,clip_lookup,len(pids))
    Xs.append(sx); unions.append(union); fallback_orders.append(fb.astype(np.int32,copy=False)); qrow_pairs.append(np.full(len(union),ridx[q],dtype=np.int32)); ptr.append(ptr[-1]+len(union))
    audit.append({'query_id':q,'union_size':len(union),'fallback_secondary':bool(use_secondary)})
  print('forward shortlist',min(st+16,len(qids)),'/',len(qids),flush=True)
X40=np.concatenate(Xs); pair_rows=np.concatenate(unions); pair_qrows=np.concatenate(qrow_pairs)
print('reverse primary',len(pair_rows),'pairs',len(np.unique(pair_rows)),'unique proteins',flush=True)
rr0=_reverse_ranks_for_pairs(pre,pe,pair_rows,pair_qrows,rlex,chunk_size=512)
print('reverse secondary',flush=True)
rr1=_reverse_ranks_for_pairs(sre,se,pair_rows,pair_qrows,rlex,chunk_size=512)
X52=append_reciprocal_features(X40,rr0,rr1,len(rids))
# Frozen reciprocal ranker selected before this external pass.
ranker_path=RECIP/'ranker.json'; import hashlib
assert hashlib.sha256(ranker_path.read_bytes()).hexdigest()==RECIP_SHA
booster=xgb.Booster(); booster.load_model(ranker_path); pred=booster.predict(xgb.DMatrix(X52))
rows=[]
for qi,q in enumerate(qids):
  a,b=ptr[qi],ptr[qi+1]; union=unions[qi]; local=pred[a:b]; order=np.lexsort((plex[union],-local)); selected=union[order[:min(100,len(order))]]
  mask=np.zeros(len(pids),dtype=bool); mask[selected]=True; full=np.concatenate([selected,fallback_orders[qi][~mask[fallback_orders[qi]]]]).astype(np.int32)
  assert len(full)==len(pids) and len(np.unique(full))==len(pids)
  projected=full[sup_mask[full]]; inv=np.empty(len(support),dtype=np.int32)
  for rank,fr in enumerate(projected,1): inv[sup_index[pids[int(fr)]]]=rank
  pr=np.asarray([inv[sup_index[p]] for p in positives[q]],dtype=np.int32); rows.append({'query_id':q,**evaluate_full_candidate_ranks(pr,len(support))})
frame=pd.DataFrame(rows); frame.to_csv(OUT/'query_metrics.csv',index=False); pd.DataFrame(audit).to_csv(OUT/'audit.csv',index=False)
metrics=summarize_query_metrics(frame,budgets=DEFAULT_BUDGETS,top_percents=DEFAULT_TOP_PERCENTS)
struct=pd.read_csv(STRUCT_MET,dtype={'query_id':str}); j=frame.merge(struct,on='query_id',suffixes=('_new','_struct'),validate='one_to_one')
metric_cols={'mrr':'reciprocal_rank','map':'average_precision','macro_roc_auc':'roc_auc','ndcg_at_10':'ndcg_at_10','hit_at_10':'hit_at_10','hit_at_20':'hit_at_20','hit_at_50':'hit_at_50'}
rng=np.random.default_rng(20260905); B=50000; paired={}
for metric,col in metric_cols.items():
  nv=j[f'{col}_new'].to_numpy(float); bv=j[f'{col}_struct'].to_numpy(float); d=nv-bv; vals=np.empty(B,dtype=np.float32)
  for bs in range(0,B,1000):
    idx=rng.integers(0,len(d),size=(min(1000,B-bs),len(d))); vals[bs:bs+len(idx)]=d[idx].mean(axis=1)
  paired[metric]={'structural_baseline':float(bv.mean()),'reciprocal':float(nv.mean()),'delta':float(d.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
summary={'protocol':'Frozen reciprocal-consistency BiME-Rank, selected only on internal clean-dev, single evaluation on pre-existing Rhea128->141 strict mutual-train-cold 144x166202 support','external_metrics_used_for_selection':False,'queries':144,'common_candidates':166202,'positive_pairs':len(pairs),'metrics':metrics,'paired_bootstrap_50000_vs_structural_bime':paired,'reciprocal_ranker_sha256':RECIP_SHA}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2),flush=True)
