from __future__ import annotations
import hashlib,json,sys,math
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
from scipy import sparse
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library
from projects.active.terpene_screening.evaluate_broad_rhea_benchmark import encode_chunks
from projects.active.terpene_screening.bime_rank_r2e_runtime import fuse_bime_r2e_scores
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks,summarize_query_metrics,DEFAULT_BUDGETS,DEFAULT_TOP_PERCENTS
from projects.active.terpene_screening.run_bime_r2e_homology_context_v1 import rank_features,lexical

PRIMARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1'; SECONDARY=ROOT/'results/catalyst_clean_mainline_v1/r2e_enzgfm_center_router_v1'
PP=ROOT/'data/catalyst_candidate_universes/general_merged/proteins';SP=ROOT/'data/external/enzgfm_current/general_merged_650m_mean_v1';RF=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
BASE_BUNDLE=ROOT/'results/catalyst_clean_mainline_v1/r2e_lambdarank_fusion_v1';BASE_SHA='86b6fc7ff43fe1c59916dc6692cb38f513c877e1beed2c88902f00909cb7bb6e'
STRUCT_BUNDLE=ROOT/'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected';STRUCT_SHA='383dca7a176c47f3b0e431bb4b33492ae2f954c417f37f873969887b2d6f03e5';PCFG=json.loads((STRUCT_BUNDLE/'config.json').read_text())
P_ASSET=ROOT/'results/bime_rank_unified_v1/clipzyme_r2e_candidate_asset_v1';R_ASSET=ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
HOM=ROOT/'results/bime_rank_unified_v1/r2e_homology_context_v1/selected';HCFG=json.loads((HOM/'config.json').read_text());HOM_SHA=HCFG['ranker_sha256']
QFILE=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_query_ids.txt';PAIR=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_same_support_v1/mutual_cold_test_pairs.csv';SUP=ROOT/'results/clipzyme_native_extension_v1/r2e_strict650_candidate_ids.txt'
TRAIN=ROOT/'results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/train_pairs.csv';DIFF=ROOT/'results/rhea128_to141_external_v2/posthoc_difficulty/rhea128_to141_sprot_strict_double_cold_v2/reaction_slices.csv'
BASE_MET=ROOT/'results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/query_metrics.csv';OUT=ROOT/'results/bime_rank_unified_v1/r2e_homology_context_retention_v1';OUT.mkdir(parents=True,exist_ok=True)
POOL=100;PREFIX=100;BOOT=50000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def reaction_neighbors(qids,train_rids,all_rids,rf,dim):
 idx={r:i for i,r in enumerate(all_rids)};q=sparse.csr_matrix((np.asarray(rf[[idx[x] for x in qids],:dim])>0).astype(np.float32));tids=sorted(set(train_rids)&set(all_rids));t=sparse.csr_matrix((np.asarray(rf[[idx[x] for x in tids],:dim])>0).astype(np.float32));inter=(q@t.T).toarray();qc=np.asarray(q.sum(1)).ravel();tc=np.asarray(t.sum(1)).ravel();u=qc[:,None]+tc[None,:]-inter;s=np.divide(inter,u,out=np.zeros_like(inter),where=u>0);arr=np.asarray(tids,dtype=object);out={}
 for i,qid in enumerate(qids):o=np.lexsort((arr,-s[i]))[:5];out[qid]=[(str(arr[j]),float(s[i,j])) for j in o]
 return out
def hom_top100(raw_t,pidx,enz,neighbors):
 weights={}
 for rid,rw in neighbors:
  if rw<=0:continue
  for p in enz.get(rid,[]):
   if p in pidx:weights[p]=max(weights.get(p,0.),rw)
 if not weights:return np.empty(0,dtype=np.int32)
 seeds=sorted(weights);rows=np.asarray([pidx[p] for p in seeds],dtype=np.int64);w=np.asarray([weights[p] for p in seeds],dtype=np.float32);best=torch.full((raw_t.shape[0],),-1e9,dtype=torch.float32,device=raw_t.device)
 for st in range(0,len(rows),128):
  rr=torch.as_tensor(rows[st:st+128],dtype=torch.long,device=raw_t.device);ww=torch.as_tensor(w[st:st+128],dtype=torch.float32,device=raw_t.device)
  with torch.no_grad():tmp=(raw_t@raw_t[rr].T)*ww[None,:];best=torch.maximum(best,tmp.max(1).values)
 with torch.no_grad():_,ix=torch.topk(best,k=POOL,largest=True,sorted=True)
 return ix.cpu().numpy().astype(np.int32)
def boot(new,base):
 j=new.merge(base,on='query_id',suffixes=('_new','_base'),validate='one_to_one');rng=np.random.default_rng(20260905);out={}
 for m in ['reciprocal_rank','average_precision','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']:
  d=j[f'{m}_new'].to_numpy(float)-j[f'{m}_base'].to_numpy(float);vals=np.empty(BOOT,dtype=np.float32)
  for st in range(0,BOOT,1000):
   n=min(1000,BOOT-st);ii=rng.integers(0,len(d),size=(n,len(d)));vals[st:st+n]=d[ii].mean(1)
  out[m]={'delta':float(d.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
 return out

def main():
 assert sha(HOM/'ranker.json')==HOM_SHA and HCFG['external_metrics_used'] is False;ranker=xgb.Booster();ranker.load_model(HOM/'ranker.json');device=torch.device('cuda')
 pf,pids=load_protein_library(PP);sf,sids=load_protein_library(SP);assert pids==sids;raw_t=torch.as_tensor(np.asarray(pf,dtype=np.float32),device=device);pidx={p:i for i,p in enumerate(pids)};lex=lexical(pids)
 ps=load_feature_schema(PRIMARY);ss=load_feature_schema(SECONDARY);rf,rids=load_registered_reaction_feature_library(RF,ps);rf2,rids2=load_registered_reaction_feature_library(RF,ss);assert rids==rids2 and np.array_equal(rf,rf2);ridx={r:i for i,r in enumerate(rids)}
 pm=load_models(PRIMARY/'models','production',device);sm=load_models(SECONDARY/'models','production',device);pe=encode_chunks(pm[0],pf,kind='protein',device=device,chunk_size=8192);se=encode_chunks(sm[0],sf,kind='protein',device=device,chunk_size=8192);pre=encode_chunks(pm[0],rf,kind='reaction',device=device,chunk_size=8192);sre=encode_chunks(sm[0],rf,kind='reaction',device=device,chunk_size=8192)
 qids=[x.strip() for x in open(QFILE) if x.strip()];pairs=pd.read_csv(PAIR,dtype=str);pos=pairs.groupby('reaction_id').protein_id.agg(lambda s:set(s.astype(str))).to_dict();train=pd.read_csv(TRAIN,dtype=str).fillna('');enz=train.groupby('reaction_id').protein_id.agg(lambda s:sorted(set(s.astype(str)))).to_dict();rn=reaction_neighbors(qids,train.reaction_id.astype(str).unique(),rids,rf,int(ps.get('drfp_dimension',2048)))
 diff=pd.read_csv(DIFF,dtype={'reaction_id':str});sims=dict(zip(diff.reaction_id.astype(str),diff.max_train_drfp_tanimoto.astype(float)));support=[x.strip() for x in open(SUP) if x.strip()];supset=set(support);supmask=np.asarray([p in supset for p in pids],dtype=bool);supidx={p:i for i,p in enumerate(support)}
 rows=[];aud=[]
 for st in range(0,len(qids),8):
  qs=qids[st:st+8];qr=torch.as_tensor([ridx[q] for q in qs],dtype=torch.long,device=device)
  with torch.no_grad():psc=(pre[qr]@pe.T).cpu().numpy();ssc=(sre[qr]@se.T).cpu().numpy()
  for j,q in enumerate(qs):
   fused=fuse_bime_r2e_scores(psc[j],ssc[j],pids,reaction_id=q,similarity=float(sims[q]),threshold=.9,base_ranker_bundle=BASE_BUNDLE,base_ranker_sha256=BASE_SHA,structural_ranker_bundle=STRUCT_BUNDLE,structural_ranker_sha256=STRUCT_SHA,clip_protein_asset=P_ASSET,clip_reaction_asset=R_ASSET,clip_protein_manifest_sha256=PCFG['clip_protein_manifest_sha256'],clip_reaction_manifest_sha256=PCFG['clip_reaction_manifest_sha256'],device='cuda',expected_pool_k=100,expected_prefix_k=100);bord=fused.full_order;binv=np.empty(len(pids),dtype=np.int32);binv[bord]=np.arange(1,len(pids)+1,dtype=np.int32)
   hord=hom_top100(raw_t,pidx,enz,rn[q]);hinv=np.full(len(pids),len(pids)+1,dtype=np.int32);hinv[hord]=np.arange(1,len(hord)+1,dtype=np.int32);u=np.unique(np.concatenate([bord[:POOL],hord])).astype(np.int32);X=rank_features(u,binv,hinv,len(pids),rn[q][0][1] if rn[q] else 0.);pred=ranker.predict(xgb.DMatrix(X));take=np.lexsort((lex[u],-pred))[:min(PREFIX,len(u))];sel=u[take];mask=np.zeros(len(pids),dtype=bool);mask[sel]=1;order=np.concatenate([sel,bord[~mask[bord]]]);projected=order[supmask[order]];inv={pids[int(r)]:k+1 for k,r in enumerate(projected)};pr=np.asarray([inv[p] for p in pos[q]],dtype=np.int32);rows.append({'query_id':q,**evaluate_full_candidate_ranks(pr,len(support))});aud.append({'query_id':q,'homology_top100':len(hord),'homology_max_reaction_similarity':rn[q][0][1] if rn[q] else 0.,'train_only_neighbor_associations':True,'structure_expert_applied':bool(fused.structure_expert_applied)})
  print('retention',min(st+8,len(qids)),'/',len(qids),flush=True)
 new=pd.DataFrame(rows);new.to_csv(OUT/'query_metrics.csv',index=False);pd.DataFrame(aud).to_csv(OUT/'audit.csv',index=False);baseq=pd.read_csv(BASE_MET,dtype={'query_id':str});metrics=summarize_query_metrics(new,budgets=DEFAULT_BUDGETS,top_percents=DEFAULT_TOP_PERCENTS);bm=summarize_query_metrics(baseq,budgets=DEFAULT_BUDGETS,top_percents=DEFAULT_TOP_PERCENTS);paired=boot(new,baseq);delta={k:float(metrics[k]-bm[k]) for k in ['mrr','map','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']};safe=delta['mrr']>=-.002 and delta['map']>=-.002 and delta['hit_at_20']>=-.01 and delta['hit_at_50']>=-.01;material=delta['mrr']>=.003 or delta['map']>=.003 or delta['hit_at_20']>=.01 or delta['hit_at_50']>=.01;retain=bool(safe and material);res={'status':'retention_passed' if retain else 'retention_failed','protocol':'Frozen R2E homology-context after internal admission; Rhea128->141 strict mutual-train-cold 144-query x 166202 common support. Homology uses only benchmark train-side reaction->enzyme associations; full 185918 structural BiME ranking is formed first, then Top100 union with homology context, then common-support projection. External labels are veto-only.','queries':len(qids),'full_candidates':len(pids),'common_candidates':len(support),'positive_pairs':len(pairs),'external_metrics_used_for_selection':False,'external_metrics_used_for_retuning':False,'homology_ranker_sha256':HOM_SHA,'base_structural_bime':bm,'homology_context':metrics,'delta':delta,'paired_bootstrap_50000':paired,'retention_safe':safe,'retention_material':material,'retention_gate_passed':retain};(OUT/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2));raise SystemExit(0 if retain else 2)
if __name__=='__main__':main()
