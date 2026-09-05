from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,xgboost as xgb
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.bime_rank_e2r_runtime import BiMEE2RRuntime
from projects.active.terpene_screening.bime_rank_r2e_seed_runtime import context_features,lexical_rank,masked_order
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.run_bime_e2r_seed_context_v1 import seed_rows

SEED=ROOT/'results/bime_rank_unified_v1/e2r_seed_context_v1/selected'
PAIR=ROOT/'results/rhea128_to141_external_v2/rhea128_to141_sprot_strict_double_cold_v2/test_pairs.csv'
CLIP_BASE=ROOT/'results/clipzyme_native_extension_v1/e2r_strict650_mutual_cold_10131_v2_lexical'
RDIR=ROOT/'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1'
OUT=ROOT/'results/bime_rank_unified_v1/e2r_seed_context_retention_v1'
BOOT=50000;POOL=100;PREFIX=100

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(df):
 return {'mrr':float(df.reciprocal_rank.mean()),'map':float(df.average_precision.mean()),'macro_roc_auc':float(df.roc_auc.mean()),'ndcg_at_10':float(df.ndcg_at_10.mean()),'hit_at_10':float(df.hit_at_10.mean()),'hit_at_20':float(df.hit_at_20.mean()),'hit_at_50':float(df.hit_at_50.mean()),'median_best_positive_rank':float(df.best_positive_rank.median())}
def filtered_metrics(order,hidden,common_mask):
 filtered=order[common_mask[order]]; inv=np.full(len(common_mask),len(filtered)+1,dtype=np.int32);inv[filtered]=np.arange(1,len(filtered)+1,dtype=np.int32);return evaluate_full_candidate_ranks(inv[hidden],len(filtered))
def context_order(ranker,base_order,seed_scores,srow,lex):
 bo=base_order[base_order!=srow];binv=np.full(len(base_order),len(bo)+1,dtype=np.int32);binv[bo]=np.arange(1,len(bo)+1,dtype=np.int32);so,sinv=masked_order(seed_scores,lex,{int(srow)});u=np.unique(np.concatenate([bo[:POOL],so[:POOL]])).astype(np.int32);u=u[u!=srow];X=context_features(u,binv,seed_scores,sinv,len(bo));pred=ranker.predict(xgb.DMatrix(X));take=np.lexsort((lex[u],-pred))[:min(PREFIX,len(u))];sel=u[take];chosen=np.zeros(len(base_order),dtype=bool);chosen[sel]=True;tail=bo[~chosen[bo]];return np.concatenate([sel,tail]).astype(np.int32,copy=False)
def boot(a,b,seed):
 j=a.merge(b,on='trial_id',suffixes=('_new','_base'),validate='one_to_one');rng=np.random.default_rng(seed);out={}
 for m in ['reciprocal_rank','average_precision','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']:
  d=j[f'{m}_new'].to_numpy(float)-j[f'{m}_base'].to_numpy(float);vals=np.empty(BOOT,dtype=np.float32)
  for st in range(0,BOOT,1000):
   n=min(1000,BOOT-st);idx=rng.integers(0,len(d),size=(n,len(d)));vals[st:st+n]=d[idx].mean(1)
  out[m]={'delta':float(d.mean()),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
 return out

def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=json.loads((SEED/'config.json').read_text());assert sha(SEED/'ranker.json')==cfg['ranker_sha256'] and cfg['external_metrics_used'] is False;ranker=xgb.Booster();ranker.load_model(SEED/'ranker.json')
 rt=BiMEE2RRuntime(device='cuda');ids=list(rt.candidate_ids);idx={r:i for i,r in enumerate(ids)};lex=lexical_rank(ids)
 rent=pd.read_csv(RDIR/'entries.csv',dtype=str).fillna('');common=set(rent.loc[rent.clipzyme_supported.str.lower().eq('true'),'reaction_id'].astype(str));cmask=np.asarray([r in common for r in ids],dtype=bool);assert int(cmask.sum())==10131
 clipq=pd.read_csv(CLIP_BASE/'official_clipzyme_query_metrics.csv',dtype={'query_id':str});qids=clipq.query_id.astype(str).tolist();assert len(qids)==248
 pairs=pd.read_csv(PAIR,dtype=str).fillna('');pos=pairs[pairs.protein_id.isin(qids)&pairs.reaction_id.isin(common)].groupby('protein_id').reaction_id.agg(lambda s:sorted(set(s.astype(str)))).to_dict();assert sum(map(len,pos.values()))==348
 # Four non-structural frozen reaction embedding matrices used by BiME base.
 mats=[rt.base._reaction_embeddings[n].detach().cpu().numpy().astype(np.float32,copy=False) for n in rt.base.expert_names]
 rec={'context':[],'zero_shot':[],'seed_only':[]};audit=[];eligible=0
 for i,q in enumerate(qids,1):
  prows=np.asarray([idx[r] for r in pos[q]],dtype=np.int32);seeds=seed_rows(q,prows)
  if not seeds:continue
  eligible+=1;base=rt.rank_registered(q);assert base.structure_expert_applied
  for rep,srow in enumerate(seeds):
   hidden=prows[prows!=srow];ss=np.zeros(len(ids),dtype=np.float32)
   for M in mats:ss+=M@M[srow]
   ss/=len(mats);so,_=masked_order(ss,lex,{int(srow)});bo=base.order[base.order!=srow];co=context_order(ranker,base.order,ss,srow,lex);tid=f'{q}|seed={ids[srow]}|rep={rep}'
   for name,o in [('context',co),('zero_shot',bo),('seed_only',so)]:rec[name].append({'trial_id':tid,'query_id':q,'seed_id':ids[srow],**filtered_metrics(o,hidden,cmask)})
   audit.append({'trial_id':tid,'query_id':q,'seed_id':ids[srow],'hidden_positives':len(hidden),'structure_expert_applied':True})
  if i%25==0 or i==len(qids):print('e2r seed retention',i,'/',len(qids),flush=True)
 frames={k:pd.DataFrame(v) for k,v in rec.items()};[v.to_csv(OUT/f'{k}_query_metrics.csv',index=False) for k,v in frames.items()];pd.DataFrame(audit).to_csv(OUT/'audit.csv',index=False);mm={k:metrics(v) for k,v in frames.items()};delta={k:mm['context'][k]-mm['zero_shot'][k] for k in mm['context'] if k!='median_best_positive_rank'};boots={'context_minus_zero_shot':boot(frames['context'],frames['zero_shot'],20260905),'context_minus_seed_only':boot(frames['context'],frames['seed_only'],20260906)}
 retain=bool(delta['mrr']>=-.002 and delta['map']>=-.002 and delta['hit_at_20']>=-.01 and delta['hit_at_50']>=-.01 and (delta['mrr']>0 or delta['map']>0 or delta['hit_at_20']>0 or delta['hit_at_50']>0));res={'status':'retention_passed' if retain else 'retention_failed','protocol':'Frozen E2R seed-context evaluated once after internal admission on pre-existing Rhea128->141 strict mutual-cold 248-query labels; one known positive reaction masked, hidden positives scored after full 11081 BiME order then filtered to identical 10131 CLIP-supported common reaction support','external_metrics_used_for_selection':False,'external_metrics_used_for_retuning':False,'retention_is_veto_only':True,'eligible_queries':eligible,'trials':len(frames['context']),'full_candidate_count':11081,'common_candidate_count':10131,'seed_ranker_sha256':cfg['ranker_sha256'],'metrics':mm,'delta_context_vs_zero_shot':delta,'paired_bootstrap_50000':boots,'retention_gate_passed':retain};(OUT/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2));raise SystemExit(0 if retain else 2)
if __name__=='__main__':main()
