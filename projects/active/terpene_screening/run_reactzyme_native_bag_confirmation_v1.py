from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.broad_rhea_metrics import candidate_ranking_context,evaluate_full_candidate_scores
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_models,load_protein_library,load_registered_reaction_feature_library
from projects.active.terpene_screening.train_reactzyme_native_bag_adapter_v1 import BagAdapter,bag_feature
SPLIT=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation/split'; TEACHER=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation/teacher'; OUT=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation'; FEATURE=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'; PROT=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'; RHEA=ROOT/'data/external/reactzyme/rhea_molecules.tsv'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aggregate(q,prefix):
 return {'mrr':float(q[f'{prefix}_reciprocal_rank'].mean()),'map':float(q[f'{prefix}_average_precision'].mean()),'ndcg_at_10':float(q[f'{prefix}_ndcg_at_10'].mean()),'hit_at_10':float(q[f'{prefix}_hit_at_10'].mean()),'hit_at_20':float(q[f'{prefix}_hit_at_20'].mean()),'hit_at_50':float(q[f'{prefix}_hit_at_50'].mean()),'median_best_positive_rank':float(q[f'{prefix}_best_positive_rank'].median())}
def main():
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); trp=SPLIT/'training_pairs.csv'; dvp=SPLIT/'dev_pairs.csv'; train=pd.read_csv(trp,dtype=str); dev=pd.read_csv(dvp,dtype=str)
 tr_ids=sorted(set(train.reaction_id.astype(str))); dv_ids=sorted(set(dev.reaction_id.astype(str))); assert not(set(tr_ids)&set(dv_ids)); assert not(set(train.protein_id)&set(dev.protein_id))
 bags=pd.read_csv(RHEA,sep='\t',dtype=str).fillna('').set_index('Rhea ID'); schema=load_feature_schema(FEATURE); rf,rids=load_registered_reaction_feature_library(FEATURE,schema); rix={x:i for i,x in enumerate(rids)}; assert all(x in bags.index and x in rix for x in tr_ids+dv_ids)
 teacher=load_models(TEACHER/'models','production',device)[0]; teacher.eval()
 all_ids=tr_ids+dv_ids; bf=np.stack([bag_feature(bags.loc[r,'substrate'],bags.loc[r,'product'])[0] for r in all_ids]).astype(np.float32)
 with torch.no_grad():
  tl=[]
  for s in range(0,len(all_ids),512): tl.append(teacher.encode_reactions(torch.from_numpy(np.stack([rf[rix[r]] for r in all_ids[s:s+512]])).to(device)).cpu().numpy())
 tl=np.concatenate(tl).astype(np.float32); ntr=len(tr_ids)
 torch.manual_seed(20260901); np.random.seed(20260901); adapter=BagAdapter().to(device); opt=torch.optim.AdamW(adapter.parameters(),lr=1e-3,weight_decay=1e-4); rng=np.random.default_rng(20260901); hist=[]; tx=torch.from_numpy(bf[:ntr]); ty=torch.from_numpy(tl[:ntr])
 for epoch in range(1,81):
  adapter.train(); order=rng.permutation(ntr); losses=[]
  for s in range(0,ntr,256):
   idx=torch.from_numpy(order[s:s+256]); pred=adapter(tx[idx].to(device)); y=ty[idx].to(device); loss=(1-(pred*y).sum(-1)).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
  hist.append({'epoch':epoch,'train_cosine_loss':float(np.mean(losses))})
 adapter.eval(); out=OUT/'confirmation'; out.mkdir(parents=True,exist_ok=True); torch.save({'model_type':'reactzyme_native_bag_adapter_v1_confirmation','model_state_dict':adapter.state_dict(),'seed':20260901},out/'adapter.pt'); pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False)
 dev_rows=np.arange(ntr,len(all_ids));
 with torch.no_grad(): pred=adapter(torch.from_numpy(bf[dev_rows]).to(device)).cpu().numpy().astype(np.float32)
 target=tl[dev_rows]; cos=(pred*target).sum(1)
 pf,pids=load_protein_library(PROT); pidx={x:i for i,x in enumerate(pids)}; pe=[]
 with torch.no_grad():
  for s in range(0,len(pf),4096): pe.append(teacher.encode_proteins(torch.from_numpy(pf[s:s+4096]).to(device)).cpu().numpy())
 pe=np.concatenate(pe).astype(np.float32); context=candidate_ranking_context(pids); pos=dev.groupby('reaction_id').protein_id.apply(set).to_dict(); rows=[]
 for i,rid in enumerate(dv_ids):
  tscore=target[i]@pe.T; ascore=pred[i]@pe.T; tm=evaluate_full_candidate_scores(tscore,pids,pos[rid],candidate_index=context[0],lexical_order=context[1]); am=evaluate_full_candidate_scores(ascore,pids,pos[rid],candidate_index=context[0],lexical_order=context[1]); rows.append({'reaction_id':rid,'teacher_cosine':float(cos[i]),**{'teacher_'+k:v for k,v in tm.items()},**{'adapter_'+k:v for k,v in am.items()}})
 q=pd.DataFrame(rows); q.to_csv(out/'query_metrics.csv',index=False); t=aggregate(q,'teacher'); a=aggregate(q,'adapter'); ret={k:a[k]/t[k] for k in ['mrr','map','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']}; rank_ratio=a['median_best_positive_rank']/t['median_best_positive_rank']; checks={'teacher_cosine_mean_ge_0p80':float(q.teacher_cosine.mean())>=0.80,'mrr_retention_ge_0p90':ret['mrr']>=0.90,'map_retention_ge_0p90':ret['map']>=0.90,'ndcg10_retention_ge_0p88':ret['ndcg_at_10']>=0.88,'hit10_retention_ge_0p88':ret['hit_at_10']>=0.88,'hit20_retention_ge_0p88':ret['hit_at_20']>=0.88,'hit50_retention_ge_0p90':ret['hit_at_50']>=0.90,'median_rank_ratio_le_1p50':rank_ratio<=1.50}
 result={'status':'pass' if all(checks.values()) else 'reject','n_queries':len(q),'teacher_cosine_mean':float(q.teacher_cosine.mean()),'teacher_cosine_median':float(q.teacher_cosine.median()),'teacher':t,'adapter':a,'retention':ret,'median_best_positive_rank_ratio':rank_ratio,'checks':checks,'external_reactzyme_metrics_used':False,'confirmation_retraining_allowed':False,'training_pairs_sha256':sha(trp),'dev_pairs_sha256':sha(dvp),'teacher_checkpoint_sha256':sha(TEACHER/'models/production_seed20260723.pt'),'adapter_checkpoint_sha256':sha(out/'adapter.pt')}; (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
