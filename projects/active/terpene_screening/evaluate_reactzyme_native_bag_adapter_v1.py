from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig,TerpeneDualTower
from projects.active.terpene_screening.train_reactzyme_native_bag_adapter_v1 import BagAdapter,BASE,OUT
PROT=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'

def metrics(scores:np.ndarray,pos_rows:list[int]):
    order=np.argsort(-scores,kind='stable'); ranks=np.empty(len(scores),dtype=np.int64); ranks[order]=np.arange(1,len(scores)+1); pr=np.sort(ranks[np.asarray(pos_rows,dtype=int)])
    mrr=1.0/pr[0]; ap=float(np.mean(np.arange(1,len(pr)+1)/pr)); hit=lambda k:float(pr[0]<=k)
    dcg=sum(1/np.log2(r+1) for r in pr if r<=10); ideal=sum(1/np.log2(i+2) for i in range(min(len(pr),10))); ndcg=float(dcg/ideal) if ideal else 0.0
    return {'mrr':float(mrr),'ap':ap,'ndcg_at_10':ndcg,'hit_at_10':hit(10),'hit_at_20':hit(20),'hit_at_50':hit(50),'best_positive_rank':int(pr[0])}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True,choices=[0,1,2]); ap.add_argument('--root',type=Path,default=OUT); a=ap.parse_args(); f=a.fold; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); d=a.root/f'fold{f}'
 e=pd.read_csv(d/'entries.csv',dtype=str); feat=np.load(d/'bag_features.npy').astype(np.float32); teacher_lat=np.load(d/'teacher_reaction_latents.npy').astype(np.float32); dev=e[e['split']=='dev'].copy(); dev_rows=dev['row'].astype(int).to_numpy(); dev_ids=dev.reaction_id.astype(str).tolist()
 bp=torch.load(d/'adapter.pt',map_location=device,weights_only=False); adapter=BagAdapter().to(device); adapter.load_state_dict(bp['model_state_dict']); adapter.eval()
 with torch.no_grad(): pred=adapter(torch.from_numpy(feat[dev_rows]).to(device)).cpu().numpy().astype(np.float32)
 target=teacher_lat[dev_rows]; cos=np.sum(pred*target,axis=1)
 ck=BASE/f'fold{f}/models/production_seed20260723.pt'; cp=torch.load(ck,map_location=device,weights_only=False); model=TerpeneDualTower(ModelConfig(**cp['model_config'])).to(device); model.load_state_dict(cp['model_state_dict']); model.eval()
 pe=pd.read_csv(PROT/'entries.csv',dtype=str).sort_values('row'); pm=np.load(PROT/'embeddings.npy').astype(np.float32); pids=pe['Entry'].astype(str).tolist(); pidx={x:i for i,x in enumerate(pids)}
 prot=[]
 with torch.no_grad():
  for s in range(0,len(pm),4096): prot.append(model.encode_proteins(torch.from_numpy(pm[s:s+4096]).to(device)).cpu().numpy())
 prot=np.concatenate(prot).astype(np.float32)
 pairs=pd.read_csv(BASE/f'fold{f}/dev_pairs.csv',dtype=str); pos=pairs.groupby('reaction_id').protein_id.apply(list).to_dict(); rows=[]
 for i,rid in enumerate(dev_ids):
  pr=[pidx[x] for x in pos[rid] if x in pidx]; assert pr
  b=metrics(teacher_lat[dev_rows[i]]@prot.T,pr); q=metrics(pred[i]@prot.T,pr)
  rows.append({'reaction_id':rid,'teacher_cosine':float(cos[i]),**{'teacher_'+k:v for k,v in b.items()},**{'adapter_'+k:v for k,v in q.items()}})
 q=pd.DataFrame(rows); q.to_csv(d/'dev_r2e_query_metrics.csv',index=False)
 agg={'fold':f,'n_queries':len(q),'teacher_cosine_mean':float(q.teacher_cosine.mean()),'teacher_cosine_median':float(q.teacher_cosine.median())}
 for prefix in ['teacher','adapter']:
  agg[prefix]={ 'mrr':float(q[f'{prefix}_mrr'].mean()),'map':float(q[f'{prefix}_ap'].mean()),'ndcg_at_10':float(q[f'{prefix}_ndcg_at_10'].mean()),'hit_at_10':float(q[f'{prefix}_hit_at_10'].mean()),'hit_at_20':float(q[f'{prefix}_hit_at_20'].mean()),'hit_at_50':float(q[f'{prefix}_hit_at_50'].mean()),'median_best_positive_rank':float(q[f'{prefix}_best_positive_rank'].median()) }
 agg['retention']={k:agg['adapter'][k]/agg['teacher'][k] if agg['teacher'][k] else None for k in ['mrr','map','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']}
 (d/'evaluation.json').write_text(json.dumps(agg,indent=2),encoding='utf-8'); print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
