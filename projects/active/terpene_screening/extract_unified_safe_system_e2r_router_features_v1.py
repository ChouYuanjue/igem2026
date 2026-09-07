from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_unified_safe_system_e2r_full_reaction_v1 import BASE,CAND,load_bundle,encode_rows
OUT=ROOT/'results/unified_safe_system_v1/e2r_router_v1'
DIFF=ROOT/'results/cleanroom_internal_full_candidate_difficulty_v1'
def top_features(scores:np.ndarray,ids:list[str]):
 order=np.lexsort((np.asarray(ids,dtype=object),-scores))[:10]; vals=scores[order]; return int(order[0]),set(order.tolist()),float(vals[0]),float(vals[0]-vals[1]),float(vals[0]-vals[-1])
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True,choices=[0,1,2]); a=ap.parse_args(); f=a.fold; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 bd=BASE/f'fold{f}'; cd=CAND/f'fold{f}'; dev=pd.read_csv(bd/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates(); assert dev.equals(pd.read_csv(cd/'dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates())
 bundles=[]
 for root in [BASE,CAND]:
  d,s,m,pf,pids,rf,rids=load_bundle(root,f,device); bundles.append((d,s,m,pf,pids,rf,rids))
 common=sorted(set(bundles[0][6])&set(bundles[1][6])); assert len(common)==11081
 query_ids=sorted(dev.protein_id.unique()); embs=[]
 for d,s,m,pf,pids,rf,rids in bundles:
  pi={x:i for i,x in enumerate(pids)}; ri={x:i for i,x in enumerate(rids)}; pe=encode_rows(m,'encode_proteins',pf,[pi[x] for x in query_ids],device); re=encode_rows(m,'encode_reactions',rf,[ri[x] for x in common],device); embs.append((pe,re))
 ps=pd.read_csv(DIFF/f'clean2023_internal_double_cold_fold{f}/protein_slices.csv',dtype=str); pmeta=ps.set_index('protein_id'); rows=[]; batch=256
 rt=[torch.from_numpy(x[1]).to(device) for x in embs]
 with torch.no_grad():
  for s in range(0,len(query_ids),batch):
   score=[(torch.from_numpy(x[0][s:s+batch]).to(device)@r.T).cpu().numpy() for x,r in zip(embs,rt)]
   for j,q in enumerate(query_ids[s:s+batch]):
    b=top_features(score[0][j],common); c=top_features(score[1][j],common); meta=pmeta.loc[q]; ident=float(meta.mmseqs_fident) if str(meta.protein_identity_bucket)!='no_hit' else 0.0
    rows.append({'fold':f,'query_id':q,'baseline_top1_score':b[2],'baseline_top1_top2_margin':b[3],'baseline_top1_top10_margin':b[4],'candidate_top1_score':c[2],'candidate_top1_top2_margin':c[3],'candidate_top1_top10_margin':c[4],'top1_agreement':int(b[0]==c[0]),'top10_jaccard':len(b[1]&c[1])/len(b[1]|c[1]),'protein_identity':ident,'protein_no_hit':int(str(meta.protein_identity_bucket)=='no_hit')})
 out=OUT/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/'router_features.csv',index=False); print(json.dumps({'fold':f,'queries':len(rows),'candidate_count':len(common),'labels_read_for_feature_extraction':False},indent=2))
if __name__=='__main__': main()
