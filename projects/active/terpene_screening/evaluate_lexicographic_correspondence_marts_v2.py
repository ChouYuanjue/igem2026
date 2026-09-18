from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.product_correspondence_field import correspondence_state
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';OUT=ROOT/'results/terpene_lexicographic_correspondence_dev_v2';BUD=(3,10,20)

def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def geo(w):
 w=csr_matrix(w,dtype=float).maximum(csr_matrix(w,dtype=float).T).tocsr();w.setdiag(0);w.eliminate_zeros();coo=w.tocoo();u=coo.row<coo.col;raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[u],1e-300,1.0)),1e-12));ell=float(np.median(raw));edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((edge,w.indices,w.indptr),shape=w.shape);nc,_=connected_components(g,directed=False);assert nc==1;return np.asarray(shortest_path(g,directed=False,unweighted=False)),ell
def rank_lex(primary,secondary,ids,pos,budgets):
 order=np.lexsort((np.asarray(ids),secondary,primary));ranked=[ids[i] for i in order];positions=np.asarray([k+1 for k,v in enumerate(ranked) if v in pos],int);best=int(positions.min()) if len(positions) else None
 out={'candidate_count':len(ranked),'n_positives':len(pos),'best_positive_rank':best,'reciprocal_rank':1/best if best else 0.0,'mean_positive_rank':float(positions.mean()) if len(positions) else None}
 for k in budgets:
  panel=ranked[:k];hits=sum(x in pos for x in panel);out[f'hits_at_{k}']=hits;out[f'hit_at_{k}']=int(hits>0);out[f'positive_recall_at_{k}']=hits/len(pos) if pos else 0.0
 return out
def aggregate(q):
 rows=[]
 for direction,g in q.groupby('direction'):
  r={'direction':direction,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:r[f'hit{k}']=g[f'hit_at_{k}'].mean();r[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(r)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('');pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 Dr,lr=geo(load_npz(RG/'partial_pullback_affinity.npz'));De,le=geo(load_npz(PG/'partial_pullback_affinity.npz'));Dr2=Dr*Dr;De2=De*De
 rows=[];spl=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);arr=np.asarray([(ri[x.rhea_id],pi[x.Entry]) for x in train.itertuples()],int);st=correspondence_state(Dr2,De2,arr)
   spl.append({'split_id':sid,'train_pairs':len(train),'test_pairs':len(test),'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for rid,g in test.groupby('rhea_id',sort=True):
    rr=ri[rid];rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank_lex(st.defect[rr],st.joint_sq[rr],pids,set(g.Entry.astype(str)),BUD)})
   for pid,g in test.groupby('Entry',sort=True):
    ee=pi[pid];rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank_lex(st.defect[:,ee],st.joint_sq[:,ee],rids,set(g.rhea_id.astype(str)),BUD)})
 q=pd.DataFrame(rows);sp=pd.DataFrame(spl);met=aggregate(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);sp.to_csv(OUT/'split_summary.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-lexicographic-correspondence-dev-v2','partition':'development_only','ranking':'lexicographic minimization of (Delta_Omega, J_Omega), equivalent to Delta+epsilon J as epsilon->0+; candidate ID only breaks exact remaining ties','interpretation':'correspondence defect is primary; shared-precedent distance only resolves its geometric level sets and can never override a smaller defect','parameter_selection':'none','characteristic_lengths':{'reaction':lr,'protein':le},'all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records')}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
