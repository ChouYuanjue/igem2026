from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.sparse import load_npz,csr_matrix
from scipy.sparse.csgraph import shortest_path,connected_components
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.geometry.correspondence import correspondence_state
from projects.active.terpene_screening.geometry.multiscale import _self_tuning_scale
CACHE=ROOT/'data/terpene_marts_adaptation';PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4';RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1';RC=ROOT/'data/terpene_multiresolution_reaction_geometry_v2';OUT=ROOT/'results/terpene_mechanistic_tie_refinement_dev_v1';BUD=(3,10,20)

def bools(s):return s.astype(str).str.lower().isin({'1','true','yes'})
def geo(w):
 w=csr_matrix(w,dtype=float).maximum(csr_matrix(w,dtype=float).T).tocsr();w.setdiag(0);w.eliminate_zeros();coo=w.tocoo();u=coo.row<coo.col;raw=np.sqrt(np.maximum(-np.log(np.clip(coo.data[u],1e-300,1.0)),1e-12));ell=float(np.median(raw));edge=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell;g=csr_matrix((edge,w.indices,w.indptr),shape=w.shape);nc,_=connected_components(g,directed=False);assert nc==1;return np.asarray(shortest_path(g,directed=False,unweighted=False))
def center_energy():
 ds=[np.load(RC/'center_transition_wasserstein_distance.npy').astype(float),np.load(RC/'center_token_jaccard_distance.npy').astype(float)]
 av=[np.load(RC/'center_transition_wasserstein_available.npy').astype(bool),np.load(RC/'center_token_jaccard_available.npy').astype(bool)]
 E=np.zeros_like(ds[0]);C=np.zeros_like(ds[0],dtype=int)
 for d,a in zip(ds,av):
  s=_self_tuning_scale(d,a,epsilon=1e-8);den=s[:,None]*s[None,:];v=a[:,None]&a[None,:]&np.isfinite(den)&(den>1e-8);E[v]+=d[v]**2/den[v];C[v]+=1
 out=np.full_like(E,np.inf);ok=C>0;out[ok]=E[ok]/C[ok];np.fill_diagonal(out,0);return out
def groups(train,ri,pi):
 by_r={};by_e={}
 for x in train.itertuples(index=False):
  rr=ri[str(x.rhea_id)];ee=pi[str(x.Entry)];by_r.setdefault(rr,set()).add(ee);by_e.setdefault(ee,set()).add(rr)
 return {k:np.array(sorted(v),int) for k,v in by_r.items()},{k:np.array(sorted(v),int) for k,v in by_e.items()}
def witness_r2e(q,Dr2,De2,by_r):
 ne=De2.shape[0];best=np.full(ne,np.inf);wr=np.full(ne,-1,int)
 for rr,es in by_r.items():
  cost=float(Dr2[q,rr])+np.min(De2[:,es],axis=1);m=cost<best;best[m]=cost[m];wr[m]=rr
 return best,wr
def witness_e2r(q,Dr2,De2,by_e):
 nr=Dr2.shape[0];best=np.full(nr,np.inf);wr=np.full(nr,-1,int)
 for ee,rs in by_e.items():
  cost=float(De2[q,ee])+np.min(Dr2[:,rs],axis=1);m=cost<best;best[m]=cost[m];wr[m]=rs[np.argmin(Dr2[:,rs],axis=1)[m]]
 return best,wr
def rank(primary,secondary,ids,pos):
 p=np.asarray(primary,float);s=np.asarray(secondary,float);ids=np.asarray(ids);base=np.argsort(p,kind='stable');scale=max(1.0,float(np.max(np.abs(p[np.isfinite(p)]))));tol=64*np.finfo(float).eps*scale;ordered=[];i=0
 while i<len(base):
  ref=p[base[i]];j=i+1
  while j<len(base) and abs(p[base[j]]-ref)<=tol:j+=1
  g=base[i:j];loc=np.lexsort((ids[g],s[g]));ordered.extend(g[loc].tolist());i=j
 ranked=ids[ordered].tolist();posi=np.array([k+1 for k,v in enumerate(ranked) if v in pos],int);best=int(posi.min()) if len(posi) else None
 out={'best_positive_rank':best,'reciprocal_rank':1/best if best else 0.0}
 for k in BUD:
  panel=ranked[:k];h=sum(x in pos for x in panel);out[f'hit_at_{k}']=int(h>0);out[f'positive_recall_at_{k}']=h/len(pos) if pos else 0
 return out
def agg(q):
 rows=[]
 for d,g in q.groupby('direction'):
  x={'direction':d,'n_queries':len(g),'mrr':g.reciprocal_rank.mean(),'median_rank':g.best_positive_rank.median()}
  for k in BUD:x[f'hit{k}']=g[f'hit_at_{k}'].mean();x[f'recall{k}']=g[f'positive_recall_at_{k}'].mean()
  rows.append(x)
 return pd.DataFrame(rows)
def main():
 p=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('');r=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('');pairs=pd.read_csv(CACHE/'marts_pair_folds.csv',dtype=str).fillna('');pairs[['protein_fold','reaction_fold']]=pairs[['protein_fold','reaction_fold']].astype(int);pairs['protein_seen']=bools(pairs.protein_seen);pairs['reaction_seen']=bools(pairs.reaction_seen)
 pids=p.protein_id.tolist();rids=r.reaction_id.tolist();pi={x:i for i,x in enumerate(pids)};ri={x:i for i,x in enumerate(rids)}
 Dr=geo(load_npz(RG/'partial_pullback_affinity.npz'));De=geo(load_npz(PG/'partial_pullback_affinity.npz'));Dr2=Dr*Dr;De2=De*De;CE=center_energy()
 rows=[];sp=[]
 for pf in range(5):
  for rf in range(5):
   if not(pf==4 or rf==4):continue
   sid=f'p{pf}_r{rf}';train=pairs[pairs.protein_fold.ne(pf)&pairs.reaction_fold.ne(rf)].drop_duplicates(['rhea_id','Entry']);test=pairs[pairs.protein_fold.eq(pf)&pairs.reaction_fold.eq(rf)&~pairs.protein_seen&~pairs.reaction_seen].drop_duplicates(['rhea_id','Entry']);arr=np.asarray([(ri[x.rhea_id],pi[x.Entry]) for x in train.itertuples()],int);st=correspondence_state(Dr2,De2,arr);byr,bye=groups(train,ri,pi);sp.append({'split_id':sid,'protein_overlap':len(set(train.Entry)&set(test.Entry)),'reaction_overlap':len(set(train.rhea_id)&set(test.rhea_id))})
   for rid,g in test.groupby('rhea_id',sort=True):
    q=ri[rid];_,wr=witness_r2e(q,Dr2,De2,byr);sec=np.array([CE[q,x] if x>=0 else np.inf for x in wr]);rows.append({'split_id':sid,'direction':'reaction_to_enzyme','query_id':rid,**rank(st.defect[q],sec,pids,set(g.Entry.astype(str)))})
   for pid,g in test.groupby('Entry',sort=True):
    q=pi[pid];_,wr=witness_e2r(q,Dr2,De2,bye);sec=np.array([CE[i,x] if x>=0 else np.inf for i,x in enumerate(wr)]);rows.append({'split_id':sid,'direction':'enzyme_to_reaction','query_id':pid,**rank(st.defect[:,q],sec,rids,set(g.rhea_id.astype(str)))})
 q=pd.DataFrame(rows);sp=pd.DataFrame(sp);met=agg(q);OUT.mkdir(parents=True,exist_ok=True);q.to_csv(OUT/'query_metrics.csv',index=False);met.to_csv(OUT/'metrics.csv',index=False)
 summary={'version':'terpene-mechanistic-tie-refinement-dev-v1','partition':'development_only','ranking':'primary: machine-stable correspondence-defect level set; secondary only within that level set: self-tuned reaction-center tangent energy to the shared positive precedent; tertiary candidate id','guarantee':'reaction-center evidence cannot override any strictly smaller correspondence defect and cannot create/rewrite factor-manifold edges','parameter_selection':'none','all_overlaps_zero':bool(((sp.protein_overlap==0)&(sp.reaction_overlap==0)).all()),'metrics':met.to_dict('records')};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(met.to_string(index=False));print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))
if __name__=='__main__':main()
