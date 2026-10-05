from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import gammaln
from scipy.stats import hypergeom
from projects.active.bridge.model.assets import ROOT

R=ROOT/'results/bridge_r2e_edgewise_v13/edge_metrics.csv.gz'
E=ROOT/'results/bridge_e2r_edgewise_v4/edge_metrics.csv.gz'
BR=ROOT/'results/bridge_relation_unseen_edgewise_broad_v1/r2e_edges.csv.gz'
BE=ROOT/'results/bridge_relation_unseen_edgewise_broad_v1/e2r_edges.csv.gz'
OUT=ROOT/'results/bridge_pool_size_scaling_v1'
KS=(1000,10000)

def logcomb(n,k):
 if k<0 or k>n:return -np.inf
 return float(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1))
def expected_rr(N,r,K):
 if K>=N:return 1.0/r
 M=N-1;n=r-1;m=K-1
 # E[1/(X+1)] for X~Hypergeom(M,n,m)
 a=N/K
 lb=logcomb(M-n,m+1)-logcomb(M,m)
 b=0.0 if not np.isfinite(lb) else float(np.exp(lb))
 return max(0.0,min(1.0,(a-b)/r))
def expected_hit(N,r,K,k):
 if K>=N:return float(r<=k)
 M=N-1;n=r-1;m=K-1
 return float(hypergeom.cdf(k-1,M,n,m))
def summarize(df,K):
 vals=[];h10=[];h100=[]
 for N,r in zip(df.candidate_count_filtered.astype(int),df.full_rank.astype(int),strict=True):
  kk=min(int(K),int(N)); vals.append(expected_rr(int(N),int(r),kk));h10.append(expected_hit(int(N),int(r),kk,10));h100.append(expected_hit(int(N),int(r),kk,100))
 return {'pool_size':int(K),'edges':len(df),'expected_mrr':float(np.mean(vals)),'expected_hit10':float(np.mean(h10)),'expected_hit100':float(np.mean(h100))}
def load(final_path,broad_path,direction):
 f=pd.read_csv(final_path,dtype={'protein_id':str,'reaction_id':str});b=pd.read_csv(broad_path,dtype={'protein_id':str,'reaction_id':str})[['protein_id','reaction_id','candidate_count_filtered']]
 d=f.merge(b,on=['protein_id','reaction_id'],how='left',validate='one_to_one');return d[(~d.protein_seen)&(~d.reaction_seen)].copy()
def main():
 OUT.mkdir(parents=True,exist_ok=True);r=load(R,BR,'r2e');e=load(E,BE,'e2r');res={'schema':'bridge-pool-size-scaling-v1','protocol':{'novelty_group':'double-cold only','unit':'edge-wise filtered relation','pool_reduction':'exact expectation under uniform sampling without replacement from the full filtered candidate universe, target always included','purpose':'diagnostic isolation of candidate-pool-size difficulty; not a headline model benchmark','matched_pool_sizes':[1000,10000],'full_pool_sizes':{'r2e':185918,'e2r':11081}},'r2e':[],'e2r':[]}
 for name,d,fullN in [('r2e',r,185918),('e2r',e,11081)]:
  rows=[summarize(d,k) for k in KS];rows.append({'pool_size':'full','edges':len(d),'expected_mrr':float((1/d.full_rank).mean()),'expected_hit10':float((d.full_rank<=10).mean()),'expected_hit100':float((d.full_rank<=100).mean())});res[name]=rows
 (OUT/'summary.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))
if __name__=='__main__':main()
