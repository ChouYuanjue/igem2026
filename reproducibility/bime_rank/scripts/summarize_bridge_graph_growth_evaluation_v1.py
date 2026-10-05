from __future__ import annotations
import json
from pathlib import Path
import numpy as np,pandas as pd
from projects.active.bridge.model.assets import ROOT

BENCH=ROOT/'results/bridge_post2023_growth_benchmark_v1'
R2E=ROOT/'results/bridge_post2023_growth_r2e_v1/edge_metrics.csv.gz'
E2R=ROOT/'results/bridge_post2023_growth_e2r_v1/edge_metrics.csv.gz'
BR=ROOT/'results/bridge_post2023_growth_broad_v1/r2e_edges.csv.gz'
BE=ROOT/'results/bridge_post2023_growth_broad_v1/e2r_edges.csv.gz'
OUT=ROOT/'reproducibility/bime_rank/records/BRIDGE_GRAPH_GROWTH_EVALUATION_V1_RESULT.json'

def metric(d,c):
 r=d[c].astype(int).to_numpy();return {'edges':int(len(r)),'mrr':float(np.mean(1/r)),'hit10':float(np.mean(r<=10)),'hit100':float(np.mean(r<=100)),'hit1000':float(np.mean(r<=1000)),'median_rank':float(np.median(r))}
def macro(d,c,group):
 x=d.assign(rr=1/d[c].astype(float),h10=(d[c].astype(int)<=10).astype(float),h100=(d[c].astype(int)<=100).astype(float),h1000=(d[c].astype(int)<=1000).astype(float)).groupby(group).agg(rr=('rr','mean'),h10=('h10','mean'),h100=('h100','mean'),h1000=('h1000','mean'))
 return {'events':int(len(x)),'mrr_macro':float(x.rr.mean()),'hit10_macro':float(x.h10.mean()),'hit100_macro':float(x.h100.mean()),'hit1000_macro':float(x.h1000.mean())}
def main():
 summary=json.load(open(BENCH/'summary.json'));t=pd.read_csv(BENCH/'targets.csv',dtype=str).fillna('')
 r=pd.read_csv(R2E,dtype={'protein_id':str,'reaction_id':str});e=pd.read_csv(E2R,dtype={'protein_id':str,'reaction_id':str});be=pd.read_csv(BE,dtype={'protein_id':str,'reaction_id':str})[['protein_id','reaction_id','rank']].rename(columns={'rank':'broad_rank'})
 key=t[['protein_id','reaction_id','event_type','event_size']];r=r.merge(key,on=['protein_id','reaction_id'],validate='one_to_one');e=e.merge(key,on=['protein_id','reaction_id'],validate='one_to_one').merge(be,on=['protein_id','reaction_id'],validate='one_to_one')
 results={}
 for typ in ['attachment','frontier']:
  rr=r[r.event_type==typ];ee=e[e.event_type==typ]
  results[typ]={'r2e_edge_micro':{'broad':metric(rr,'broad_rank'),'bridge':metric(rr,'full_rank')},'e2r_edge_micro':{'broad':metric(ee,'broad_rank'),'bridge':metric(ee,'full_rank')},'e2r_protein_event_macro':{'broad':macro(ee,'broad_rank','protein_id'),'bridge':macro(ee,'full_rank','protein_id')}}
 # Temporal topology diagnostic.
 root=ROOT/'results/broad_rhea_fair_benchmarks_v1';tr=pd.read_csv(root/'temporal_post2020_protein_cold/train_pairs.csv',dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);te=pd.read_csv(root/'temporal_post2020_protein_cold/test_pairs.csv',dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);rdeg=tr.groupby('reaction_id').protein_id.nunique();sel=te.reaction_id.map(rdeg).astype(float)
 dc=pd.read_csv(root/'temporal_post2020_double_cold/test_pairs.csv',dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);adj={}
 from collections import defaultdict
 a=defaultdict(set)
 for p,q in dc[['protein_id','reaction_id']].itertuples(index=False):a['P:'+p].add('R:'+q);a['R:'+q].add('P:'+p)
 seen=set();comps=[]
 for n in a:
  if n in seen:continue
  stack=[n];seen.add(n);nodes=[];edges=0
  while stack:
   x=stack.pop();nodes.append(x);edges+=len(a[x])
   for y in a[x]:
    if y not in seen:seen.add(y);stack.append(y)
  comps.append((sum(x.startswith('P:') for x in nodes),sum(x.startswith('R:') for x in nodes),edges//2))
 record={'schema':'bridge-graph-growth-evaluation-v1','status':'completed','thesis':'Biochemical database growth is a selective bipartite graph-growth process rather than IID missing-edge sampling. Evaluation therefore separates graph-growth events from candidate-universe size and reports event-macro metrics.','dbtl':[
  {'round':1,'design':'random exact-pair holdout with both endpoints retained','test':'useful sanity check but models fixed-node edge completion rather than database growth','learn':'both-seen pair completion over-represents closed-world behavior and can hide candidate-domain limits'},
  {'round':2,'design':'protein-cold/reaction-cold/double-cold cells','test':'metrics vary non-monotonically across nominal coldness; temporal double-cold can score higher than temporal protein-cold','learn':'endpoint novelty, candidate-pool size, degree distribution, and curation selection are entangled; cold labels alone do not define difficulty'},
  {'round':3,'design':'post-clean2023 protein-arrival events with fixed full candidate universes; Attachment versus Frontier only','test':'event-level results align with expected graph-growth difficulty and expose CAGE closed-domain limits','learn':'use graph-growth benchmark as headline; keep cold cells and pool-size curves as diagnostics'}],
 'graph_growth_evidence':{
  'biochemical_graph':'bipartite protein-reaction graph G_t=(P_t,R_t,E_t)',
  'post2020_attachment_bias':{'selected_old_reaction_train_degree_median':float(sel.median()),'selected_old_reaction_train_degree_mean':float(sel.mean()),'all_train_reaction_degree_median':float(rdeg.median()),'all_train_reaction_degree_mean':float(rdeg.mean())},
  'post2020_double_cold_components':{'components':len(comps),'isolated_pair_fraction':float(sum(c==(1,1,1) for c in comps)/len(comps)),'components_le_5_edges_fraction':float(sum(c[2]<=5 for c in comps)/len(comps)),'largest_component_edge_fraction':float(max(c[2] for c in comps)/len(dc))},
 },
 'post2023_benchmark':summary,
 'sequence_novelty':{'post2023_eligible_proteins':136,'exact_sequence_seen_in_clean2023':1,'exact_sequence_seen_fraction':1/136},
 'results':results,
 'recommended_headline':'E2R protein-arrival event-macro Attachment vs Frontier; R2E edge-micro is a reverse-retrieval companion, not the event-definition axis.',
 'limitations':['UniProt creation date timestamps protein entry creation, not the exact curation time of each protein-reaction edge.','Current local data cannot directly reconstruct old-old relation additions by publication/curation timestamp; publication-level Rhea provenance is the next refinement.'],
 'reproduction':{'build':'reproducibility/bime_rank/scripts/build_bridge_post2023_growth_benchmark_v1.py','summarize':'reproducibility/bime_rank/scripts/summarize_bridge_graph_growth_evaluation_v1.py','r2e_evaluator':'reproducibility/bime_rank/scripts/evaluate_bridge_r2e_edgewise_v13.py','e2r_evaluator':'reproducibility/bime_rank/scripts/evaluate_bridge_e2r_edgewise_v4.py'}}
 OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
