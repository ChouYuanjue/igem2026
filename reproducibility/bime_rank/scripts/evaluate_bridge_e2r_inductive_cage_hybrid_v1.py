"""Fixed native EnzymeCAGE candidate gate scored by released E2R C2 BRIDGE.

For each query use unchanged historically defined native CAGE pool and the
same trained gate/C2 score for its candidates. Candidate head membership,
training known positives, test labels and all calibration coefficients remain
unchanged. Rank target only within the gated (training-known masked) set.
"""
from __future__ import annotations
import json,pickle,os,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from projects.active.bridge.runtime.e2r_query_gate import FrozenE2RSurface,predict_joint_route
from projects.active.bridge.runtime.inductive_relation import InductiveE2RRelation,InductiveRelationConfig
from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced
from reproducibility.bime_rank.scripts.analyze_bridge_difficulty_standardized_v3 import annotate
ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'results/bridge_e2r_inductive_c2_outer_v1/single_heldout_edge_ranks.csv.gz'
MEM=ROOT/'results/bridge_layered_v4_e2r_cage/pair_membership.csv.gz'
BASE=ROOT/'results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz'
GATE=ROOT/'projects/active/bridge/release/runtime/final_bridge_v1/e2r_joint_query_v4.production.pkl'
ASSET=ROOT/'projects/active/bridge/release/runtime/final_bridge_v1/e2r_inductive_relation.production.json'
OUT=ROOT/'results/bridge_e2r_inductive_cage_gate_hybrid_v1'
PARTIAL=OUT/'partial.csv.gz'
FINAL=OUT/'edge_ranks.csv.gz'
def main():
 f=pd.read_csv(SRC,dtype={'protein_id':str,'reaction_id':str})
 base=pd.read_csv(BASE,dtype={'protein_id':str,'reaction_id':str})
 f=f.merge(base[['protein_id','reaction_id','candidate_count_filtered']],
           on=['protein_id','reaction_id'],validate='one_to_one')
 assert len(f)==21505
 m=pd.read_csv(MEM,dtype=str,usecols=['protein_id','reaction_id','native_pool'])
 m=m[m.native_pool.str.lower().eq('true')].drop_duplicates(['protein_id','reaction_id'])
 groups={q:list(g.reaction_id) for q,g in m.groupby('protein_id',sort=False)}
 target_groups=f.groupby('protein_id',sort=True)
 keys=list(target_groups.groups)
 OUT.mkdir(exist_ok=True,parents=True)
 if FINAL.exists():raise RuntimeError('Fixed EnzymeCAGE+BRIDGE test hybrid already finalized')
 rt=FinalBridgeRuntime(device='cuda')
 sf=FrozenE2RSurface(rt)
 with GATE.open('rb') as fp:router=pickle.load(fp)
 asset=json.loads(ASSET.read_text())
 c=asset['neighborhood']
 w=asset['coefficients']
 model=InductiveE2RRelation(rt,InductiveRelationConfig(
    observed_coefficient=w['observed_links'],inferred_coefficient=w['reaction_analogy'],
    nearest_proteins=c['protein_neighbors'],nearest_reactions=c['reaction_neighbors'],
    reaction_neighbor_pool=c['reaction_retrieval_pool'],
    protein_attention_temperature=c['protein_temperature'],
    reaction_attention_temperature=c['reaction_temperature'],
    prior_normalizer=c['observation_prior_factor'],
    reaction_min_cosine=c['reaction_confidence_floor']))
 data={}
 if PARTIAL.exists():
  d=pd.read_csv(PARTIAL,dtype={'protein_id':str,'reaction_id':str})
  for q,g in d.groupby('protein_id',sort=True):
   assert g.reaction_id.tolist()==target_groups.get_group(q).reaction_id.tolist()
   data[q]=g
  print('RESUMED_GATE_HYBRID',len(data),'queries',flush=True)
 t=time.time()
 for i,q in enumerate(keys,1):
  if q in data:continue
  g=target_groups.get_group(q)
  gate_ids=groups.get(q,[])
  target_ids=list(g.reaction_id)
  n=g.candidate_count_filtered.to_numpy(np.int64)
  ranks=n.copy()
  if gate_ids:
   with torch.no_grad():
    s=sf.score(q)
    _,a,_=predict_joint_route(router,s['features'],s)
    top=s['top']
    score=(s['core'][top]+a[0]*s['functional'][top]+a[1]*s['structure'][top]+
           a[2]*s['relation'][top]+
           model.correction(q,top,float(np.std(s['broad']))))
    local=np.lexsort((sf.lex[top],-score))
    full=np.concatenate((top[local],s['order'][1000:]))
   # Native CAGE pool membership is frozen, even if it includes recorded
   # training reactions. Clean2023 associations are filtered, consistently
   # with the primary E2R test.
   allowed={int(x) for x in full}
   native={rt.index.reaction_index[r] for r in gate_ids
           if r in rt.index.reaction_index}
   native=native & allowed
   pool=[int(x) for x in full if int(x) in native]
   byrow={p:k+1 for k,p in enumerate(pool)}
   targets=[rt.index.reaction_index[x] for x in target_ids]
   pos=[byrow.get(int(x)) for x in targets]
   for j,raw in enumerate(pos):
    if raw is not None:
     ranks[j]=raw-sum(z is not None and z<raw for k,z in enumerate(pos) if k!=j)
  data[q]=g[['protein_id','reaction_id']].assign(cage_gate_bridge_rank=ranks.astype(np.int64))
  if i%250==0 or i==len(keys):
   df=pd.concat(data.values(),ignore_index=True)
   tmp=OUT/'partial.tmp.csv.gz';df.to_csv(tmp,index=False);os.replace(tmp,PARTIAL)
   print('CAGE_C2_HYBRID_PROGRESS',i,len(keys),'edges',len(df),'elapsed_s',round(time.time()-t,1),flush=True)
 all=pd.concat(data.values(),ignore_index=True)
 all=all.merge(f[['protein_id','reaction_id','novelty','difficulty_stratum']].assign(_position=np.arange(len(f))),on=['protein_id','reaction_id'],validate='one_to_one').sort_values('_position').drop(columns='_position')
 assert len(all)==21505 and all[['protein_id','reaction_id']].equals(f[['protein_id','reaction_id']])
 assert all.cage_gate_bridge_rank.ge(1).all()
 all.to_csv(FINAL,index=False)
 metrics=balanced(all,'cage_gate_bridge_rank')
 (OUT/'summary.json').write_text(json.dumps({
  'schema':'BRIDGE_E2R_C2_FROZEN_CAGE_NATIVE_GATE_RERANK_V1',
  'test_edges':len(all),'protein_queries':len(keys),
  'gate':'fixed native EnzymeCAGE selected reaction candidates; drop training known positives',
  'ranker':'current frozen BRIDGE C2 global filtered ordering restricted to the fixed CAGE native gate',
  'no_training_or_test_selection':True,'metrics':metrics
 },indent=2)+'\n')
 print('E2R_CAGE_GATE_CURRENT_BRIDGE_HYBRID',json.dumps(metrics,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
