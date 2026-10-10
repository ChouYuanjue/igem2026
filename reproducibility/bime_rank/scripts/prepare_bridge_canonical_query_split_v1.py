"""One canonical heldout BRIDGE R2E/E2R direct/balanced table protocol.

TRAIN: clean2023 documented edges, fixed 218,537.
E2R development: distinct protein queries, 3,900 learn + 1,316 validate.
TEST: fixed 21,505 relations/15,751 protein queries/1,857 reaction queries.
Both directions use the same pair IDs, filtered entity ranks and graph-degree
novelty macro averaging. R2E reuses existing exact ranking files. E2R uses
the accepted frozen C2 predictions and scientifically rerun ablation ranks.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced
from reproducibility.bime_rank.scripts import analyze_bridge_difficulty_standardized_v3 as v3

SPLIT=ROOT/'results/bridge_gate_split_v2'
DEV=ROOT/'results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz'
C2=ROOT/'results/bridge_e2r_inductive_c2_outer_v1/single_heldout_edge_ranks.csv.gz'
ABL=ROOT/'results/bridge_e2r_inductive_ablation_v1/edge_ranks.csv.gz'
R2E_CAGE=ROOT/'results/bridge_layered_v4_cage_edgewise/r2e_edge_metrics.csv.gz'
R2E_NEW=ROOT/'results/bridge_r2e_single_validation_release_v1/edge_metrics.csv.gz'
R2E_GATE_NEW=ROOT/'results/bridge_r2e_single_validation_cage_gate_v1/r2e_edge_metrics.csv.gz'
HYBRID=ROOT/'results/bridge_e2r_inductive_cage_gate_hybrid_v1/edge_ranks.csv.gz'
OUT=ROOT/'results/bridge_canonical_query_split_v1'
METHODS={
 'EnzymeCAGE':'enzymecage_rank',
 'Broad Retrieval':'broad_rank',
 'Broad Retrieval + CAGE Reranking':'broad_cage_rank',
 'CAGE Gate + BRIDGE Reranking':'cage_gate_bridge_rank',
 'BRIDGE':'full_rank',
 'w/o Functional':'minus_functional_rank',
 'w/o Structure/Mechanism':'minus_structure_mechanism_rank',
 'w/o Long-term Relation Context':'minus_relational_memory_rank',
 'w/o Family/Domain':'minus_family_domain_rank',
}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()

def main():
 manifest=json.loads((SPLIT/'manifest.json').read_text())
 cohort=pd.read_csv(SPLIT/'evaluation_pairs.csv.gz',dtype={'protein_id':str,'reaction_id':str})
 assert manifest['parts']['evaluation']['edges']==len(cohort)==21505
 assert not cohort.duplicated(v3.KEYS).any()
 dev=pd.read_csv(DEV,dtype=str)
 assert len(dev)==5216
 assert dev.groupby('partition').size().to_dict()=={'learn':3900,'validation':1316}
 assert not dev.duplicated(v3.KEYS).any()
 assert not (set(cohort.protein_id)&set(dev.protein_id))
 assert not (set(zip(cohort.protein_id,cohort.reaction_id)) & set(zip(dev.protein_id,dev.reaction_id)))
 known=pd.read_csv(v3.TRAIN,dtype=str)
 assert len(known)==218537
 assert not known.duplicated(v3.KEYS).any()
 assert not (set(zip(cohort.protein_id,cohort.reaction_id)) & set(zip(known.protein_id,known.reaction_id)))
 dp=known.protein_id.value_counts().to_dict()
 dr=known.reaction_id.value_counts().to_dict()
 frozen_c2=pd.read_csv(C2,dtype={'protein_id':str,'reaction_id':str})
 assert len(frozen_c2)==21505
 assert frozen_c2[v3.KEYS].equals(cohort[v3.KEYS])
 ablation=pd.read_csv(ABL,dtype={'protein_id':str,'reaction_id':str})
 assert len(ablation)==21505 and not ablation.duplicated(v3.KEYS).any()
 assert ablation[v3.KEYS].equals(cohort[v3.KEYS])
 assert np.array_equal(ablation.full_rank.to_numpy(np.int64),frozen_c2.inductive_rank.to_numpy(np.int64))
 hybrid=None
 if HYBRID.exists():
  hybrid=pd.read_csv(HYBRID,dtype={'protein_id':str,'reaction_id':str})
  assert len(hybrid)==21505 and hybrid[v3.KEYS].equals(cohort[v3.KEYS])
 OUT.mkdir(parents=True,exist_ok=True)
 dataset={}
 for direction in ('r2e','e2r'):
  old,specs,_=v3.load_direction(direction)
  frame=cohort.merge(old,on=v3.KEYS,how='left',validate='one_to_one',sort=False)
  assert len(frame)==len(cohort) and frame['full_rank'].notna().all()
  if direction=='r2e':
   revised=pd.read_csv(R2E_NEW,dtype={'protein_id':str,'reaction_id':str})
   revised_cols=['full_rank','minus_functional_rank','minus_structure_mechanism_rank','minus_relational_memory_rank','minus_family_domain_rank']
   assert len(revised)==23773 and not revised.duplicated(v3.KEYS).any()
   for name in ['broad_rank']:
    verify=frame[v3.KEYS+[name]].merge(revised[v3.KEYS+[name]],on=v3.KEYS,validate='one_to_one',suffixes=('_original','_new'))
    assert verify[name+'_original'].equals(verify[name+'_new']), 'Frozen Broad rank changed'
   frame=frame.drop(columns=revised_cols).merge(revised[v3.KEYS+revised_cols],on=v3.KEYS,validate='one_to_one')
   hybrid_r2e=pd.read_csv(R2E_GATE_NEW,dtype={'protein_id':str,'reaction_id':str})
   assert len(hybrid_r2e)==23773 and not hybrid_r2e.duplicated(v3.KEYS).any()
   frame=frame.drop(columns=['cage_gate_bridge_rank']).merge(
    hybrid_r2e[v3.KEYS+['cage_gate_bridge_rank']],on=v3.KEYS,validate='one_to_one')
   native=pd.read_csv(R2E_CAGE,dtype={'protein_id':str,'reaction_id':str})
   assert len(native)==23773 and not native.duplicated(v3.KEYS).any()
   frame=frame.merge(native[v3.KEYS+['enzymecage_rank','broad_cage_rank']],
                     on=v3.KEYS,validate='one_to_one')
   assert frame[['enzymecage_rank','broad_cage_rank']].notna().all().all()
  else:
   cols=['full_rank','minus_functional_rank','minus_structure_mechanism_rank',
         'minus_relational_memory_rank','minus_family_domain_rank']
   frame=frame.drop(columns=cols)
   frame=frame.merge(ablation[v3.KEYS+cols],on=v3.KEYS,validate='one_to_one',sort=False)
   if hybrid is not None:
    frame=frame.drop(columns=['cage_gate_bridge_rank'])
    frame=frame.merge(hybrid[v3.KEYS+['cage_gate_bridge_rank']],
                      on=v3.KEYS,validate='one_to_one',sort=False)
  frame=v3.annotate(frame,dp,dr)
  assert len(frame)==21505
  assert frame[v3.KEYS].equals(cohort[v3.KEYS])
  assert all(frame[c].gt(0).all() for c in METHODS.values() if c in frame and
            (direction=='r2e' or c!='cage_gate_bridge_rank' or hybrid is not None))
  out_method={}
  for mode in ('direct','balanced'):
   rows=[]
   for name,c in METHODS.items():
    if direction=='e2r' and c=='cage_gate_bridge_rank' and hybrid is None:
     continue
    if c not in frame:raise RuntimeError('Missing score '+direction+' '+c)
    result=out_method.setdefault(name,balanced(frame,c))
    value=result[mode]
    rows.append({'Method':name,'MRR':f'{value["mrr"]:.5f}',
                 'Hit@3':f'{100*value["hit3"]:.2f}%',
                 'Hit@10':f'{100*value["hit10"]:.2f}%',
                 'Hit@100':f'{100*value["hit100"]:.2f}%'})
   pd.DataFrame(rows).to_csv(OUT/(direction+'_'+mode+'.csv'),index=False)
   print(direction.upper(),mode.upper(),pd.DataFrame(rows).to_string(index=False),flush=True)
  dataset[direction]={
   'same_heldout_positive_pairs':len(frame),
   'unique_queries':int(frame.reaction_id.nunique() if direction=='r2e' else frame.protein_id.nunique()),
   'novelty_edge_counts':frame.novelty.value_counts().to_dict(),
   'methods':out_method,
   'removed_unverified_comparison_row': (
      [] if direction=='r2e' or hybrid is not None else
      ['CAGE Gate + BRIDGE Reranking requires independent current C2 hybrid rerank']
   ),
  }
 result={
  'schema':'BRIDGE_CANONICAL_21505_QUERY_SPLIT_MAIN_AND_ABLATION_V1',
  'core_training_edges':218537,
  'development':{'E2R_source_expansion_learn_edges':3900,'E2R_validation_edges':1316,
                 'E2R_total_dev_edges':5216,'E2R_distinct_protein_queries':int(dev.protein_id.nunique()),
                 'relation_pairs_overlap_final_test':0,'protein_queries_overlap_final_test':0},
  'main_test':{
   'strict_train_relation_unseen_edges':21505,
   'protein_queries':15751,
   'reaction_queries':int(cohort.reaction_id.nunique()),
   'E2R_protein_query_disjoint_from_dev':True,
   'R2E_reaction_query_disjoint_from_dev':'not guaranteed by a protein-keyed split',
   'source_split_sha256':digest(SPLIT/'manifest.json'),
   'test_pairs_sha256':digest(SPLIT/'evaluation_pairs.csv.gz'),
   'current_relation_gate_exact_positive_test_overlap':0,
   'test_as_a_metric_selection_set':True,
   'test_result_inspected_before_model_iteration':True,
  },
  'metric_definition':'Filtered edge rank; direct = mean; balanced = equal four novelty classes, equal degree strata within each; NO chance baseline correction; R2E 185918 complete proteins E2R 11081 complete reactions',
  'recomputed_E2R_four_group_ablation':True,
  'R2E_base_and_four_group_ablation_reuses_original_per_edge_ranks_without_model_rescore':False,
  'R2E_relation_gate_fit_on_same_5216_validation_relations':True,
  'R2E_functional_5216_validation_selection_beta':1.25,
  'R2E_legacy_16108_supervision_excluded_from_final_gate':True,
  'scientific_status':'post-hoc exploratory: model revised after initial final test inspection',
  'external_enzymeCAGE_results_unaltered':True,
  'direction':dataset,
  'input_hashes':{
   'dev':digest(DEV),'training':digest(v3.TRAIN),
   'C2_frozen_full':digest(C2),'C2_four_component_ablation':digest(ABL),
   'R2E_native_CAGE':digest(R2E_CAGE),
   'R2E_refitted_four_evidence_rank_cache':digest(R2E_NEW),
   'R2E_refitted_CAGE_gate_BRIDGE_rank_cache':digest(R2E_GATE_NEW),
   'C2_CAGE_hybrid':digest(HYBRID) if hybrid is not None else None,
  }
 }
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('CANONICAL_HELDOUT_MAIN_TABLES_CONFIRMED',json.dumps({a:{
   'test_relations':b['same_heldout_positive_pairs'],'queries':b['unique_queries'],
   'BRIDGE':{k:round(v,6) for k,v in b['methods']['BRIDGE']['direct'].items()},
   'BALANCED':{k:round(v,6) for k,v in b['methods']['BRIDGE']['balanced'].items()}
 } for a,b in dataset.items()}),flush=True)
if __name__=='__main__':main()
