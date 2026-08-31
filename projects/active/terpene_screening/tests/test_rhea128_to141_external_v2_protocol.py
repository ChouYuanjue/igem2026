import json
from pathlib import Path
import pandas as pd
from projects.active.terpene_screening.prepare_rhea_snapshot_delta_external_benchmark_v2 import build_unambiguous_protein_alias_map,map_to_frozen_universe,build
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json'
I=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V1_INVALIDATION.json'
def test_v1_invalidated_pre_reveal_and_v2_uses_rhea_id_alias_semantics():
 i=json.loads(I.read_text()); p=json.loads(P.read_text())
 assert i['target_labels_read_before_invalidation'] is False and i['target_model_scores_read_before_invalidation'] is False
 assert p['status']=='frozen_before_release141_association_extraction_and_model_performance'
 assert 'RHEA_ID + ID' in p['source_snapshots']['pair_identity']
 assert p['minimum_support_rule']['min_query_reactions']==50 and p['minimum_support_rule']['min_test_pairs']==200
 assert p['selection_without_model_scores']['no_model_score_or_rank_may_enter_query_selection'] is True
def test_alias_mapping_is_unambiguous_and_semicolon_based():
 meta=pd.DataFrame([{'protein_id':'PNEW','canonical_accession':'PNEW','aliases':'PNEW;POLD'},{'protein_id':'Q','canonical_accession':'Q','aliases':'Q'}])
 mapping,audit=build_unambiguous_protein_alias_map(meta)
 assert mapping['POLD']=='PNEW' and audit['nontrivial_alias_rows']==1 and audit['ambiguous_alias_keys']==0
 source=pd.DataFrame([{'protein_source_id':'POLD','reaction_source_id':'RHEA:1'}])
 mapped,_=map_to_frozen_universe(source,mapping,{'RHEA:1'})
 assert mapped.to_dict('records')==[{'protein_id':'PNEW','reaction_id':'RHEA:1'}]
def test_builder_uses_mapped_release_delta_and_complete_cold_positive_support():
 meta=pd.DataFrame([{'protein_id':'P1','canonical_accession':'P1','aliases':'P1;OLD1'},{'protein_id':'P2','canonical_accession':'P2','aliases':'P2;OLD2'},{'protein_id':'SEEN','canonical_accession':'SEEN','aliases':'SEEN'}])
 compact=pd.DataFrame([{'protein_source_id':'SEEN','reaction_source_id':'RHEA:9'}]); clean=pd.DataFrame([{'protein_id':'SEEN','reaction_id':'RHEA:9'}])
 old=pd.DataFrame([{'protein_source_id':'OLD1','reaction_source_id':'RHEA:2'}])
 new=pd.DataFrame([{'protein_source_id':'OLD1','reaction_source_id':'RHEA:2'},{'protein_source_id':'OLD2','reaction_source_id':'RHEA:3'},{'protein_source_id':'P1','reaction_source_id':'RHEA:3'}])
 _,test,trigger,audit=build(old,new,compact,clean,meta,{'RHEA:2','RHEA:3','RHEA:9'})
 assert set(map(tuple,test[['protein_id','reaction_id']].itertuples(index=False,name=None)))=={('P1','RHEA:3'),('P2','RHEA:3')}
 assert set(map(tuple,trigger[['protein_id','reaction_id']].itertuples(index=False,name=None)))=={('P1','RHEA:3'),('P2','RHEA:3')}
 assert audit['clean2023_exact_reconstruction'] is True
