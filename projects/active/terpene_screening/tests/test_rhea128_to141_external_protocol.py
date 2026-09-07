import json
from pathlib import Path
import pandas as pd
from projects.active.terpene_screening.prepare_rhea_snapshot_delta_external_benchmark import build
from projects.active.terpene_screening.model_capability_registry import scenario_map
ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V1.json'

def test_external_protocol_is_frozen_before_141_labels_and_performance():
    p=json.loads(P.read_text())
    assert p['status']=='frozen_before_release141_label_download_and_model_performance'
    assert p['source_snapshots']['training_boundary']['rhea_release']==128
    assert p['source_snapshots']['external_snapshot']['rhea_release']==141
    assert p['minimum_support_rule']['min_query_reactions']==50
    assert p['minimum_support_rule']['min_test_pairs']==200
    assert p['selection_without_model_scores']['no_model_score_or_rank_may_enter_query_selection'] is True
    assert p['fixed_model_training_before_external_scoring']['candidate']['only_trainable_parameter']=='aux_to_hidden.weight'

def test_builder_requires_query_complete_supported_positive_set_to_be_protein_cold():
    old=pd.DataFrame([{'protein_id':'OLDP','reaction_id':'RHEA:1'}])
    new=pd.DataFrame([
      {'protein_id':'PNEW','reaction_id':'RHEA:2'},
      {'protein_id':'PSEEN','reaction_id':'RHEA:2'},
      {'protein_id':'PNEW2','reaction_id':'RHEA:3'},
      {'protein_id':'PNEW3','reaction_id':'RHEA:3'},
    ])
    clean=pd.DataFrame([{'protein_id':'PSEEN','reaction_id':'RHEA:9'}])
    train,test,trigger,audit=build(old,new,clean,{'PNEW','PSEEN','PNEW2','PNEW3'},{'RHEA:2','RHEA:3','RHEA:9'})
    assert set(test.reaction_id)=={'RHEA:3'}
    assert set(test.protein_id)=={'PNEW2','PNEW3'}
    assert audit['train_test_protein_overlap']==0 and audit['train_test_reaction_overlap']==0

def test_fresh_snapshot_scenario_is_registered():
    s=scenario_map()['rhea128_to141_sprot_strict_double_cold']
    assert s.strict_clean is True and s.confirmatory is True
    assert s.directions==('reaction_to_enzyme',)
