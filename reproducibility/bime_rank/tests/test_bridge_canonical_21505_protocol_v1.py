"""Published BRIDGE canonical cohort and four-component evidence table contract.

Pure provenance/format checks; source-only official iGEM Git checkout runs
these without large scientific data or GPU availability.
"""
import json
import hashlib
import pickle
import numpy as np
from pathlib import Path

REC=Path(__file__).resolve().parents[1]/'records'/'BRIDGE_CANONICAL_21505_RESULT.json'
C2=Path(__file__).resolve().parents[1]/'records'/'BRIDGE_E2R_INDUCTIVE_BIPARTITE_V1_RESULT.json'
EXPECTED={
    'both_seen':180,'protein_cold':7496,
    'reaction_cold':1070,'double_cold':12759,
}
REQUIRED_METHODS={
    'EnzymeCAGE','Broad Retrieval',
    'Broad Retrieval + CAGE Reranking',
    'CAGE Gate + BRIDGE Reranking',
    'BRIDGE','w/o Functional',
    'w/o Structure/Mechanism','w/o Long-term Relation Context',
    'w/o Family/Domain',
}

def load():
    return json.loads(REC.read_text())

def test_all_main_tables_use_one_heldout_relation_cohort():
    obj=load()
    assert obj['schema']=='BRIDGE_CANONICAL_21505_QUERY_SPLIT_MAIN_AND_ABLATION_V1'
    cohort=obj['main_test']
    assert cohort['strict_train_relation_unseen_edges']==21505
    assert cohort['protein_queries']==15751
    assert cohort['reaction_queries']==1857
    assert cohort['E2R_protein_query_disjoint_from_dev'] is True
    assert cohort['R2E_reaction_query_disjoint_from_dev']=='not guaranteed by a protein-keyed split'
    assert obj['core_training_edges']==218537
    dev=obj['development']
    assert dev['E2R_source_expansion_learn_edges']==3900
    assert dev['E2R_validation_edges']==1316
    assert dev['E2R_total_dev_edges']==5216
    assert dev['protein_queries_overlap_final_test']==0
    assert dev['relation_pairs_overlap_final_test']==0

def test_every_published_comparator_has_every_metric():
    obj=load()
    for direction,queries in [('r2e',1857),('e2r',15751)]:
        d=obj['direction'][direction]
        assert d['same_heldout_positive_pairs']==21505
        assert d['unique_queries']==queries
        assert d['novelty_edge_counts']==EXPECTED
        assert set(d['methods'])==REQUIRED_METHODS
        for score in d['methods'].values():
            for mode in ('direct','balanced'):
                s=score[mode]
                assert 0<=s['hit3']<=s['hit10']<=s['hit100']<=1
                assert 0<=s['mrr']<=1

def test_deployed_e2r_full_matches_original_one_shot_without_rescoring():
    record=load()
    published=json.loads(C2.read_text())
    full=record['direction']['e2r']['methods']['BRIDGE']
    for mode in ('direct','balanced'):
        for metric,value in full[mode].items():
            assert abs(value-published['metrics'][mode][metric])<1e-11

def test_e2r_ablation_has_proper_missing_expert_neutrality_and_relation_contribution():
    obj=load()
    e=obj['direction']['e2r']['methods']
    assert e['BRIDGE']==e['w/o Family/Domain']
    assert e['BRIDGE']['direct']['hit10']>e['w/o Long-term Relation Context']['direct']['hit10']
    assert e['BRIDGE']['direct']['hit10']>e['w/o Functional']['direct']['hit10']
    assert e['BRIDGE']['balanced']['hit10']>e['w/o Structure/Mechanism']['balanced']['hit10']
    assert obj['recomputed_E2R_four_group_ablation'] is True
    assert obj['R2E_base_and_four_group_ablation_reuses_original_per_edge_ranks_without_model_rescore'] is False
    assert obj['R2E_relation_gate_fit_on_same_5216_validation_relations'] is True
    assert obj['R2E_functional_5216_validation_selection_beta']==1.25
    assert obj['R2E_legacy_16108_supervision_excluded_from_final_gate'] is True

def test_chance_baseline_is_not_applied_to_headline_balance():
    definition=load()['metric_definition'].lower()
    assert 'no chance baseline correction' in definition
    assert 'equal four novelty classes' in definition


def test_tps_semantic_slice_uses_the_same_heldout_pairs_and_cached_r2e_ranks():
    record=json.loads((REC.parent/'BRIDGE_TPS_CANONICAL_QUERY_SLICE_V1_RESULT.json').read_text())
    assert record['formal_main_test_pairs']==21505
    assert record['tps_semantic_active_reaction_queries']==10
    assert record['tps_semantic_active_positive_edges']==12
    assert record['all_r2e_ranks_reused_from_frozen_cache'] is True
    assert record['measurements']['bridge']['query_hit10']==.1
    assert record['measurements']['without_family_domain']['query_hit10']==0


def test_r2e_frozen_updated_model_functional_expert_contributes_on_all_headline_measures():
    d=load()['direction']['r2e']['methods']
    assert d['BRIDGE']['direct']['mrr']>d['w/o Functional']['direct']['mrr']
    assert d['BRIDGE']['direct']['hit10']>d['w/o Functional']['direct']['hit10']
    assert d['BRIDGE']['direct']['hit100']>d['w/o Functional']['direct']['hit100']
    assert d['BRIDGE']['balanced']['mrr']>d['w/o Functional']['balanced']['mrr']
    assert d['BRIDGE']['balanced']['hit10']>d['w/o Functional']['balanced']['hit10']
    assert d['BRIDGE']['balanced']['hit100']>d['w/o Functional']['balanced']['hit100']


def test_one_validation_corpus_and_formal_test_provenance_are_honest():
    protocol=json.loads((REC.parent/'BRIDGE_COMMON_5216_VALIDATION_RESULT.json').read_text())
    assert protocol['unified_validation']['positive_pairs']==5216
    assert protocol['unified_validation']['enzyme_queries']==3454
    assert protocol['unified_validation']['reaction_queries']==990
    assert protocol['formal_final_test']['positive_pairs']==21505
    assert protocol['formal_final_test']['shared_exact_positive_pairs_with_validation']==0
    assert protocol['formal_final_test']['shared_protein_queries_with_validation']==0
    assert protocol['formal_final_test']['shared_reaction_queries_with_validation']==516
    assert protocol['superseded_legacy_gate_provenance']['legacy_exact_positive_overlap_with_final_test']==14523
    assert protocol['superseded_legacy_gate_provenance']['legacy_source_disallowed_for_active_R2E_and_fallback_E2R_model_fitting'] is True
    assert 'post-hoc' in protocol['formal_final_test']['scientific_status']
    assert load()['main_test']['test_as_a_metric_selection_set'] is True
    assert load()['main_test']['current_relation_gate_exact_positive_test_overlap']==0


def test_both_production_relation_gate_branches_only_use_unified_validation():
    root=Path(__file__).resolve().parents[3]/'projects/active/bridge/release/runtime/final_bridge_v1'
    meta=json.loads((root/'manifest.json').read_text())
    val=meta['production_validation']['adaptive_relation_gate']
    result=json.loads((REC.parent/'BRIDGE_COMMON_5216_VALIDATION_RESULT.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    a=root/'adaptive_relation_gate.production.pkl'
    r=root/'r2e_shared_validation_gate.production.pkl'
    e=root/'e2r_shared_validation_fallback_gate.production.pkl'
    assert sha(a)==val['sha256']==result['production_checksums']['combined_recipient_relation_gate']
    assert sha(r)==val['r2e_standalone_gate_sha256']
    assert sha(e)==val['e2r_fallback_gate_sha256']
    assert val['r2e_validation_relations']==val['e2r_fallback_validation_relations']==5216
    assert val['r2e_functional_score_scale']==1.25
    with a.open('rb') as f:bundle=pickle.load(f)
    assert set(bundle['directions'])=={'r2e','e2r'}
    with r.open('rb') as f:r2e=pickle.load(f)
    with e.open('rb') as f:e2r=pickle.load(f)
    for direction,gate in [('r2e',r2e),('e2r',e2r)]:
        assert gate['direction']==direction
        assert bundle['directions'][direction]['permission_power']==gate['permission_power']
        assert bundle['directions'][direction]['selected_params']==gate['selected_params']
        x=np.zeros((2,len(gate['feature_names'])),float)
        a0=gate['scaler'].transform(x)
        b0=bundle['directions'][direction]['scaler'].transform(x)
        assert np.array_equal(a0,b0)
        assert np.array_equal(gate['permission_model'].predict_proba(a0),
                              bundle['directions'][direction]['permission_model'].predict_proba(b0))


def test_r2e_functional_coefficient_selected_in_one_validation_pool():
    protocol=json.loads((REC.parent/'BRIDGE_COMMON_5216_VALIDATION_RESULT.json').read_text())
    a=protocol['r2e_functional_beta1_metrics']
    b=protocol['r2e_functional_beta125_metrics']
    for mode in ('direct','balanced'):
        for metric in ('mrr','hit10','hit100'):
            assert b[mode][metric]>=a[mode][metric]
    assert protocol['r2e_functional_single_validation_beta']==1.25


def test_both_episodic_memory_gates_are_from_the_one_shared_validation_set():
    from pathlib import Path
    import hashlib, pickle
    directory = Path(__file__).resolve().parents[3] / 'projects/active/bridge/release/runtime/final_bridge_v1'
    record=json.loads((REC.parent/'BRIDGE_EPISODIC_SINGLE_5216_VALIDATION_RESULT.json').read_text())
    common=json.loads((REC.parent/'BRIDGE_COMMON_5216_VALIDATION_RESULT.json').read_text())
    meta=json.loads((directory/'manifest.json').read_text())
    gatefile=directory/'episodic_memory_gate.production.pkl'
    sha=hashlib.sha256(gatefile.read_bytes()).hexdigest()
    assert sha==record['production_epi_gate_sha256']==meta['production_validation']['episodic_memory_gate']['sha256']
    assert record['unique_development_positive_pairs']==5216
    assert record['dev_final_test_exact_positive_overlap']==0
    assert record['dev_final_test_protein_query_overlap']==0
    assert (record['r2e']['episodes'],record['e2r']['episodes'])==(535,766)
    assert common['runtime_episodic_memory_gate_shared_validation']['production_sha256']==sha
    with gatefile.open('rb') as f: asset=pickle.load(f)
    assert asset['schema']=='bridge-episodic-memory-gate-bundle-v1'
    assert asset['validation_positive_edges']==5216
    assert set(asset['directions'])=={'r2e','e2r'}
    for direction in ('r2e','e2r'):
        result=record[direction]['selected_oof']
        gate=asset['directions'][direction]
        assert result['safe']
        assert result['oof']['hit10']>result['base']['hit10']
        assert result['oof']['mrr']>result['base']['mrr']
        assert result['oof']['hit100']>result['base']['hit100']
        assert gate['direction']==direction
        assert gate['feature_names']==list(gate['feature_names'])
        assert gate['external_metrics_used'] is False


def test_unified_validation_has_stable_content_digest_even_if_gzip_is_rewritten():
    """The development generator rewrites gzip with a fresh header timestamp."""
    from hashlib import sha256
    import pandas as pd
    root=Path(__file__).resolve().parents[3]
    common=json.loads((REC.parent/'BRIDGE_COMMON_5216_VALIDATION_RESULT.json').read_text())
    episodes=json.loads((REC.parent/'BRIDGE_EPISODIC_SINGLE_5216_VALIDATION_RESULT.json').read_text())
    expected=common['unified_validation']['sha256']
    assert episodes['source_validation_sha256']==expected
    assert len(expected)==64
    source=root/'results/bridge_e2r_query_gate_v3/gate_development_pairs.csv.gz'
    if source.is_file():
        frame=pd.read_csv(source,dtype=str)
        columns=['protein_id','reaction_id','partition','novelty','difficulty_stratum','source']
        assert len(frame)==5216 and set(frame)==set(columns)
        content=frame[columns].sort_values(['protein_id','reaction_id']).reset_index(drop=True).to_csv(index=False,lineterminator='\n').encode('utf-8')
        assert sha256(content).hexdigest()==expected
