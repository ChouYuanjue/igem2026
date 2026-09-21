from __future__ import annotations

import pandas as pd
import pytest

from projects.active.fibre.core.candidate_universes import MARTS_CORRESPONDENCE_UNIVERSE
from scripts.starase_navigator.serve import NavigatorRuntime


@pytest.fixture(scope='module')
def runtime():
    rt=NavigatorRuntime()
    rt.route_designer.known_uniprot_ids=lambda _rid: []
    rt.route_designer.known_rhea_ids=lambda _pid: []
    return rt


def r2e_plan(**_kwargs):
    return {
        'top_k':10,'ranking_objective':'top10','enzyme_taxonomy_scope':'all',
        'known_association_policy':'separate_known','known_enzyme_ids':[],
        'seed_mode':'none','seed_source':'none','homology_filter_requested':False,
        'homology_filter_applied':False,'homology_anchor_ids':[],'homology_anchor_source':'none',
        'retrieval_scope':'application_domain','analysis_depth':'standard','observation_mode':'standard',
        'candidate_universe':MARTS_CORRESPONDENCE_UNIVERSE,
        'candidate_universe_source':'deepseek_semantic_scope','planned_route_id':'fibre-r2e-v1',
        'warnings':[],
    }


def e2r_plan(**_kwargs):
    return {
        'top_k':10,'ranking_objective':'top10','known_association_policy':'separate_known',
        'known_reaction_ids':[],'use_known_activity_seeds':False,'seed_mode':'none','seed_source':'none',
        'mask_reaction_ids':[],'retrieval_scope':'application_domain','analysis_depth':'deep','observation_mode':'deep',
        'candidate_universe':MARTS_CORRESPONDENCE_UNIVERSE,
        'candidate_universe_source':'deepseek_semantic_scope','planned_route_id':'fibre-e2r-v1',
        'warnings':[],
    }


def test_external_reaction_runs_real_query_extension_and_reports_executed_observations(runtime):
    runtime.route_planner.plan=r2e_plan
    # Deliberately outside the MARTS reaction signature registry.
    reaction='CCO>>CC=O'
    result=runtime.rank(
        '',reaction_smiles=reaction,query_id='EXT-ETHANOL-OX',route_mode='intelligent',
        observation_mode='standard',ui_language='en',
    )
    assert result['ranking']['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE
    assert result['ranking']['candidate_universe_size']==1421
    assert result['ranking']['score_source']=='correspondence_geometry'
    assert result['ranking']['retrieval_scope']=='application_domain'
    assert result['ranking']['analysis_depth']=='standard'
    assert result['ranking']['stratified_correspondence']['total_rank_source']=='coarse_global_correspondence'
    assert result['ranking']['stratified_correspondence']['catalytic_strata_order_bearing'] is False
    relation=result['ranking']['biological_relation']
    assert relation['schema']=='fibre-biological-relation-v2'
    assert relation['status']=='relation_unavailable_for_external_query'
    assert relation['order_bearing'] is False
    assert result['ranking']['geometric_uncertainty']
    assert result['ranking']['application_profile']['status']=='ready'
    assert result['ranking']['application_profile']['benchmark_claims_allowed'] is False
    assert len(result['candidates'])==10
    assert all(str(row['candidate_id']) for row in result['candidates'])
    assert all(row['application_refinement']['profile']=='starase-application' for row in result['candidates'])
    assert all(row['application_refinement']['benchmark_evidence'] is False for row in result['candidates'])
    assert all('fibre_relation' in row for row in result['candidates'])
    assert all(row['fibre_relation']['pareto_front'] is None for row in result['candidates'])
    assert all('fibre_resolution' in row for row in result['candidates'])
    assert all('support_applicability' in row for row in result['candidates'])
    assert all(
        'not an activity probability' in row['support_applicability']['interpretation']
        for row in result['candidates']
    )
    assert all(
        row['support_applicability']['canonical_candidate_support_distance'] >= 0
        for row in result['candidates']
    )
    assert all(
        row['support_applicability']['nearest_joint_positive_distance'] >= 0
        for row in result['candidates']
    )
    assert result['ranking']['enzymology_evidence_index']['status']=='ready'
    assert all('enzymology_state' in row for row in result['candidates'])
    assert all(row['enzymology_state']['ranking_effect'] is False for row in result['candidates'])
    assert all(
        row['enzymology_state']['pair']['scope'] in {
            'accepted_pair_specific_state','no_pair_specific_state_observed'
        }
        for row in result['candidates']
    )
    plan=result['observation_plan']
    assert set(plan['executed_measurements'])=={'drfp','reactant_product_neighbourhood'}
    rows={row['measurement_id']:row for row in plan['measurements']}
    assert rows['drfp']['status']=='computed_now'
    assert rows['reactant_product_neighbourhood']['status']=='computed_now'
    assert rows['atom_mapping']['status']=='defer'
    assert plan['atlas_policy'].startswith('out_of_sample')
    assert plan['correspondence_snapshot']['atlas_version']=='terpene-correspondence-deployment-atlas-v2'
    candidate_obs=plan['candidate_reference_observations']
    assert candidate_obs['entity_kind']=='protein'
    assert candidate_obs['candidate_count']==10
    assert candidate_obs['reference_resolved_count']==10
    assert candidate_obs['measurement_counts']['global_esmc']==10
    assert 'whole_3di' in candidate_obs['factor_measurements']
    assert 'pocket_ot' in candidate_obs['factor_measurements']


def test_reference_protein_reuses_cached_multiresolution_state_without_encoder(runtime,monkeypatch):
    runtime.e2r_planner.plan=e2r_plan
    # Select a reference protein that genuinely has all three catalytic-pocket
    # coordinates; missing local coordinates must remain unresolved, not be faked.
    service=runtime.model_gateway.correspondence_service()
    e=int(service.local_global_rows[0])
    row=service.proteins.iloc[e]
    query_id=str(service.protein_primary[e])
    # If the exact sequence is recognized as a reference state, no ESM-C query encoder should run.
    import scripts.starase_navigator.retrieval.focused as module
    monkeypatch.setattr(module,'encode_external_enzymes_with_audit',lambda *_a,**_k: (_ for _ in ()).throw(AssertionError('reference query must not invoke encoder')))
    result=runtime.rank_reactions(
        '',enzyme_sequence=str(row.sequence),query_id=query_id,route_mode='intelligent',
        observation_mode='deep',ui_language='en',
    )
    assert result['ranking']['candidate_universe']==MARTS_CORRESPONDENCE_UNIVERSE
    assert result['ranking']['candidate_universe_size']==453
    assert result['ranking']['score_source']=='correspondence_geometry'
    assert result['ranking']['retrieval_scope']=='application_domain'
    assert result['ranking']['analysis_depth']=='deep'
    assert result['ranking']['stratified_correspondence']['total_rank_source']=='coarse_global_correspondence'
    assert result['ranking']['stratified_correspondence']['catalytic_strata_order_bearing'] is False
    relation=result['ranking']['biological_relation']
    assert relation['schema']=='fibre-biological-relation-v2'
    assert relation['status']=='available_non_order_bearing'
    assert relation['order_bearing'] is False
    assert relation['canonical_rank_unchanged'] is True
    assert relation['support_role']=='applicability_only_not_ordering'
    assert relation['biological_state']['ranking_policy'].startswith('no fused biological score')
    assert len(relation['pareto_front_sizes'])>=1
    assert result['ranking']['geometric_uncertainty']
    assert result['ranking']['application_profile']['status']=='ready'
    assert result['ranking']['application_profile']['within_level_refinement']=='tps_pair_supervised_fibre_coordinate'
    assert len(result['candidates'])==10
    assert all('application_refinement' in row for row in result['candidates'])
    assert all(row['application_refinement']['within_primary_level_order_bearing'] is True for row in result['candidates'])
    assert all('fibre_relation' in row for row in result['candidates'])
    assert all(row['fibre_relation']['order_bearing'] is False for row in result['candidates'])
    assert all('fibre_resolution' in row for row in result['candidates'])
    assert all('support_applicability' in row for row in result['candidates'])
    assert all(
        row['support_applicability']['canonical_candidate_support_distance'] >= 0
        for row in result['candidates']
    )
    assert all(
        row['support_applicability']['current_candidate_support_distance'] >= 0
        for row in result['candidates']
    )
    assert all(
        row['support_applicability']['nearest_joint_positive_distance'] >= 0
        for row in result['candidates']
    )
    assert result['ranking']['enzymology_evidence_index']['status']=='ready'
    assert all('enzymology_state' in row for row in result['candidates'])
    assert all(row['enzymology_state']['ranking_effect'] is False for row in result['candidates'])
    assert all(
        row['enzymology_state']['protein']['scope'] in {
            'protein_entity_annotation_not_pair_assay','protein_state_unresolved'
        }
        for row in result['candidates']
    )
    plan=result['observation_plan']
    assert plan['executed_measurements']==[]
    assert plan['query_is_reference_entity'] is True
    assert 'global_esmc' in plan['selected_factor_measurements']
    assert any(row['status']=='reuse_cached' for row in plan['measurements'])
    assert plan['correspondence_snapshot']['atlas_version']=='terpene-correspondence-deployment-atlas-v2'
    candidate_obs=plan['candidate_reference_observations']
    assert candidate_obs['entity_kind']=='reaction'
    assert candidate_obs['candidate_count']==10
    assert candidate_obs['reference_resolved_count']==10
    assert candidate_obs['measurement_counts']['drfp']==10
    assert candidate_obs['measurement_counts']['reactant_product_neighbourhood']==10
    assert 'reaction_center_transition' in candidate_obs['evidence_only_measurements']


def test_focused_route_reuses_verified_general_positive_as_automatic_oos_seed(runtime, monkeypatch):
    import numpy as np
    import scripts.starase_navigator.retrieval.focused as module

    def focused_plan(**_kwargs):
        plan=r2e_plan()
        plan.update({
            'known_enzyme_ids':['Q5AU81'],
            'seed_mode':'catalog_known',
            'seed_source':'catalog_known_associations',
            'shot_mode':'few_shot',
        })
        return plan

    runtime.route_planner.plan=focused_plan
    service=runtime.model_gateway.correspondence_service()
    reference_embedding=np.asarray(service.protein_global[0],dtype=np.float32)[None,:]

    class Audit:
        def __init__(self):
            self.__dict__={'status':'valid','warning':''}

    calls=[]
    def fake_encode(frame,*_args,**_kwargs):
        calls.append(frame[['enzyme_id','sequence']].copy())
        assert frame.enzyme_id.astype(str).tolist()==['Q5AU81']
        assert len(str(frame.iloc[0].sequence))==809
        return reference_embedding.copy(), [Audit()]

    monkeypatch.setattr(module,'encode_external_enzymes_with_audit',fake_encode)
    result=runtime.rank(
        '',reaction_smiles='CCO>>CC=O',query_id='EXT-WITH-KNOWN-POSITIVE',
        route_mode='intelligent',observation_mode='deep',ui_language='en',
    )
    assert len(calls)==1
    extension=result['routing']['automatic_positive_extension']
    assert extension['count']==1
    assert extension['source']=='local_general_sequence_registry'
    audit=result['routing']['seed_candidate_universe_audit']
    assert audit['requested_seed_count']==1
    assert audit['effective_seed_count']==1
    assert audit['dropped_seed_count']==0
    assert audit['temporary_extension_seed_count']==1
    assert result['ranking']['shot_mode']=='few_shot'
    stability=result['ranking']['seed_update_stability']
    assert stability['status']=='applied_exact'
    assert stability['seed_count']==1
    assert stability['verified_seed_weight_policy']=='exact_observation_no_downweighting'
    assert stability['query_section_influence']['candidate_count']==1421
    assert stability['registered_seed_influence_count']==0


def test_deep_external_protein_reuses_cached_whole_structure_in_same_geometry(runtime, monkeypatch):
    import numpy as np
    import scripts.starase_navigator.retrieval.focused as module

    runtime.e2r_planner.plan=e2r_plan
    sequence=runtime.evidence.candidate_protein_sequence('A5G9B7')
    assert sequence and not runtime.model_gateway.correspondence_contains_protein('A5G9B7')
    structure=runtime.evidence.candidate_protein_structure('A5G9B7')
    assert structure is not None and structure.is_file()
    service=runtime.model_gateway.correspondence_service()
    vector=np.asarray(service.protein_global[0],dtype=np.float32).copy()
    vector[0]+=0.07
    vector/=np.linalg.norm(vector)

    class Audit:
        def __init__(self): self.__dict__={'status':'valid','warning':''}
    monkeypatch.setattr(module,'encode_external_enzymes_with_audit',lambda *_a,**_k:(vector[None,:],[Audit()]))
    result=runtime.rank_reactions(
        '',enzyme_sequence=sequence,query_id='A5G9B7',user_text='深入利用结构信息寻找可能反应',
        route_mode='intelligent',observation_mode='standard',ui_language='zh',
    )
    assert result['ranking']['retrieval_scope']=='application_domain'
    assert result['ranking']['analysis_depth']=='deep'
    plan=result['observation_plan']
    assert {'global_esmc','resolved_structure','whole_3di'} <= set(plan['executed_measurements'])
    rows={row['measurement_id']:row for row in plan['measurements']}
    assert rows['resolved_structure']['status']=='computed_now'
    assert rows['whole_3di']['status']=='computed_now'
    assert result['routing']['deep_structure_observation']['status']=='reuse_local_cache'
    assert result['routing']['deep_structure_observation']['query_time_network_fetch'] is False
