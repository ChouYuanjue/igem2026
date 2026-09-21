from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.starase_navigator.retrieval.focused as module
from scripts.starase_navigator.retrieval.focused import CorrespondenceGeometryService
from projects.active.fibre.evidence.assay_context import (
    AssayContext,
    AssayObservation,
    NumericInterval,
)


def direct_r2e_scores(service: CorrespondenceGeometryService, q: int) -> np.ndarray:
    dq2 = service.Dr2[q]
    joint = np.full(len(service.protein_ids), np.inf, dtype=np.float64)
    for rr, ee in service.positive_pairs:
        joint = np.minimum(joint, dq2[rr] + service.Dp2[:, ee])
    mr = np.min(dq2[service.r_support])
    return -np.maximum(joint - mr - service.protein_marginal_sq, 0.0)


def direct_e2r_scores(service: CorrespondenceGeometryService, q: int) -> np.ndarray:
    dq2 = service.Dp2[q]
    joint = np.full(len(service.reaction_ids), np.inf, dtype=np.float64)
    for rr, ee in service.positive_pairs:
        joint = np.minimum(joint, service.Dr2[:, rr] + dq2[ee])
    me = np.min(dq2[service.e_support])
    return -np.maximum(joint - service.reaction_marginal_sq - me, 0.0)


def assert_application_refinement_preserves_primary_levels(
    service: CorrespondenceGeometryService,
    scores: np.ndarray,
    returned_indices: list[int],
    top_k: int,
) -> None:
    levels,_=module.stable_level_ids(-np.asarray(scores,dtype=np.float64))
    got=np.asarray(returned_indices,dtype=np.int64)
    got_levels=levels[got]
    assert got_levels.tolist()==sorted(got_levels.tolist())
    cutoff=int(np.sort(levels)[min(int(top_k),len(levels))-1])
    required=set(np.flatnonzero(levels<cutoff).tolist())
    assert required.issubset(set(got.tolist()))
    assert all(int(levels[i])<=cutoff for i in got)


def test_known_reaction_section_matches_exact_correspondence_formula():
    s = CorrespondenceGeometryService()
    q = 17
    result = s.rank_enzymes({'reaction_id': s.reaction_ids[q], 'top_k': len(s.protein_ids)})
    by_internal = {row['canonical_candidate_id']: row['score'] for row in result['candidates']}
    expected = direct_r2e_scores(s, q)
    got = np.asarray([by_internal[pid] for pid in s.protein_ids])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert result['query']['observation_execution']['executed_measurements'] == []


def test_known_protein_section_matches_exact_correspondence_formula():
    s = CorrespondenceGeometryService()
    q = 31
    alias = str(s.protein_primary[q])
    result = s.rank_reactions({'enzyme_id': alias, 'top_k': len(s.reaction_ids)})
    by_internal = {row['canonical_candidate_id']: row['score'] for row in result['candidates']}
    expected = direct_e2r_scores(s, q)
    got = np.asarray([by_internal[rid] for rid in s.reaction_ids])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
    assert result['query']['observation_execution']['executed_measurements'] == []


def test_raw_smiles_of_reference_reaction_uses_exact_reference_state():
    s = CorrespondenceGeometryService()
    q = 8
    row = s.reactions.iloc[q]
    distance, meta = s.reaction_distances(reaction_smiles=str(row.reaction_smiles))
    np.testing.assert_allclose(distance, np.asarray(s.Dr[q], dtype=np.float64), rtol=0, atol=0)
    assert meta['query_is_reference_entity'] is True
    assert meta['executed_measurements'] == []


def test_raw_sequence_of_reference_protein_uses_exact_reference_state_without_encoder(monkeypatch):
    s = CorrespondenceGeometryService()
    q = 12
    sequence = str(s.proteins.iloc[q].sequence)
    monkeypatch.setattr(module, 'encode_external_enzymes_with_audit', lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('encoder should not run')))
    distance, meta = s.protein_distances(enzyme_sequence=sequence)
    np.testing.assert_allclose(distance, np.asarray(s.Dp[q], dtype=np.float64), rtol=0, atol=0)
    assert meta['query_is_reference_entity'] is True


def test_external_protein_executes_global_query_extension(monkeypatch):
    s = CorrespondenceGeometryService()
    vector = np.asarray(s.protein_global[0], dtype=np.float32).copy()
    vector[0] += 0.1
    vector /= np.linalg.norm(vector)

    class Audit:
        status = 'valid'
        warning = ''
        def __init__(self):
            self.__dict__ = {'status': 'valid', 'warning': ''}

    monkeypatch.setattr(module, 'encode_external_enzymes_with_audit', lambda *_a, **_k: (vector[None, :], [Audit()]))
    distance, meta = s.protein_distances(enzyme_sequence='ACDEFGHIKLMNPQRSTVWY' * 4)
    assert np.all(np.isfinite(distance))
    assert meta['query_is_reference_entity'] is False
    assert meta['executed_measurements'] == ['global_esmc']
    assert meta['attachment_count'] > 0


def test_candidate_subset_mask_and_seed_use_atlas_identities():
    s = CorrespondenceGeometryService()
    q = s.reaction_ids[0]
    candidates = [str(s.protein_primary[0]), str(s.protein_primary[1]), str(s.protein_primary[2])]
    seed = str(s.protein_primary[5])
    result = s.rank_enzymes({
        'reaction_id': q,
        'known_enzyme_ids': [seed],
        'candidate_ids': candidates,
        'mask_enzyme_ids': [candidates[1]],
        'top_k': 10,
    })
    assert {row['candidate_id'] for row in result['candidates']} <= {candidates[0], candidates[2]}
    assert len(result['candidates']) == 2
    assert result['query']['shot_mode'] == 'few_shot'


def test_seed_update_stability_is_first_class_in_both_directions():
    s=CorrespondenceGeometryService()

    r2e=s.rank_enzymes({
        'reaction_id':s.reaction_ids[0],
        'known_enzyme_ids':[str(s.protein_primary[5])],
        'top_k':5,
    })
    rstab=r2e['query']['seed_update_stability']
    assert rstab['status']=='applied_exact'
    assert rstab['seed_count']==1
    assert rstab['verified_seed_weight_policy']=='exact_observation_no_downweighting'
    rsec=rstab['query_section_influence']
    assert rsec['candidate_count']==len(s.protein_ids)
    assert (
        rsec['defect_decreased_count']
        + rsec['defect_unchanged_count']
        + rsec['defect_increased_count']
    ) == len(s.protein_ids)
    assert 0.0 <= rsec['affected_candidate_fraction'] <= 1.0
    assert rstab['registered_seed_influence_count']==1
    rglobal=rstab['registered_seed_influence'][0]
    assert rglobal['canonical_query_id']==s.reaction_ids[0]
    assert rglobal['canonical_seed_id']==s.protein_ids[5]
    assert 0.0 <= rglobal['affected_pair_fraction'] <= 1.0
    assert rglobal['seed_product_isolation_sq'] >= 0.0

    e2r=s.rank_reactions({
        'enzyme_id':s.protein_ids[0],
        'known_reaction_ids':[str(s.reaction_primary[5])],
        'top_k':5,
    })
    estab=e2r['query']['seed_update_stability']
    assert estab['status']=='applied_exact'
    assert estab['seed_count']==1
    esec=estab['query_section_influence']
    assert esec['candidate_count']==len(s.reaction_ids)
    assert (
        esec['defect_decreased_count']
        + esec['defect_unchanged_count']
        + esec['defect_increased_count']
    ) == len(s.reaction_ids)
    assert estab['registered_seed_influence_count']==1
    eglobal=estab['registered_seed_influence'][0]
    assert eglobal['canonical_query_id']==s.protein_ids[0]
    assert eglobal['canonical_seed_id']==s.reaction_ids[5]

    zero=s.rank_enzymes({'reaction_id':s.reaction_ids[0],'top_k':3})
    zstab=zero['query']['seed_update_stability']
    assert zstab['status']=='not_applied'
    assert zstab['seed_count']==0
    assert 'query_section_influence' not in zstab


def test_reaction_alias_table_is_one_state_with_many_product_aliases_not_duplicate_states():
    s = CorrespondenceGeometryService()
    assert len(set(s.reaction_primary.tolist())) == len(s.reaction_ids)
    multi = s.reactions[s.reactions.rhea_aliases.str.contains(';', regex=False)]
    assert len(multi) > 0
    row = multi.iloc[0]
    aliases = str(row.rhea_aliases).split(';')
    internals = {s._reaction_internal(alias) for alias in aliases}
    assert internals == {str(row.reaction_id)}


def test_multiple_external_positive_seeds_share_one_encoder_batch(tmp_path, monkeypatch):
    import pandas as pd
    import scripts.starase_navigator.retrieval.focused as module

    s=CorrespondenceGeometryService()
    path=tmp_path/'external_seeds.csv'
    pd.DataFrame([
        {'enzyme_id':'EXT-SEED-A','sequence':'ACDEFGHIKLMNPQRSTVWY'*4},
        {'enzyme_id':'EXT-SEED-B','sequence':'YWVTSRQPNMLKIHGFEDCA'*4},
    ]).to_csv(path,index=False)
    calls=[]

    class Audit:
        def __init__(self): self.__dict__={'status':'valid','warning':''}

    def fake_encode(frame,*_args,**_kwargs):
        calls.append(frame.copy())
        assert frame.enzyme_id.astype(str).tolist()==['EXT-SEED-A','EXT-SEED-B']
        return np.stack([
            np.asarray(s.protein_global[0],dtype=np.float32),
            np.asarray(s.protein_global[1],dtype=np.float32),
        ]), [Audit(),Audit()]

    monkeypatch.setattr(module,'encode_external_enzymes_with_audit',fake_encode)
    distance,missing,count=s._protein_seed_distance_sq(
        ['EXT-SEED-A','EXT-SEED-B'],path
    )
    assert len(calls)==1
    assert missing==[]
    assert count==2
    assert distance is not None and distance.shape==(len(s.protein_ids),)
    assert np.all(np.isfinite(distance))


def test_external_protein_can_refine_same_attachment_with_cached_whole_structure(monkeypatch):
    s=CorrespondenceGeometryService()
    vector=np.asarray(s.protein_global[0],dtype=np.float32).copy()
    vector[0]+=0.1
    vector/=np.linalg.norm(vector)

    class Audit:
        def __init__(self): self.__dict__={'status':'valid','warning':''}
    monkeypatch.setattr(module,'encode_external_enzymes_with_audit',lambda *_a,**_k:(vector[None,:],[Audit()]))
    sequence='ACDEFGHIKLMNPQRSTVWY'*5
    base,base_meta=s.protein_distances(enzyme_sequence=sequence)
    structure='results/clipzyme_native_extension_v1/structures/af_v6/AF-A5G9B7-F1-model_v6.cif'
    deep,deep_meta=s.protein_distances(enzyme_sequence=sequence,protein_structure_path=structure)
    assert np.all(np.isfinite(base)) and np.all(np.isfinite(deep))
    assert not np.allclose(base,deep)
    assert deep_meta['executed_measurements']==['global_esmc','resolved_structure','whole_3di']
    assert deep_meta['failed_measurements']=={}
    assert deep_meta['whole_3di_hit_count']>0
    assert base_meta['executed_measurements']==['global_esmc']


def test_geometric_uncertainty_is_threshold_free_and_eligible_only():
    scores=np.asarray([0.0,0.0,-1.0,-2.0])
    eligible=np.asarray([True,True,False,True])
    marginal=np.asarray([1.0,4.0,9.0,16.0])
    u=CorrespondenceGeometryService._geometric_uncertainty(
        scores,eligible,9.0,marginal
    )
    assert u['schema']=='fibre-geometric-uncertainty-v1'
    assert u['status']=='available'
    assert u['calibrated_probability'] is False
    assert u['query_support_distance']==3.0
    assert u['best_level_size']==2
    assert u['best_level_fraction']==2/3
    assert u['next_level_gap']==2.0
    assert u['best_level_candidate_support_distance_min']==1.0
    assert u['best_level_candidate_support_distance_median']==1.5
    assert 'tier' not in u
    assert 'probability' not in u


def test_focused_result_exposes_level_membership_without_changing_rank():
    s=CorrespondenceGeometryService()
    q=17
    result=s.rank_enzymes({'reaction_id':s.reaction_ids[q],'top_k':20})
    u=result['query']['geometric_uncertainty']
    assert u['calibrated_probability'] is False
    assert u['best_level_size'] >= 1
    assert u['best_level_fraction'] > 0
    assert result['candidates'][0]['in_best_numerical_level'] is True
    ranks=[row['rank'] for row in result['candidates']]
    assert ranks==list(range(1,len(ranks)+1))


def test_stratified_r2e_is_non_order_bearing_and_application_refines_only_within_primary_levels():
    s=CorrespondenceGeometryService()
    q=s.ri['MARTS_RXN_ed3cf125a033969c']
    scores=direct_r2e_scores(s,q)
    eligible=np.ones(len(scores),dtype=bool)
    result=s.rank_enzymes({'reaction_id':s.reaction_ids[q],'top_k':25})
    got=[s.pi[row['canonical_candidate_id']] for row in result['candidates']]
    assert_application_refinement_preserves_primary_levels(s,scores,got,25)
    assert result['query']['application_profile']['status']=='ready'
    assert result['query']['application_profile']['ordering_policy'].startswith(
        'primary FIBRE numerical level'
    )
    for level in sorted({row['fibre_resolution']['coarse_level'] for row in result['candidates']}):
        block=[
            row for row in result['candidates']
            if row['fibre_resolution']['coarse_level']==level
        ]
        defects=[
            row['application_refinement']['tps_domain_defect']
            for row in block
            if row['application_refinement']['tps_domain_defect'] is not None
        ]
        assert defects==sorted(defects)
    meta=result['query']['stratified_correspondence']
    assert meta['schema']=='fibre-stratified-section-v2'
    assert meta['total_rank_source']=='coarse_global_correspondence'
    assert meta['catalytic_strata_order_bearing'] is False
    assert meta['mechanistic_strata_order_bearing'] is False
    assert meta['mechanistic_status']=='available_non_order_bearing'
    assert meta['mechanistic_refined_parent_chart_count'] > 0
    assert meta['promotion_status']=='not_promoted_strict_inductive_non_degradation_gate_failed'
    relation=result['query']['biological_relation']
    assert relation['schema']=='fibre-biological-relation-v2'
    assert relation['status']=='available_non_order_bearing'
    assert relation['order_bearing'] is False
    assert relation['canonical_rank_unchanged'] is True
    assert relation['support_role']=='applicability_only_not_ordering'
    assert relation['context_constraint_authority']=='matched pair-specific inactive/below-detection assay only'
    assert relation['relation_scope']=='returned_candidate_set'
    assert all('fibre_relation' in row for row in result['candidates'])
    assert all(row['fibre_relation']['order_bearing'] is False for row in result['candidates'])
    assert all('fibre_resolution' in row for row in result['candidates'])
    assert all('coarse_level' in row['fibre_resolution'] for row in result['candidates'])
    assert all('catalytic_observed' in row['fibre_resolution'] for row in result['candidates'])
    assert all('mechanistic_chart' in row['fibre_resolution'] for row in result['candidates'])
    assert all('mechanistic_stratum' in row['fibre_resolution'] for row in result['candidates'])
    assert all('mechanistic_coordinates' in row['fibre_resolution'] for row in result['candidates'])


def test_stratified_e2r_is_non_order_bearing_and_application_refines_only_within_primary_levels():
    s=CorrespondenceGeometryService()
    q=s.pi['MARTS_SEQ_cd2c2cfa45ad818a']
    scores=direct_e2r_scores(s,q)
    eligible=np.ones(len(scores),dtype=bool)
    result=s.rank_reactions({'enzyme_id':s.protein_ids[q],'top_k':25})
    got=[s.ri[row['canonical_candidate_id']] for row in result['candidates']]
    assert_application_refinement_preserves_primary_levels(s,scores,got,25)
    assert result['query']['application_profile']['status']=='ready'
    for level in sorted({row['fibre_resolution']['coarse_level'] for row in result['candidates']}):
        block=[
            row for row in result['candidates']
            if row['fibre_resolution']['coarse_level']==level
        ]
        defects=[
            row['application_refinement']['tps_domain_defect']
            for row in block
            if row['application_refinement']['tps_domain_defect'] is not None
        ]
        assert defects==sorted(defects)
    meta=result['query']['stratified_correspondence']
    assert meta['total_rank_source']=='coarse_global_correspondence'
    assert meta['catalytic_strata_order_bearing'] is False
    assert meta['mechanistic_strata_order_bearing'] is False
    assert meta['mechanistic_status']=='available_non_order_bearing'
    assert meta['mechanistic_refined_parent_chart_count'] > 0
    assert meta['query_mechanistic_chart'] == ['typeI_aspartate','nse_dte']
    assert meta['query_mechanistic_coordinates'] == ['typeI_aspartate','nse_dte']
    relation=result['query']['biological_relation']
    assert relation['schema']=='fibre-biological-relation-v2'
    assert relation['status']=='available_non_order_bearing'
    assert relation['order_bearing'] is False
    assert relation['canonical_rank_unchanged'] is True
    assert relation['support_role']=='applicability_only_not_ordering'
    assert relation['context_constraint_authority']=='matched pair-specific inactive/below-detection assay only'
    assert all('fibre_relation' in row for row in result['candidates'])
    assert all(row['fibre_relation']['order_bearing'] is False for row in result['candidates'])
    assert all('fibre_resolution' in row for row in result['candidates'])
    assert all('catalytic_observed' in row['fibre_resolution'] for row in result['candidates'])
    assert all(row['fibre_resolution']['mechanistic_chart'] == ['typeI_aspartate','nse_dte'] for row in result['candidates'])


def test_real_source_bound_assay_context_is_scoped_and_non_order_bearing():
    s=CorrespondenceGeometryService()
    reaction_id='MARTS_RXN_bca813a0b5f35b09'
    protein_id='MARTS_SEQ_0800c44f5abfe5f0'
    payload={
        'reaction_id':reaction_id,
        'candidate_ids':[protein_id],
        'top_k':1,
        'target_conditions':{'ph':6.5},
    }
    matched=s.rank_enzymes(payload)
    relation=matched['query']['biological_relation']
    assert relation['requested_assay_dimensions']==['ph']
    assert relation['biological_state']['context_status_counts']['supported']==1
    assert relation['order_bearing'] is False
    row=matched['candidates'][0]
    assert row['canonical_candidate_id']==protein_id
    assert row['fibre_relation']['biological_state']['context_status']=='supported'
    assert 'pair_assay_context' in row['fibre_relation']['biological_state']['catalytic_state_components']
    assays=row['enzymology_state']['pair']['assay_context_observations']
    assert len(assays)==1
    assert assays[0]['context']['ph']=={'lower':6.5,'unit':'pH','upper':6.5}

    mismatched=s.rank_enzymes({
        **payload,
        'target_conditions':{'ph':7.0},
    })
    mismatch_relation=mismatched['query']['biological_relation']
    assert mismatch_relation['biological_state']['context_status_counts']['unresolved']==1
    assert mismatched['candidates'][0]['fibre_relation']['biological_state']['context_status']=='unresolved'
    assert mismatched['candidates'][0]['score']==matched['candidates'][0]['score']


def test_exact_condition_matched_negative_assay_censors_only_that_r2e_candidate():
    s=CorrespondenceGeometryService()
    reaction_id=str(s.reaction_ids[0])
    excluded_id=str(s.protein_ids[0])
    retained_id=str(s.protein_ids[1])

    negative=AssayObservation(
        observation_id='negative-r2e',
        enzyme_id=excluded_id,
        reaction_id=reaction_id,
        context=AssayContext(ph=NumericInterval.point(7.0,'pH')),
        outcome='below_detection',
        source_scope='pair_assay',
        source_uri='https://example.test/assay',
        source_record='assay:r2e',
        evidence_texts=('below detection at pH 7.0',),
        target_binding_status='resolved',
    )

    class Evidence:
        def assay_observations(self,protein_id,reaction_id_value):
            if (str(protein_id),str(reaction_id_value))==(excluded_id,reaction_id):
                return (negative,)
            return ()
        @staticmethod
        def state(_protein_id,_reaction_id):
            return {'observed_components':[]}
        @staticmethod
        def status():
            return {'schema':'fake','ranking_effect':False}

    s.enzymology_evidence=Evidence()
    payload={
        'reaction_id':reaction_id,
        'candidate_ids':[str(s.protein_primary[0]),str(s.protein_primary[1])],
        'top_k':2,
        'target_conditions':{'ph':7.0},
    }
    matched=s.rank_enzymes(payload)
    assert [row['canonical_candidate_id'] for row in matched['candidates']]==[retained_id]
    constraint=matched['query']['assay_context_constraint']
    assert constraint['status']=='applied'
    assert constraint['excluded_candidate_count']==1
    assert constraint['excluded_canonical_candidate_ids']==[excluded_id]
    assert constraint['score_mutated'] is False

    mismatched=s.rank_enzymes({**payload,'target_conditions':{'ph':8.0}})
    assert {
        row['canonical_candidate_id'] for row in mismatched['candidates']
    }=={excluded_id,retained_id}
    mismatch_constraint=mismatched['query']['assay_context_constraint']
    assert mismatch_constraint['status']=='evaluated_no_exclusion'
    assert mismatch_constraint['excluded_candidate_count']==0


def test_exact_condition_matched_negative_assay_censors_only_that_e2r_candidate():
    s=CorrespondenceGeometryService()
    protein_id=str(s.protein_ids[0])
    excluded_id=str(s.reaction_ids[0])
    retained_id=str(s.reaction_ids[1])

    negative=AssayObservation(
        observation_id='negative-e2r',
        enzyme_id=protein_id,
        reaction_id=excluded_id,
        context=AssayContext(
            temperature_c=NumericInterval.point(30.0,'°C')
        ),
        outcome='no_conversion',
        source_scope='pair_assay',
        source_uri='https://example.test/assay',
        source_record='assay:e2r',
        evidence_texts=('no conversion at 30 C',),
        target_binding_status='resolved',
    )

    class Evidence:
        def assay_observations(self,protein_id_value,reaction_id):
            if (str(protein_id_value),str(reaction_id))==(protein_id,excluded_id):
                return (negative,)
            return ()
        @staticmethod
        def state(_protein_id,_reaction_id):
            return {'observed_components':[]}
        @staticmethod
        def status():
            return {'schema':'fake','ranking_effect':False}

    s.enzymology_evidence=Evidence()
    result=s.rank_reactions({
        'enzyme_id':protein_id,
        'candidate_ids':[str(s.reaction_primary[0]),str(s.reaction_primary[1])],
        'top_k':2,
        'target_conditions':{'temperature_c':30.0},
    })
    assert [row['canonical_candidate_id'] for row in result['candidates']]==[retained_id]
    constraint=result['query']['assay_context_constraint']
    assert constraint['status']=='applied'
    assert constraint['excluded_canonical_candidate_ids']==[excluded_id]


def test_dynamic_positive_update_keeps_fine_resolution_non_order_bearing_and_unprojected():
    s=CorrespondenceGeometryService()
    result=s.rank_enzymes({
        'reaction_id':s.reaction_ids[0],
        'known_enzyme_ids':[str(s.protein_primary[5])],
        'top_k':10,
    })
    meta=result['query']['stratified_correspondence']
    assert meta['status']=='local_resolution_not_projected_through_dynamic_positive_update'
    assert meta['catalytic_strata_order_bearing'] is False
    assert meta['mechanistic_strata_order_bearing'] is False
    relation=result['query']['biological_relation']
    assert relation['status']=='relation_not_projected_through_dynamic_positive_update'
    assert relation['order_bearing'] is False
    assert all(row['fibre_relation']['pareto_front'] is None for row in result['candidates'])
    assert all(row['fibre_resolution']['catalytic_stratum'] is None for row in result['candidates'])
    assert all(row['fibre_resolution']['mechanistic_stratum'] is None for row in result['candidates'])
