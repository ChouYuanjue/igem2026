from projects.active.terpene_screening.evidence.mechanism import (
    family_motif_observables,
    build_sheet,
)


def test_geosmin_variant_aspartate_motif_is_observed_without_gate():
    x=family_motif_observables('AAAAADDHFLEAAAAANDLFSYQREAAAA','bacterial_classI')
    assert x['scope']=='class_I_terpene_synthase_observables'
    assert x['motifs']['typeI_aspartate_rich_variants'][0]['match']=='DDHFLE'
    assert x['motifs']['nse_like'][0]['match']=='NDLFSYQRE'
    assert 'compatible' not in x


def test_mechanism_sheet_is_ranking_neutral_and_reaction_conditioned():
    x=build_sheet('MARTS_EXT_RXN_012d56c064db','A0A1E7JZ38',max_references=4)
    assert x['ranking_modified'] is False
    assert x['reaction']['substrate_name']=='(2E,6E)-FPP'
    assert x['reaction']['product_name']=='geosmin'
    assert x['candidate']['domain_family']=='bacterial_classI'
    assert x['candidate']['architecture_observed_among_known_references'] is True
    assert x['known_positive_references']
    h=x['known_positive_references'][0]['candidate_to_reference_sequence_homology']
    assert 0.0 < h['global_identity'] <= 1.0
    assert 0.0 < h['local_identity'] <= 1.0
    assert 'compatible' not in x['evidence_interpretation']['architecture_statement'].lower()
