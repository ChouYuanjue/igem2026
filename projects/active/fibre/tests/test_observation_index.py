from __future__ import annotations

import pandas as pd

from projects.active.fibre.evidence.observation_index import (
    build_marts_observations,
    build_marts_mechanism_steps,
    build_uniprot_entity_annotations,
    canonical_positive_pairs,
    observation_property_terms,
)


def _real_observations():
    raw=pd.read_csv('data/terpene_marts/marts_reaction_pairs.tsv',sep='\t',dtype=str).fillna('')
    proteins=pd.read_csv('data/terpene_marts_adaptation/protein_entities.csv',dtype=str).fillna('')
    reactions=pd.read_csv('data/terpene_marts_adaptation/reaction_entities.csv',dtype=str).fillna('')
    return raw,build_marts_observations(raw,proteins,reactions)


def test_marts_observation_projection_exactly_recovers_canonical_omega():
    _raw,obs=_real_observations()
    omega=pd.read_csv('data/terpene_marts_adaptation/marts_pair_folds.csv',dtype=str).fillna('')
    expected=set(zip(omega.Entry.astype(str),omega.rhea_id.astype(str)))
    assert canonical_positive_pairs(obs) == expected


def test_all_raw_rows_are_retained_even_when_reaction_is_outside_current_atlas():
    raw,obs=_real_observations()
    assert len(obs) == len(raw) == 2833
    assert sum(x.canonical_enzyme_id is None for x in obs) == 0
    assert sum(x.canonical_reaction_id is None for x in obs) == 307


def test_observation_index_does_not_invent_assay_conditions_or_kinetics():
    _raw,obs=_real_observations()
    terms=observation_property_terms(obs)
    forbidden={
        'strenda:reaction_conditions.temperature',
        'strenda:reaction_conditions.pH_value',
        'strenda:results.turnover_number',
        'strenda:results.michaelis_constant',
        'strenda:results.catalytic_efficiency',
        'strenda:results.specific_activity',
    }
    assert not (terms & forbidden)


def test_marts_species_is_source_organism_not_expression_host():
    _raw,obs=_real_observations()
    terms=observation_property_terms(obs)
    assert 'strenda:Biocatalyst.origin_organism' in terms
    assert 'enzymeml:Protein.organism' not in terms


def test_flat_reaction_fields_do_not_invent_enzymeml_slots():
    _raw,obs=_real_observations()
    terms=observation_property_terms(obs)
    assert 'fibre:reaction.reactant_name' in terms
    assert 'fibre:reaction.product_smiles' in terms
    assert 'enzymeml:Reaction.reactant_name' not in terms


def test_no_mechanism_placeholder_is_not_materialised_as_mechanistic_evidence():
    raw,obs=_real_observations()
    placeholder_rows=set(raw.index[raw.mechanism_marts_id.eq('no_mechanism')])
    assert placeholder_rows
    for i in list(placeholder_rows)[:50]:
        terms={p.term for p in obs[i].properties}
        assert 'fibre:MARTS.mechanism_id' not in terms


def test_mechanism_steps_preserve_source_evidence_without_tiering():
    df=pd.read_csv(
        'data/terpene_marts/marts_mechanism_steps.tsv',sep='\t',dtype=str
    ).fillna('')
    steps=build_marts_mechanism_steps(df)
    assert len(steps)==3395
    assert len({x.mechanism_id for x in steps})==504
    assert {x.source_evidence for x in steps} >= {
        'experiment','similarity','calculation',None
    }
    row=steps[0].to_dict()
    assert 'quality_tier' not in row
    assert 'evidence_level' not in row


def test_uniprot_state_annotations_preserve_eco_and_exact_subject_mapping():
    proteins=pd.DataFrame([
        {'protein_id':'P_CANON','sequence':'AAAA','aliases':'OLDALIAS'}
    ])
    uni=pd.DataFrame([{
        'Entry':'QTEST1',
        'Reviewed':'reviewed',
        'Catalytic activity':'CATALYTIC ACTIVITY: Reaction=A = B; Xref=Rhea:RHEA:1; Evidence={ECO:0000269|PubMed:123};',
        'Cofactor':'COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence={ECO:0000250|UniProtKB:QX};',
        'Active site':'',
        'Binding site':'',
        'EC number':'1.2.3.4',
    }])
    obs=build_uniprot_entity_annotations(
        uni,proteins,extra_subject_map={'QTEST1':'P_CANON'}
    )
    by_pred={x.predicate:x for x in obs}
    assert by_pred['uniprot:CatalyticActivity'].canonical_subject_id=='P_CANON'
    assert by_pred['uniprot:CatalyticActivity'].evidence==(
        'ECO:0000269|PubMed:123',
    )
    assert by_pred['uniprot:Cofactor'].evidence==(
        'ECO:0000250|UniProtKB:QX',
    )
    assert 'quality_tier' not in by_pred['uniprot:Cofactor'].to_dict()


def test_publication_provenance_is_preserved_without_quality_tier():
    raw,obs=_real_observations()
    assert sum(bool(x.provenance[0].source_uri) for x in obs) == int(raw.publication.ne('').sum())
    as_dict=obs[0].to_dict()
    assert 'quality_tier' not in as_dict
    assert 'evidence_level' not in as_dict
