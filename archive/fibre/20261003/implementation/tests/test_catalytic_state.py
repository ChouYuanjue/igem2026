from projects.active.fibre.evidence.assay_context import (
    AssayContext, AssayObservation, NumericInterval,
)
from projects.active.fibre.evidence.catalytic_state import build_pair_catalytic_states


def test_catalytic_state_preserves_pair_vs_entity_scope():
    correspondence=[
        {
            "association":"reported_positive",
            "canonical_enzyme_id":"E1",
            "canonical_reaction_id":"R1",
            "properties":[
                {"term":"fibre:MARTS.mechanism_id","value":"M1"},
            ],
        }
    ]
    steps=[
        {
            "mechanism_id":"M1","step_index":1,"reaction_type":"cyclization",
            "source_evidence":"experiment",
        },
        {
            "mechanism_id":"M1","step_index":2,"reaction_type":"hydride shift",
            "source_evidence":"similarity",
        },
    ]
    annotations=[
        {
            "canonical_subject_id":"E1","predicate":"uniprot:CatalyticActivity",
            "value":"CATALYTIC ACTIVITY: Xref=Rhea:RHEA:12345;",
            "evidence":["ECO:0000269|PubMed:1"],
        },
        {
            "canonical_subject_id":"E1","predicate":"uniprot:Cofactor",
            "value":"COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420;",
            "evidence":["ECO:0000250|UniProtKB:P1"],
        },
        {
            "canonical_subject_id":"E1","predicate":"uniprot:ActiveSite",
            "value":"ACT_SITE 100;",
            "evidence":["ECO:0000269|PubMed:1"],
        },
    ]
    assay=[
        AssayObservation(
            observation_id="A1",enzyme_id="E1",reaction_id="R1",
            context=AssayContext(ph=NumericInterval.point(7.0,"pH")),
            outcome="reported_positive",source_scope="pair_assay",
            source_uri=None,source_record="x",evidence_texts=("pH 7",),
            target_binding_status="resolved",
        )
    ]
    states=build_pair_catalytic_states(correspondence,steps,annotations,assay)
    assert len(states)==1
    row=states[0]
    assert row.mechanism_ids==("M1",)
    assert row.mechanism_step_count==2
    assert row.mechanism_reaction_types==("cyclization","hydride shift")
    assert row.protein_catalytic_rhea_ids==("RHEA:12345",)
    assert row.protein_cofactors==("Mg(2+)",)
    assert row.protein_active_site_annotations==1
    assert row.pair_assay_observation_ids==("A1",)
    assert set(row.observed_components)=={
        "pair_mechanism","protein_catalytic_activity","protein_cofactor",
        "protein_active_site","pair_assay_context",
    }
    assert row.to_dict()["semantics"]["protein_annotations"].startswith("entity-level")


def test_catalytic_state_missingness_stays_absent():
    states=build_pair_catalytic_states(
        [{
            "association":"reported_positive",
            "canonical_enzyme_id":"E1",
            "canonical_reaction_id":"R1",
            "properties":[],
        }],
        [],[],[],
    )
    assert len(states)==1
    row=states[0]
    assert row.observed_components==()
    assert row.mechanism_step_count==0
    assert row.protein_cofactors==()
