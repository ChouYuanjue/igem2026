import json
from pathlib import Path

import pandas as pd

from projects.active.fibre.evidence.runtime_state import EnzymologyEvidenceIndex


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(x)+"\n" for x in rows),encoding="utf-8")


def test_runtime_state_preserves_scopes_and_pair_corroboration(tmp_path):
    pair=tmp_path/"pairs.jsonl"
    protein=tmp_path/"protein.jsonl"
    assay=tmp_path/"assay.jsonl"
    cross=tmp_path/"cross.csv"
    _write_jsonl(pair,[{
        "enzyme_id":"E1","reaction_id":"R1",
        "mechanism_ids":["M1"],"mechanism_step_count":2,
        "mechanism_reaction_types":["cyclization"],
        "mechanism_source_evidence":["experiment"],
        "pair_assay_observation_ids":["A1"],
    }])
    _write_jsonl(protein,[
        {
            "canonical_subject_id":"E1","source_subject_id":"U1",
            "predicate":"uniprot:CatalyticActivity",
            "value":"Reaction=x; Xref=Rhea:RHEA:12345;",
            "evidence":["ECO:0000269|PubMed:1"],
        },
        {
            "canonical_subject_id":"E1","source_subject_id":"U1",
            "predicate":"uniprot:Cofactor",
            "value":"COFACTOR: Name=Mg(2+);",
            "evidence":["ECO:0000250|UniProtKB:U2"],
        },
    ])
    _write_jsonl(assay,[{
        "observation_id":"A1","enzyme_id":"E1","reaction_id":"R1",
        "outcome":"reported_positive","context":{"ph":{"lower":7.0,"upper":7.0,"unit":"pH"}},
        "source_uri":"https://example.test/1",
    }])
    pd.DataFrame([{
        "enzyme_id":"E1","reaction_id":"R1",
        "status":"independent_experimental_uniprot_rhea_match",
        "overlap_rhea_ids":"RHEA:12345",
        "independent_experimental_support":"true",
    }]).to_csv(cross,index=False)

    idx=EnzymologyEvidenceIndex(
        pair_states=pair,protein_annotations=protein,
        assay_observations=assay,cross_source_pairs=cross,
    )
    state=idx.state("E1","R1")
    status=idx.status()
    assert status["pair_specific_negative_assay_observations"]==0
    assert status["conditional_eligibility_censor_supported"] is True
    assert status["current_dataset_censor_can_fire"] is False
    assert state["ranking_effect"] is False
    assert state["protein"]["scope"]=="protein_entity_annotation_not_pair_assay"
    assert state["protein"]["cofactors"]==["Mg(2+)"]
    assert state["protein"]["experimental_evidence_token_count"]==1
    assert state["reaction"]["mechanism_reaction_types"]==["cyclization"]
    assert state["pair"]["scope"]=="accepted_pair_specific_state"
    assert state["pair"]["assay_context_observations"][0]["observation_id"]=="A1"
    assert state["pair"]["cross_source_corroboration"]["independent_experimental_support"] is True


def test_runtime_state_missing_pair_is_unresolved_not_negative(tmp_path):
    pair=tmp_path/"pairs.jsonl"; protein=tmp_path/"protein.jsonl"
    _write_jsonl(pair,[])
    _write_jsonl(protein,[{
        "canonical_subject_id":"E1","source_subject_id":"U1",
        "predicate":"uniprot:Cofactor","value":"COFACTOR: Name=Mn(2+);","evidence":[],
    }])
    idx=EnzymologyEvidenceIndex(pair_states=pair,protein_annotations=protein)
    state=idx.state("E1","R-missing")
    assert state["protein"]["available"] is True
    assert state["pair"]["available"] is False
    assert state["pair"]["scope"]=="no_pair_specific_state_observed"
    assert state["reaction"]["available"] is False
    assert state["ranking_effect"] is False
