from projects.active.fibre.evidence.assay_context import (
    AssayContext,
    AssayObservation,
    NumericInterval,
    assess_pair_context,
    build_source_bound_assay_observations,
    context_from_source_bound_facts,
    marts_target_id,
)


def test_source_bound_context_extracts_pH_temperature_and_cofactor():
    ctx=context_from_source_bound_facts([
        {
            "type":"pH",
            "evidence_text":"50 mM HEPES, pH 7.2",
            "explicit_quantities":[
                {"value":"50","unit":"mM","raw":"50 mM"},
                {"value":"7.2","unit":"pH","raw":"pH 7.2"},
            ],
        },
        {
            "type":"temperature",
            "evidence_text":"incubated at 30 °C",
            "explicit_quantities":[{"value":"30","unit":"°C","raw":"30 °C"}],
        },
        {
            "type":"metal_or_cofactor",
            "evidence_text":"7.5 mM MgCl2",
            "explicit_quantities":[{"value":"7.5","unit":"mM","raw":"7.5 mM"}],
        },
    ])
    assert ctx.ph == NumericInterval.point(7.2,"pH")
    assert ctx.temperature_c == NumericInterval.point(30.0,"°C")
    assert ctx.cofactors == ("Mg2+",)
    assert ctx.observed_dimensions == ("ph","temperature","cofactors")


def test_buffer_text_does_not_create_false_iron_cofactor():
    ctx=context_from_source_bound_facts([
        {
            "type":"buffer",
            "evidence_text":"50 mM HEPES buffer, pH 7.2",
            "explicit_quantities":[
                {"value":"50","unit":"mM","raw":"50 mM"},
                {"value":"7.2","unit":"pH","raw":"pH 7.2"},
            ],
        }
    ])
    assert ctx.cofactors==()


def _obs(oid: str,outcome: str,ctx: AssayContext) -> AssayObservation:
    return AssayObservation(
        observation_id=oid,
        enzyme_id="E1",
        reaction_id="R1",
        context=ctx,
        outcome=outcome,
        source_scope="pair_assay",
        source_uri="https://example.test/source",
        source_record="source:1",
        evidence_texts=("assay evidence",),
        target_binding_status="resolved",
    )


def test_context_assessment_never_turns_mismatch_or_missingness_into_negative():
    observed=AssayContext(
        ph=NumericInterval.point(7.0,"pH"),
        temperature_c=NumericInterval.point(30.0,"°C"),
        cofactors=("Mg2+",),
    )
    rows=[_obs("positive","reported_positive",observed)]
    exact=assess_pair_context(
        rows,
        AssayContext(
            ph=NumericInterval.point(7.0,"pH"),
            temperature_c=NumericInterval.point(30.0,"°C"),
            cofactors=("Mg2+",),
        ),
        enzyme_id="E1",reaction_id="R1",
    )
    assert exact.status=="supported"
    mismatch=assess_pair_context(
        rows,
        AssayContext(ph=NumericInterval.point(8.0,"pH")),
        enzyme_id="E1",reaction_id="R1",
    )
    assert mismatch.status=="unresolved"
    missing=assess_pair_context(
        rows,
        AssayContext(cofactors=("Mn2+",)),
        enzyme_id="E1",reaction_id="R1",
    )
    assert missing.status=="unresolved"


def test_only_matched_explicit_negative_can_contradict():
    ctx=AssayContext(ph=NumericInterval.point(6.5,"pH"))
    rows=[
        _obs("positive","reported_positive",ctx),
        _obs("negative","below_detection",ctx),
    ]
    conflict=assess_pair_context(
        rows,ctx,enzyme_id="E1",reaction_id="R1"
    )
    assert conflict.status=="conflicting"

    contrad=assess_pair_context(
        [_obs("negative","below_detection",ctx)],
        ctx,enzyme_id="E1",reaction_id="R1"
    )
    assert contrad.status=="contradicted"
    assert contrad.matched_contradiction_observations==("negative",)


def test_publication_promotion_gate_requires_catalytic_scope_and_single_target():
    target=marts_target_id("enzyme-source","A>>B")
    target_map={target:("E1","R1")}
    rows=[
        {
            "status":"ok",
            "paragraph_role":"catalytic_assay",
            "source_key":"PMC1:4",
            "source_url":"https://example.test/PMC1",
            "facts":[
                {
                    "type":"pH","evidence_text":"pH 6.5",
                    "explicit_quantities":[{"value":"6.5","unit":"pH","raw":"pH 6.5"}],
                    "source_span_verified":True,"scope_resolved":True,
                    "target_ids":[target],"target_assignment_status":"candidate",
                }
            ],
        },
        {
            "status":"ok",
            "paragraph_role":"mixed",
            "source_key":"PMC1:5",
            "source_url":"https://example.test/PMC1",
            "facts":[
                {
                    "type":"temperature","evidence_text":"30 °C",
                    "explicit_quantities":[{"value":"30","unit":"°C","raw":"30 °C"}],
                    "source_span_verified":True,"scope_resolved":True,
                    "target_ids":[target],"target_assignment_status":"candidate",
                }
            ],
        },
    ]
    out=build_source_bound_assay_observations(rows,target_map)
    assert len(out)==1
    assert out[0].enzyme_id=="E1"
    assert out[0].reaction_id=="R1"
    assert out[0].context.ph == NumericInterval.point(6.5,"pH")
