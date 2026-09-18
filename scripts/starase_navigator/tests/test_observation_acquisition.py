import pytest
from scripts.starase_navigator.observations.acquisition import apply_observation_execution, build_observation_plan, normalize_observation_mode

def by_id(plan):
    return {row["measurement_id"]: row for row in plan["measurements"]}

def test_fast_external_protein_starts_from_sequence_geometry_and_defers_structure():
    plan=build_observation_plan(
        direction="enzyme_to_reaction", mode="fast",
        query_is_reference_entity=False, query_has_sequence=True,
    )
    rows=by_id(plan)
    assert rows["protein_sequence"]["status"]=="provided"
    assert rows["global_esmc"]["status"]=="planned_now"
    assert rows["global_esmc"]["available_now"] is False
    assert rows["resolved_structure"]["selected_now"] is False
    assert plan["atlas_policy"].startswith("out_of_sample")

def test_deep_external_protein_uses_reusable_structure_path_but_not_de_novo_prediction():
    plan=build_observation_plan(
        direction="enzyme_to_reaction", mode="deep",
        query_is_reference_entity=False, query_has_sequence=True,
    )
    rows=by_id(plan)
    assert rows["resolved_structure"]["status"]=="planned_now"
    assert rows["pocket_detection"]["status"]=="planned_now"
    assert rows["pocket_ot"]["status"]=="planned_now"
    assert rows["de_novo_structure"]["selected_now"] is False

def test_reproduce_allows_batch_de_novo_structure():
    plan=build_observation_plan(
        direction="enzyme_to_reaction", mode="reproduce",
        query_is_reference_entity=False, query_has_sequence=True,
    )
    assert by_id(plan)["de_novo_structure"]["status"]=="planned_now"

def test_reaction_center_is_evidence_only_even_when_deep():
    plan=build_observation_plan(
        direction="reaction_to_enzyme", mode="deep",
        query_is_reference_entity=False, query_has_reaction_structure=True,
    )
    row=by_id(plan)["reaction_center_transition"]
    assert row["status"]=="planned_now"
    assert row["ranking_role"]=="evidence_only"

def test_cached_measurement_is_reused_even_in_fast_mode():
    plan=build_observation_plan(
        direction="enzyme_to_reaction", mode="fast",
        query_is_reference_entity=True,
        cached_measurements={"protein_sequence","global_esmc","resolved_structure","whole_3di","pocket_detection","pocket_ot"},
    )
    rows=by_id(plan)
    assert rows["pocket_ot"]["status"]=="reuse_cached"
    assert rows["pocket_ot"]["available_now"] is True

def test_aliases_and_bad_mode():
    assert normalize_observation_mode("research")=="deep"
    assert normalize_observation_mode("full")=="reproduce"
    with pytest.raises(ValueError):
        normalize_observation_mode("turbo")

def test_execution_annotation_separates_plan_from_completed_measurements():
    plan=build_observation_plan(
        direction="reaction_to_enzyme", mode="standard",
        query_is_reference_entity=False, query_has_reaction_structure=True,
    )
    assert by_id(plan)["drfp"]["status"]=="planned_now"
    done=apply_observation_execution(
        plan, executed_measurements={"drfp","reactant_product_neighbourhood"}
    )
    rows=by_id(done)
    assert rows["drfp"]["status"]=="computed_now"
    assert rows["drfp"]["executed_now"] is True
    assert set(done["selected_factor_measurements"])=={"drfp","reactant_product_neighbourhood"}
    assert done["planned_factor_measurements"]==[]
