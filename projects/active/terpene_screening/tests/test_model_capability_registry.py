from projects.active.terpene_screening.model_capability_registry import (
    REQUIRED_BASELINE,
    validate_benchmark_rows,
    validate_final_model_manifest,
)


def test_every_reported_scenario_requires_enzymecage() -> None:
    rows = [{"scenario_id": "enzyme405", "model": "candidate", "status": "complete"}]
    errors = validate_benchmark_rows(rows)
    assert any("missing mandatory" in value for value in errors)


def test_enzymecage_na_requires_reason() -> None:
    rows = [{"scenario_id": "reactzyme_reaction_projected_double_cold", "model": REQUIRED_BASELINE, "status": "na"}]
    errors = validate_benchmark_rows(rows)
    assert any("incompatibility_reason" in value for value in errors)
    rows[0]["incompatibility_reason"] = "Original model requires pocket/structure features unavailable for this full candidate universe."
    assert validate_benchmark_rows(rows) == []


def test_strict_clean_routing_rejects_contaminated_expert() -> None:
    manifest = {
        "model_name": "Catalyst-Routed-v1-candidate",
        "model_role": "final_routed",
        "experts": [
            {"name": "clean", "contamination_status": "clean"},
            {"name": "production", "contamination_status": "contaminated"},
        ],
        "router": {"uses_target_test_labels": False, "selection_data": "nested development only"},
        "scenario_routing": [
            {"scenario_id": "reactzyme_reaction_projected_double_cold", "allowed_experts": ["clean", "production"]}
        ],
    }
    errors = validate_final_model_manifest(manifest)
    assert any("strict-clean" in value for value in errors)
    manifest["scenario_routing"][0]["allowed_experts"] = ["clean"]
    assert validate_final_model_manifest(manifest) == []


def test_expert_can_be_clean_for_enzyme405_but_unsupported_for_full_universe() -> None:
    manifest = {
        "model_name": "Catalyst-Routed-v1-candidate",
        "model_role": "final_routed",
        "experts": [
            {
                "name": "EnzymeCAGE",
                "contamination_status": "diagnostic_only",
                "scenario_status": {
                    "enzyme405": "clean",
                    "reactzyme_reaction_projected_double_cold": "unsupported",
                },
            }
        ],
        "router": {"uses_target_test_labels": False, "selection_data": "nested development only"},
        "scenario_routing": [{"scenario_id": "enzyme405", "allowed_experts": ["EnzymeCAGE"]}],
    }
    assert validate_final_model_manifest(manifest) == []
    manifest["scenario_routing"][0] = {
        "scenario_id": "reactzyme_reaction_projected_double_cold",
        "allowed_experts": ["EnzymeCAGE"],
    }
    errors = validate_final_model_manifest(manifest)
    assert any("unsupported" in value for value in errors)
    assert any("strict-clean" in value for value in errors)


def test_router_cannot_use_target_test_labels() -> None:
    manifest = {
        "model_name": "Catalyst-Routed-v1-candidate",
        "model_role": "final_routed",
        "experts": [{"name": "clean", "contamination_status": "clean"}],
        "router": {"uses_target_test_labels": True, "selection_data": "nested development only"},
        "scenario_routing": [{"scenario_id": "enzyme405", "allowed_experts": ["clean"]}],
    }
    errors = validate_final_model_manifest(manifest)
    assert any("uses_target_test_labels" in value for value in errors)


def test_directional_routes_reject_wrong_direction_expert() -> None:
    manifest = {
        "model_name": "Catalyst-Routed-v1-candidate",
        "model_role": "final_routed",
        "experts": [
            {"name": "r2e", "contamination_status": "clean", "directions": ["reaction_to_enzyme"]},
            {"name": "e2r", "contamination_status": "clean", "directions": ["enzyme_to_reaction"]},
        ],
        "router": {"uses_target_test_labels": False, "selection_data": "frozen internal development"},
        "scenario_routing": [{"scenario_id": "temporal_post2020_protein_cold", "direction": "enzyme_to_reaction", "allowed_experts": ["r2e"]}],
    }
    errors = validate_final_model_manifest(manifest)
    assert any("do not support direction" in value for value in errors)
    manifest["scenario_routing"][0]["allowed_experts"] = ["e2r"]
    assert validate_final_model_manifest(manifest) == []


def test_direction_must_belong_to_registered_scenario() -> None:
    manifest = {
        "model_name": "Catalyst-Routed-v1-candidate",
        "model_role": "final_routed",
        "experts": [{"name": "e2r", "contamination_status": "clean", "directions": ["enzyme_to_reaction"]}],
        "router": {"uses_target_test_labels": False, "selection_data": "frozen internal development"},
        "scenario_routing": [{"scenario_id": "enzyme405", "direction": "enzyme_to_reaction", "allowed_experts": ["e2r"]}],
    }
    errors = validate_final_model_manifest(manifest)
    assert any("not registered for scenario" in value for value in errors)
