from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

REQUIRED_BASELINE = "EnzymeCAGE"
FINAL_MODEL_ROLE = "final_routed"


@dataclass(frozen=True)
class BenchmarkScenario:
    scenario_id: str
    family: str
    role: str
    strict_clean: bool
    confirmatory: bool
    candidate_scope: str
    directions: tuple[str, ...]
    enz_cage_mode: str
    note: str = ""


DEFAULT_SCENARIOS: tuple[BenchmarkScenario, ...] = (
    BenchmarkScenario(
        "rhea128_to141_sprot_strict_double_cold", "broad_rhea_snapshot", "fresh_external_temporal", True, True,
        "full_general_universe", ("reaction_to_enzyme",), "required_or_na_with_reason",
        "Fresh official Rhea release128 (2023-07-12) to release141 (2026-06-10) Swiss-Prot snapshot-delta evaluation using direction-specific RHEA_ID and the exact clean2023-compatible unambiguous protein-alias mapping; query selection is performance-blind and requires strict clean2023 protein+reaction cold support."
    ),
    BenchmarkScenario(
        "enzyme405", "EnzymeCAGE", "primary_external", True, True,
        "official_15921_pair_reservoir", ("reaction_to_enzyme",), "author_native",
        "Primary confirmatory novel-enzyme benchmark; router/model selection must not use target labels.",
    ),
    BenchmarkScenario(
        "enzymecage_official", "EnzymeCAGE", "author_native", False, True,
        "official_pair_reservoir", ("reaction_to_enzyme",), "author_native",
        "Report author-native EnzymeCAGE metrics and common IR metrics on the same reservoir.",
    ),
    BenchmarkScenario(
        "reactzyme_reaction_projected_double_cold", "broad_rhea", "primary_generalization", True, True,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "Strict projected double-cold benchmark; contaminated production experts are forbidden.",
    ),
    BenchmarkScenario(
        "reactzyme_time_projected_protein_cold", "broad_rhea", "protein_cold", True, True,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
    ),
    BenchmarkScenario(
        "reactzyme_enzyme_projected_protein_cold", "broad_rhea", "protein_cold", True, True,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
    ),
    BenchmarkScenario(
        "temporal_post2020_double_cold", "broad_rhea", "temporal_double_cold", True, True,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "Temporal stress test, not a strict historical-snapshot claim without source snapshots.",
    ),
    BenchmarkScenario(
        "temporal_post2020_protein_cold", "broad_rhea", "temporal_protein_cold", True, True,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "Pre-reveal confirmatory cell for the frozen direction-specific representation router; temporal means UniProt creation-date extrapolation, not a strict historical source snapshot.",
    ),
    BenchmarkScenario(
        "broad_reaction_cold_protein_seen", "broad_rhea", "reaction_cold", True, False,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
    ),
    BenchmarkScenario(
        "both_seen_exact_pair_holdout", "broad_rhea", "relation_completion", False, False,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "Relation-completion sanity check; do not describe as cold generalization.",
    ),
    BenchmarkScenario(
        "production_known_recovery", "production", "retention", False, False,
        "full_general_universe", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "Retention/coverage diagnostic only; contaminated production model is allowed here.",
    ),
    BenchmarkScenario(
        "tps_frozen_double_cold", "TPS", "domain_specialist", True, True,
        "frozen_tps_candidate_pool", ("enzyme_to_reaction", "reaction_to_enzyme"), "required_or_na_with_reason",
        "TPS remains one domain-specific stress test, not the sole headline benchmark.",
    ),
)


def scenario_map(scenarios: Iterable[BenchmarkScenario] = DEFAULT_SCENARIOS) -> dict[str, BenchmarkScenario]:
    return {item.scenario_id: item for item in scenarios}


def validate_benchmark_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    scenarios: Iterable[BenchmarkScenario] = DEFAULT_SCENARIOS,
    require_all_registered: bool = False,
) -> list[str]:
    """Validate reporting rows without pretending an incompatible baseline has a score.

    Every reported scenario must contain an EnzymeCAGE row. A baseline may be marked
    ``na`` only when an explicit incompatibility reason is recorded. This makes the
    baseline impossible to silently drop from a table while preserving honest task
    compatibility reporting.
    """
    known = scenario_map(scenarios)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    errors: list[str] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id", "")).strip()
        if not scenario_id:
            errors.append("row missing scenario_id")
            continue
        if scenario_id not in known:
            errors.append(f"unregistered scenario: {scenario_id}")
            continue
        grouped.setdefault(scenario_id, []).append(row)

    if require_all_registered:
        for scenario_id in known:
            if scenario_id not in grouped:
                errors.append(f"registered scenario missing from report: {scenario_id}")

    for scenario_id, group in grouped.items():
        cage_rows = [row for row in group if str(row.get("model", "")).strip() == REQUIRED_BASELINE]
        if not cage_rows:
            errors.append(f"{scenario_id}: missing mandatory {REQUIRED_BASELINE} baseline row")
            continue
        for row in cage_rows:
            status = str(row.get("status", "complete")).strip().lower()
            if status not in {"complete", "na"}:
                errors.append(f"{scenario_id}: invalid {REQUIRED_BASELINE} status={status!r}")
            if status == "na" and not str(row.get("incompatibility_reason", "")).strip():
                errors.append(f"{scenario_id}: {REQUIRED_BASELINE} N/A requires incompatibility_reason")
    return errors


def validate_final_model_manifest(manifest: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if str(manifest.get("model_role", "")) != FINAL_MODEL_ROLE:
        errors.append(f"model_role must be {FINAL_MODEL_ROLE!r}")
    if not str(manifest.get("model_name", "")).strip():
        errors.append("model_name is required")
    experts = manifest.get("experts")
    if not isinstance(experts, list) or not experts:
        errors.append("experts must be a non-empty list")
        experts = []
    expert_status: dict[str, str] = {}
    expert_scenario_status: dict[str, dict[str, str]] = {}
    allowed_status = {"clean", "contaminated", "diagnostic_only", "unsupported"}
    scenarios = scenario_map()
    for expert in experts:
        if not isinstance(expert, Mapping):
            errors.append("each expert must be an object")
            continue
        name = str(expert.get("name", "")).strip()
        if not name:
            errors.append("expert missing name")
            continue
        status = str(expert.get("contamination_status", "")).strip().lower()
        if status not in allowed_status:
            errors.append(f"expert {name}: invalid contamination_status={status!r}")
        expert_status[name] = status
        overrides = expert.get("scenario_status", {})
        if not isinstance(overrides, Mapping):
            errors.append(f"expert {name}: scenario_status must be an object")
            overrides = {}
        parsed_overrides: dict[str, str] = {}
        for scenario_id, raw_status in overrides.items():
            scenario_id = str(scenario_id)
            scenario_status = str(raw_status).strip().lower()
            if scenario_id not in scenarios:
                errors.append(f"expert {name}: scenario_status uses unregistered scenario {scenario_id!r}")
            if scenario_status not in allowed_status:
                errors.append(f"expert {name}: invalid scenario status {scenario_status!r} for {scenario_id}")
            parsed_overrides[scenario_id] = scenario_status
        expert_scenario_status[name] = parsed_overrides

    router = manifest.get("router")
    if not isinstance(router, Mapping):
        errors.append("router must be an object")
        router = {}
    if router.get("uses_target_test_labels") is not False:
        errors.append("router.uses_target_test_labels must be false")
    if not str(router.get("selection_data", "")).strip():
        errors.append("router.selection_data is required")

    routing = manifest.get("scenario_routing")
    if not isinstance(routing, list) or not routing:
        errors.append("scenario_routing must be a non-empty list")
        routing = []
    for entry in routing:
        if not isinstance(entry, Mapping):
            errors.append("scenario routing entry must be an object")
            continue
        scenario_id = str(entry.get("scenario_id", "")).strip()
        scenario = scenarios.get(scenario_id)
        if scenario is None:
            errors.append(f"scenario routing uses unregistered scenario: {scenario_id}")
            continue
        allowed = entry.get("allowed_experts")
        if not isinstance(allowed, list) or not allowed:
            errors.append(f"{scenario_id}: allowed_experts must be non-empty")
            continue
        allowed_names = list(map(str, allowed))
        unknown = [name for name in allowed_names if name not in expert_status]
        if unknown:
            errors.append(f"{scenario_id}: unknown experts {unknown}")
            continue
        effective_status = {
            name: expert_scenario_status.get(name, {}).get(scenario_id, expert_status.get(name, ""))
            for name in allowed_names
        }
        unsupported = [name for name, status in effective_status.items() if status == "unsupported"]
        if unsupported:
            errors.append(f"{scenario_id}: routing includes unsupported experts {unsupported}")
        if scenario.strict_clean:
            dirty = [name for name, status in effective_status.items() if status != "clean"]
            if dirty:
                errors.append(f"{scenario_id}: strict-clean routing includes non-clean experts {dirty}")
    return errors


def registry_payload() -> dict[str, object]:
    return {
        "required_baseline": REQUIRED_BASELINE,
        "final_model_role": FINAL_MODEL_ROLE,
        "policy": {
            "enzymecage_every_scenario": True,
            "na_requires_incompatibility_reason": True,
            "strict_clean_masks_contaminated_experts": True,
            "confirmatory_target_labels_for_router_selection": False,
            "final_model_is_first_class_routed_system": True,
        },
        "scenarios": [asdict(item) for item in DEFAULT_SCENARIOS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit/validate the Catalyst routed-model capability registry.")
    parser.add_argument("--write-registry", type=Path)
    parser.add_argument("--report-json", type=Path, help="JSON list of benchmark rows to validate")
    parser.add_argument("--manifest-json", type=Path, help="final routed-model manifest to validate")
    parser.add_argument("--require-all-registered", action="store_true")
    args = parser.parse_args()
    if args.write_registry:
        args.write_registry.parent.mkdir(parents=True, exist_ok=True)
        args.write_registry.write_text(json.dumps(registry_payload(), indent=2, ensure_ascii=False), encoding="utf-8")
    errors: list[str] = []
    if args.report_json:
        rows = json.loads(args.report_json.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("report JSON must be a list")
        errors.extend(validate_benchmark_rows(rows, require_all_registered=args.require_all_registered))
    if args.manifest_json:
        manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
        errors.extend(validate_final_model_manifest(manifest))
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(registry_payload(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
