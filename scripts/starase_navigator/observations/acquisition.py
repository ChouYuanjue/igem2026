from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MODE_BUDGET = {
    "fast": 1,
    "standard": 1,
    "deep": 2,
    "reproduce": 3,
}
MODE_ALIASES = {
    "auto": "standard",
    "normal": "standard",
    "research": "deep",
    "full": "reproduce",
    "reproducible": "reproduce",
}


@dataclass(frozen=True)
class MeasurementSpec:
    measurement_id: str
    entity_kind: str
    biological_meaning: str
    role: str
    acquisition_cost: int
    dependencies: tuple[str, ...] = ()
    online_policy: str = "query_extension"


PROTEIN_MEASUREMENTS = (
    MeasurementSpec(
        "protein_sequence",
        "protein",
        "primary amino-acid sequence",
        "base_observation",
        0,
        online_policy="required_input_or_catalog",
    ),
    MeasurementSpec(
        "global_esmc",
        "protein",
        "global sequence-state geometry",
        "factor_geometry",
        1,
        ("protein_sequence",),
    ),
    MeasurementSpec(
        "family_motif_context",
        "protein",
        "family-applicable catalytic local sequence context",
        "factor_geometry",
        1,
        ("protein_sequence", "global_esmc"),
    ),
    MeasurementSpec(
        "resolved_structure",
        "protein",
        "existing experimental or predicted whole-protein structure",
        "base_observation",
        2,
        ("protein_sequence",),
        "reuse_or_fetch; do_not_predict_de_novo_in_interactive_request",
    ),
    MeasurementSpec(
        "whole_3di",
        "protein",
        "whole-structure topology",
        "factor_geometry",
        2,
        ("resolved_structure",),
    ),
    MeasurementSpec(
        "pocket_detection",
        "protein",
        "candidate catalytic pocket geometry",
        "base_observation",
        2,
        ("resolved_structure",),
    ),
    MeasurementSpec(
        "pocket_local_esmc",
        "protein",
        "sequence state localised around the observed pocket",
        "factor_geometry",
        2,
        ("protein_sequence", "pocket_detection"),
    ),
    MeasurementSpec(
        "pocket_3di",
        "protein",
        "pocket-local structural topology",
        "factor_geometry",
        2,
        ("resolved_structure", "pocket_detection"),
    ),
    MeasurementSpec(
        "pocket_ot",
        "protein",
        "distributional geometry over observed pockets",
        "factor_geometry",
        2,
        ("resolved_structure", "pocket_detection"),
    ),
    MeasurementSpec(
        "de_novo_structure",
        "protein",
        "newly predicted structure when no reusable structure exists",
        "base_observation",
        3,
        ("protein_sequence",),
        "batch_only; publish into a later atlas version",
    ),
)

REACTION_MEASUREMENTS = (
    MeasurementSpec(
        "reaction_structure",
        "reaction",
        "canonical reactant/product molecular structure",
        "base_observation",
        0,
        online_policy="required_input_or_catalog",
    ),
    MeasurementSpec(
        "drfp",
        "reaction",
        "global reaction-difference chemistry",
        "factor_geometry",
        1,
        ("reaction_structure",),
    ),
    MeasurementSpec(
        "reactant_product_neighbourhood",
        "reaction",
        "global reactant- and product-side molecular neighbourhoods",
        "factor_geometry",
        1,
        ("reaction_structure",),
    ),
    MeasurementSpec(
        "atom_mapping",
        "reaction",
        "atom correspondence needed to localise the reaction centre",
        "base_observation",
        2,
        ("reaction_structure",),
    ),
    MeasurementSpec(
        "reaction_center_transition",
        "reaction",
        "changed-atom and changed-bond catalytic-local transition geometry",
        "mechanism_evidence",
        2,
        ("atom_mapping",),
        "evidence_first; not canonical factor geometry until non-destructive",
    ),
    MeasurementSpec(
        "reaction_literature_context",
        "reaction",
        "curated reaction references and known biochemical context",
        "mechanism_evidence",
        2,
        ("reaction_structure",),
        "research_workspace",
    ),
)


def normalize_observation_mode(value: str | None) -> str:
    mode = str(value or "standard").strip().lower().replace("-", "_")
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODE_BUDGET:
        raise ValueError(f"unsupported observation mode: {value}")
    return mode


def _plan_measurements(
    specs: tuple[MeasurementSpec, ...],
    *,
    mode: str,
    reference_cached: set[str],
    supplied: set[str],
) -> list[dict[str, Any]]:
    budget = MODE_BUDGET[mode]
    rows: list[dict[str, Any]] = []
    selected: set[str] = set(reference_cached) | set(supplied)

    # Dependency-ordered registry: every dependency appears before its consumer.
    for spec in specs:
        cached = spec.measurement_id in reference_cached
        provided = spec.measurement_id in supplied
        deps_ready = all(dep in selected for dep in spec.dependencies)
        within_budget = spec.acquisition_cost <= budget

        if cached:
            status = "reuse_cached"
            use_now = True
            reason = "already materialised in the versioned reference atlas"
        elif provided:
            status = "provided"
            use_now = True
            reason = "supplied by the query or entity resolver"
        elif within_budget and deps_ready:
            status = "planned_now"
            use_now = True
            reason = "eligible for acquisition within the selected observation budget; execution is reported separately"
        elif spec.acquisition_cost > budget:
            status = "defer"
            use_now = False
            reason = "available only at a deeper observation budget"
        else:
            status = "blocked_by_dependency"
            use_now = False
            reason = "a prerequisite observation is unavailable at this budget"

        # Mechanistic observations are acquired when useful but do not silently
        # alter canonical ranking geometry.
        ranking_role = spec.role
        if spec.role == "mechanism_evidence":
            ranking_role = "evidence_only"

        if use_now:
            selected.add(spec.measurement_id)

        rows.append(
            {
                **asdict(spec),
                "status": status,
                "selected_now": use_now,
                "available_now": status in {"reuse_cached", "provided"},
                "planned_now": status == "planned_now",
                "executed_now": False,
                "ranking_role": ranking_role,
                "reason": reason,
            }
        )
    return rows


def apply_observation_execution(
    plan: dict[str, Any],
    *,
    executed_measurements: list[str] | tuple[str, ...] | set[str] = (),
    failed_measurements: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a copy of a plan annotated with observations actually executed.

    Planning and execution are intentionally distinct. A measurement may be
    eligible under the requested budget but must not be presented as computed
    until the runtime reports it here.
    """
    executed = {str(value) for value in executed_measurements}
    failures = {str(key): str(value) for key, value in (failed_measurements or {}).items()}
    updated = {**plan}
    rows: list[dict[str, Any]] = []
    for raw in plan.get("measurements", []):
        row = dict(raw)
        mid = str(row.get("measurement_id") or "")
        if mid in executed:
            row["status"] = "computed_now"
            row["available_now"] = True
            row["planned_now"] = False
            row["executed_now"] = True
            row["reason"] = "computed successfully for this query"
        elif mid in failures:
            row["status"] = "execution_failed"
            row["available_now"] = False
            row["planned_now"] = False
            row["executed_now"] = False
            row["reason"] = failures[mid]
        rows.append(row)
    updated["measurements"] = rows
    updated["executed_measurements"] = sorted(executed)
    updated["failed_measurements"] = dict(failures)
    updated["selected_factor_measurements"] = [
        str(row["measurement_id"]) for row in rows
        if row.get("ranking_role") == "factor_geometry" and row.get("available_now")
    ]
    updated["selected_evidence_measurements"] = [
        str(row["measurement_id"]) for row in rows
        if row.get("ranking_role") == "evidence_only" and row.get("available_now")
    ]
    updated["planned_factor_measurements"] = [
        str(row["measurement_id"]) for row in rows
        if row.get("ranking_role") == "factor_geometry" and row.get("planned_now")
    ]
    updated["planned_evidence_measurements"] = [
        str(row["measurement_id"]) for row in rows
        if row.get("ranking_role") == "evidence_only" and row.get("planned_now")
    ]
    return updated


def build_observation_plan(
    *,
    direction: str,
    mode: str | None = None,
    query_is_reference_entity: bool,
    query_has_sequence: bool = False,
    query_has_reaction_structure: bool = False,
    cached_measurements: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    """Return a frontend-facing acquisition plan without changing ranking.

    A reference entity may reuse every measurement already materialised for that
    atlas version. An external query is attached out-of-sample: only query-side
    measurements are acquired, while the reference atlas remains immutable.
    """
    normalized_mode = normalize_observation_mode(mode)
    direction = str(direction or "").strip()
    if direction not in {"reaction_to_enzyme", "enzyme_to_reaction"}:
        raise ValueError(f"unsupported retrieval direction: {direction}")

    kind = "reaction" if direction == "reaction_to_enzyme" else "protein"
    specs = REACTION_MEASUREMENTS if kind == "reaction" else PROTEIN_MEASUREMENTS
    cached = set(str(x) for x in (cached_measurements or []) if str(x))
    supplied: set[str] = set()
    if kind == "protein" and query_has_sequence:
        supplied.add("protein_sequence")
    if kind == "reaction" and query_has_reaction_structure:
        supplied.add("reaction_structure")

    measurements = _plan_measurements(
        specs,
        mode=normalized_mode,
        reference_cached=cached,
        supplied=supplied,
    )

    selected_geometry = [
        row["measurement_id"]
        for row in measurements
        if row["available_now"] and row["ranking_role"] == "factor_geometry"
    ]
    selected_evidence = [
        row["measurement_id"]
        for row in measurements
        if row["available_now"] and row["ranking_role"] == "evidence_only"
    ]
    planned_geometry = [
        row["measurement_id"]
        for row in measurements
        if row["planned_now"] and row["ranking_role"] == "factor_geometry"
    ]
    planned_evidence = [
        row["measurement_id"]
        for row in measurements
        if row["planned_now"] and row["ranking_role"] == "evidence_only"
    ]
    deferred = [row["measurement_id"] for row in measurements if not row["selected_now"]]

    return {
        "version": "observation-acquisition-v1",
        "mode": normalized_mode,
        "budget_class": MODE_BUDGET[normalized_mode],
        "query_entity_kind": kind,
        "query_is_reference_entity": bool(query_is_reference_entity),
        "atlas_policy": (
            "reuse_versioned_reference_atlas"
            if query_is_reference_entity
            else "out_of_sample_query_extension; do_not_rebuild_reference_atlas_per_request"
        ),
        "field_policy": (
            "ranking may start as soon as the minimum connected query geometry is available; "
            "deeper measurements refine the same partially observed manifold rather than select another model"
        ),
        "selected_factor_measurements": selected_geometry,
        "selected_evidence_measurements": selected_evidence,
        "planned_factor_measurements": planned_geometry,
        "planned_evidence_measurements": planned_evidence,
        "executed_measurements": [],
        "failed_measurements": {},
        "deferred_measurements": deferred,
        "measurements": measurements,
        "persistence_policy": (
            "new verified positive pairs update the correspondence field exactly without retraining; "
            "new molecular entities are promoted into a future atlas version only by an offline rebuild"
        ),
    }
