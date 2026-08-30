"""Read-only projection of production retrieval routes for Catalyst audit views."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs/production_routes/terpene_v1.yaml"
DEFAULT_TAXONOMY_SUMMARY = ROOT / "data/terpene_taxonomy_scope/summary.json"


ROUTE_MODULES: dict[str, list[str]] = {
    "r2e-current": [
        "r2e-query", "r2e-shot", "r2e-scope", "r2e-encoder",
        "r2e-universe", "r2e-taxonomy", "r2e-router", "r2e-shared", "r2e-rank",
        "r2e-trust", "r2e-output",
    ],
    "r2e-external-top3": [
        "r2e-query", "r2e-shot", "r2e-scope", "r2e-encoder",
        "r2e-universe", "r2e-taxonomy", "r2e-router", "r2e-loss075", "r2e-rank",
        "r2e-trust", "r2e-output",
    ],
    "r2e-external-residual": [
        "r2e-query", "r2e-shot", "r2e-scope", "r2e-encoder",
        "r2e-universe", "r2e-taxonomy", "r2e-router", "r2e-residual", "r2e-rank",
        "r2e-trust", "r2e-output",
    ],
    "e2r-current": [
        "e2r-query", "e2r-shot", "e2r-scope", "e2r-encoder",
        "e2r-universe", "e2r-router", "e2r-current", "e2r-rank",
        "e2r-trust", "e2r-output",
    ],
    "e2r-external-top3": [
        "e2r-query", "e2r-shot", "e2r-scope", "e2r-encoder",
        "e2r-universe", "e2r-router", "e2r-neighbor", "e2r-rank",
        "e2r-trust", "e2r-output",
    ],
    "e2r-external-top10": [
        "e2r-query", "e2r-shot", "e2r-scope", "e2r-encoder",
        "e2r-universe", "e2r-router", "e2r-neighbor", "e2r-hardneg",
        "e2r-rrf10", "e2r-rank", "e2r-trust", "e2r-output",
    ],
    "e2r-external-top20": [
        "e2r-query", "e2r-shot", "e2r-scope", "e2r-encoder",
        "e2r-universe", "e2r-router", "e2r-neighbor", "e2r-dualkernel",
        "e2r-rrf20", "e2r-rank", "e2r-trust", "e2r-output",
    ],
}

FEWSHOT_MODULES = {
    "reaction_to_enzyme": [
        "r2e-query", "r2e-shot", "r2e-scope", "r2e-encoder",
        "r2e-universe", "r2e-taxonomy", "r2e-router", "r2e-active-expert",
        "r2e-seed", "r2e-guidance-merge", "r2e-seed-mask",
        "r2e-rank", "r2e-trust", "r2e-output",
    ],
    "enzyme_to_reaction": [
        "e2r-query", "e2r-shot", "e2r-scope", "e2r-encoder",
        "e2r-universe", "e2r-router", "e2r-active-expert",
        "e2r-seed", "e2r-guidance-merge", "e2r-seed-mask",
        "e2r-rank", "e2r-trust", "e2r-output",
    ],
}


def _family(direction: str, scope: str, objective: str) -> str:
    if direction == "reaction_to_enzyme":
        if scope == "current":
            return "r2e-current"
        return "r2e-external-top3" if objective == "top3" else "r2e-external-residual"
    if scope == "current":
        return "e2r-current"
    return {
        "top3": "e2r-external-top3",
        "top10": "e2r-external-top10",
        "top20": "e2r-external-top20",
    }[objective]


def _route_use_case(direction: str, scope: str, objective: str) -> str:
    if direction == "reaction_to_enzyme":
        if scope == "current":
            return {
                "top3": "A known terpene reaction needs a very small catalyst shortlist for rapid experimental triage.",
                "top10": "A known terpene reaction needs a balanced enzyme panel for the first wet-lab round.",
                "top20": "A known terpene reaction needs a broad screening panel; validated rescue candidates may also be included.",
            }[objective]
        return {
            "top3": "A new or unregistered reaction needs the three most focused enzyme hypotheses.",
            "top10": "A new reaction needs a broader enzyme search while keeping the shortlist experimentally manageable.",
            "top20": "A new reaction needs wide catalyst exploration before diversity and assay constraints are applied.",
        }[objective]
    if scope == "current":
        return {
            "top3": "A known terpene synthase needs a concise functional annotation with three leading reactions.",
            "top10": "A known enzyme needs a broader activity profile for annotation or assay planning.",
            "top20": "A known enzyme needs a wide promiscuity map for pathway design or substrate-panel screening.",
        }[objective]
    return {
        "top3": "A new protein sequence needs a first set of likely terpene reactions.",
        "top10": "A new protein needs a balanced activity shortlist supported by two neural ranking views.",
        "top20": "A new protein needs broad promiscuity exploration using neural and graph-based evidence.",
    }[objective]


def _route_description(direction: str, scope: str, objective: str, retrieval: str) -> str:
    if direction == "reaction_to_enzyme" and scope == "current":
        return "A paired neural model compares the known reaction with every candidate enzyme and orders the proteins by predicted compatibility."
    if direction == "reaction_to_enzyme" and objective == "top3":
        return "A focused neural ensemble, tuned for short new-reaction searches, scores candidate enzymes directly."
    if direction == "reaction_to_enzyme":
        return "The new reaction is represented by an exact chemical fingerprint plus a learned correction before candidate enzymes are ranked."
    if scope == "current":
        return "A neural model specialized for enzyme-to-reaction prediction ranks reactions for an enzyme already represented in the reference data."
    if objective == "top3":
        return "The model combines a direct prediction for the new protein with activity evidence transferred from five related reference proteins."
    if objective == "top10":
        return "Two independent neural rankings are produced—one from the main model and one trained on difficult alternatives—then combined by reciprocal-rank fusion (RRF)."
    if objective == "top20":
        return "A neural ranking is combined with protein similarity, reaction similarity and the known association network using reciprocal-rank fusion (RRF)."
    return retrieval


def _overlay(
    *,
    key: str,
    route_id_pattern: str,
    direction: str,
    scope: str,
    shot_mode: str,
    objective: str,
    retrieval: str,
    family: str,
    modules: list[str],
    description: str,
    reliability_scope: str,
    conformal_scope: str,
    modifier_suffix: str | None,
    availability: str,
    category: str,
    use_case: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "route_id_pattern": route_id_pattern,
        "direction": direction,
        "scope": scope,
        "shot_mode": shot_mode,
        "objective": objective,
        "retrieval": retrieval,
        "family": family,
        "modules": modules,
        "description": description,
        "reliability_scope": reliability_scope,
        "conformal_scope": conformal_scope,
        "modifier_suffix": modifier_suffix,
        "availability": availability,
        "category": category,
        "use_case": use_case,
    }


def build_route_catalog(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    deployments = payload["deployments"]
    routes: list[dict[str, Any]] = []
    for direction, scopes in payload["routes"].items():
        for scope, objectives in scopes.items():
            for objective, raw_spec in objectives.items():
                spec = dict(raw_spec)
                family = _family(direction, scope, objective)
                settings = {
                    key: value
                    for key, value in spec.items()
                    if key not in {
                        "route_id", "deployment", "secondary_deployment",
                        "auxiliary_deployment", "retrieval",
                    }
                }
                for key, value in dict(payload.get("policies") or {}).items():
                    settings.setdefault(str(key), value)
                routes.append({
                    "key": str(spec["route_id"]),
                    "route_id": str(spec["route_id"]),
                    "direction": direction,
                    "scope": scope,
                    "shot_mode": "zero_shot",
                    "objective": objective,
                    "retrieval": str(spec.get("retrieval", "direct")),
                    "family": family,
                    "modules": ROUTE_MODULES[family],
                    "deployment_key": str(spec["deployment"]),
                    "deployment": str(deployments[str(spec["deployment"])]),
                    "secondary_deployment": (
                        str(deployments[str(spec["secondary_deployment"])])
                        if spec.get("secondary_deployment") else None
                    ),
                    "auxiliary_deployment": (
                        str(deployments[str(spec["auxiliary_deployment"])])
                        if spec.get("auxiliary_deployment") else None
                    ),
                    "settings": settings,
                    "description": _route_description(
                        direction, scope, objective, str(spec.get("retrieval", "direct"))
                    ),
                    "use_case": _route_use_case(direction, scope, objective),
                    "reliability_scope": "external_zero_shot" if scope == "external" else "not_applicable_current_entity",
                    "conformal_scope": "external_zero_shot" if scope == "external" else "not_applicable_current_entity",
                    "modifier_suffix": None,
                    "availability": "portal",
                    "category": "manifest_route",
                })

    overlays = [
        _overlay(
            key="r2e-fewshot-guidance",
            route_id_pattern="r2e-{current|external}-top{3|10|20}-v1+fewshot",
            direction="reaction_to_enzyme",
            scope="any",
            shot_mode="few_shot",
            objective="top3|top10|top20",
            retrieval="hybrid",
            family="r2e-fewshot",
            modules=FEWSHOT_MODULES["reaction_to_enzyme"],
            description="The active retrieval expert provides the main candidate ranking; verified positive enzymes add a smaller protein-space guidance signal, and the supplied positives are removed from the returned list.",
            reliability_scope="not_applicable_few_shot",
            conformal_scope="not_applicable_few_shot",
            modifier_suffix="fewshot",
            availability="portal",
            category="execution_path",
            use_case="Default when verified positive catalyst enzymes are available: keep the active expert ranking while using the positives as few-shot guidance, without returning the seeds themselves.",
        ),
        _overlay(
            key="e2r-fewshot-guidance",
            route_id_pattern="e2r-{current|external}-top{3|10|20}-*-v1+fewshot",
            direction="enzyme_to_reaction",
            scope="any",
            shot_mode="few_shot",
            objective="top3|top10|top20",
            retrieval="hybrid",
            family="e2r-fewshot",
            modules=FEWSHOT_MODULES["enzyme_to_reaction"],
            description="The active enzyme-to-reaction expert provides the main ranking; verified reactions add a smaller learned reaction-space guidance signal, and the supplied positives are removed from the returned list.",
            reliability_scope="not_applicable_few_shot",
            conformal_scope="not_applicable_few_shot",
            modifier_suffix="fewshot",
            availability="portal",
            category="execution_path",
            use_case="Default when verified reactions are available: retain direct model evidence while using known activities as few-shot guidance for activity expansion.",
        ),
        _overlay(
            key="r2e-known-association-mask-overlay",
            route_id_pattern="r2e-<base-route>+masked",
            direction="reaction_to_enzyme",
            scope="external",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="known_association_mask_before_ranking",
            family="masked-overlay",
            modules=["r2e-taxonomy", "r2e-known-mask", "r2e-rank"],
            description="In vectorized registry discovery, enzymes already linked to the reaction are removed from eligibility before the final Top-K is selected. This prevents known associations from reappearing as discoveries.",
            reliability_scope="not_applicable_known_associations_masked",
            conformal_scope="not_applicable_masked_discovery",
            modifier_suffix="masked",
            availability="batch_only",
            category="modifier",
            use_case="Use this in registry-wide discovery batches when the goal is to surface novel enzyme hypotheses rather than reproduce already known reaction–enzyme links.",
        ),
        _overlay(
            key="e2r-zero-shot-mask-overlay",
            route_id_pattern="e2r-<base-route>+masked",
            direction="enzyme_to_reaction",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="route_preserved_then_masked",
            family="masked-overlay",
            modules=["e2r-mask-only"],
            description="Specified reactions are hidden from the returned list without being treated as positive examples; the underlying scoring strategy stays unchanged.",
            reliability_scope="not_applicable_known_associations_masked",
            conformal_scope="not_applicable_masked_discovery",
            modifier_suffix="masked",
            availability="portal",
            category="modifier",
            use_case="Use this to hide already known, undesired or previously tested reactions while preserving the underlying ranking route.",
        ),
        _overlay(
            key="r2e-temporary-universe-overlay",
            route_id_pattern="r2e-<base-route>+temporary-universe",
            direction="reaction_to_enzyme",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="candidate_universe_extension",
            family="temporary-universe",
            modules=["r2e-universe"],
            description="Temporarily adds extra enzyme candidates to the collection being searched without retraining the model.",
            reliability_scope="route_dependent",
            conformal_scope="not_applicable_temporary_universe",
            modifier_suffix="temporary-universe",
            availability="cli_only",
            category="modifier",
            use_case="Use this in specialist analyses to include enzyme candidates that have not yet been added to the standard search collection.",
        ),
        _overlay(
            key="e2r-temporary-universe-overlay",
            route_id_pattern="e2r-<base-route>+temporary-universe",
            direction="enzyme_to_reaction",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="candidate_universe_extension",
            family="temporary-universe",
            modules=["e2r-universe"],
            description="Temporarily adds extra reactions to the collection being searched. Some graph-based support may be unavailable when those new records are not aligned with the stored similarity matrices.",
            reliability_scope="route_dependent",
            conformal_scope="not_applicable_temporary_universe",
            modifier_suffix="temporary-universe",
            availability="cli_only",
            category="modifier",
            use_case="Use this in specialist analyses to include reaction candidates that have not yet been added to the standard search collection.",
        ),
        _overlay(
            key="r2e-manual-override-overlay",
            route_id_pattern="r2e-<base-route>+manual",
            direction="reaction_to_enzyme",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="manual_override",
            family="manual-override",
            modules=["r2e-router"],
            description="Lets model developers choose a non-standard model or scoring mode for controlled research comparisons.",
            reliability_scope="not_applicable_manual_override",
            conformal_scope="not_applicable_manual_override",
            modifier_suffix="manual",
            availability="cli_only",
            category="modifier",
            use_case="Use this only when model developers need to compare alternative scoring strategies or diagnose a route.",
        ),
        _overlay(
            key="e2r-manual-override-overlay",
            route_id_pattern="e2r-<base-route>+manual",
            direction="enzyme_to_reaction",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="manual_override",
            family="manual-override",
            modules=["e2r-router"],
            description="Lets model developers choose a non-standard model or scoring mode for controlled research comparisons.",
            reliability_scope="not_applicable_manual_override",
            conformal_scope="not_applicable_manual_override",
            modifier_suffix="manual",
            availability="cli_only",
            category="modifier",
            use_case="Use this only when model developers need to compare alternative scoring strategies or diagnose a route.",
        ),
        _overlay(
            key="r2e-eukaryote-only-overlay",
            route_id_pattern="r2e-<base-route>+eukaryote-only",
            direction="reaction_to_enzyme",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="taxonomy_candidate_filter",
            family="taxonomy-scope",
            modules=["r2e-taxonomy"],
            description="Restricts the enzyme candidate matrix before scoring to locally classified eukaryotic proteins. Unknown, viral and prokaryotic candidates are excluded.",
            reliability_scope="not_applicable_taxonomy_restricted",
            conformal_scope="not_applicable_taxonomy_restricted",
            modifier_suffix="eukaryote-only",
            availability="portal",
            category="modifier",
            use_case="Use this when the experimental host or biological question requires candidate enzymes from eukaryotic organisms only.",
        ),
        _overlay(
            key="r2e-prokaryote-only-overlay",
            route_id_pattern="r2e-<base-route>+prokaryote-only",
            direction="reaction_to_enzyme",
            scope="any",
            shot_mode="zero_shot",
            objective="top3|top10|top20",
            retrieval="taxonomy_candidate_filter",
            family="taxonomy-scope",
            modules=["r2e-taxonomy"],
            description="Restricts the enzyme candidate matrix before scoring to locally classified bacterial, archaeal and cyanobacterial proteins. Unknown, viral and eukaryotic candidates are excluded.",
            reliability_scope="not_applicable_taxonomy_restricted",
            conformal_scope="not_applicable_taxonomy_restricted",
            modifier_suffix="prokaryote-only",
            availability="portal",
            category="modifier",
            use_case="Use this when the experimental system should only consider prokaryotic enzyme candidates.",
        ),
        _overlay(
            key="r2e-cage-rescue-overlay",
            route_id_pattern="r2e-current-top20-v1 [selection_source=cage_rescue]",
            direction="reaction_to_enzyme",
            scope="current",
            shot_mode="zero_shot",
            objective="top20",
            retrieval="conditional_result_assembly",
            family="cage-rescue",
            modules=["r2e-cage"],
            description="When independent structure-based evidence is available, up to five supported candidates can be added to a broad enzyme shortlist.",
            reliability_scope="not_applicable_current_entity",
            conformal_scope="not_applicable_current_entity",
            modifier_suffix=None,
            availability="conditional",
            category="conditional_path",
            use_case="For a known reaction, this can supplement a broad Top-20 screen with candidates supported by an independent structure-based analysis.",
        ),
    ]

    direction_order = {"reaction_to_enzyme": 0, "enzyme_to_reaction": 1}
    scope_order = {"current": 0, "external": 1}
    objective_order = {"top3": 0, "top10": 1, "top20": 2}
    routes.sort(key=lambda row: (
        direction_order[row["direction"]],
        scope_order[row["scope"]],
        objective_order[row["objective"]],
    ))

    manifest_route_ids = [row["route_id"] for row in routes]
    represented_suffixes = sorted({
        row["modifier_suffix"] for row in overlays if row.get("modifier_suffix")
    })
    expected_suffixes = [
        "fewshot", "manual", "masked", "temporary-universe",
        "eukaryote-only", "prokaryote-only",
    ]
    coverage = {
        "manifest_route_ids": manifest_route_ids,
        "manifest_route_count": len(manifest_route_ids),
        "missing_manifest_routes": [],
        "unexpected_manifest_routes": [],
        "runtime_modifier_suffixes": expected_suffixes,
        "represented_modifier_suffixes": represented_suffixes,
        "missing_runtime_modifiers": sorted(set(expected_suffixes) - set(represented_suffixes)),
        "conditional_paths": ["r2e-cage-rescue-overlay"],
    }
    coverage["complete"] = not coverage["missing_manifest_routes"] and not coverage["missing_runtime_modifiers"]

    taxonomy_scope = (
        json.loads(DEFAULT_TAXONOMY_SUMMARY.read_text(encoding="utf-8"))
        if DEFAULT_TAXONOMY_SUMMARY.is_file()
        else None
    )
    return {
        "manifest_version": payload["manifest_version"],
        "route_version": payload["route_version"],
        "candidate_universe_version": payload["candidate_universe_version"],
        "model_bundle_version": payload["model_bundle_version"],
        "routes": routes,
        "overlays": overlays,
        "route_count": len(routes),
        "overlay_count": len(overlays),
        "display_path_count": len(routes) + len(overlays),
        "coverage": coverage,
        "taxonomy_scope": taxonomy_scope,
        "read_only": True,
    }
