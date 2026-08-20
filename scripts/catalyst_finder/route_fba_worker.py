#!/usr/bin/env python3
"""Isolated COBRApy worker for E. coli route-feasibility analysis.

The worker evaluates *route-supported* flux: every candidate pathway reaction and a
route-specific target demand must carry at least a common positive flux while the
host maintains a requested fraction of wild-type growth. This prevents a native
bypass from making an unused candidate route look feasible.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results/catalyst_finder_runtime/route_feasibility/fba_site"
MODEL_PATH = ROOT / "results/catalyst_finder_runtime/route_feasibility/iML1515.json"
sys.path.insert(0, str(SITE))

from cobra import Metabolite, Reaction  # type: ignore  # noqa: E402
from cobra.io import load_json_model  # type: ignore  # noqa: E402


def _chebi_values(annotation: dict[str, Any]) -> list[str]:
    raw = annotation.get("chebi") if isinstance(annotation, dict) else None
    if raw is None:
        return []
    values = raw if isinstance(raw, list) else [raw]
    out = []
    for value in values:
        text = str(value or "").strip().upper()
        if not text:
            continue
        if not text.startswith("CHEBI:"):
            text = "CHEBI:" + text.replace("CHEBI", "").lstrip(":")
        out.append(text)
    return list(dict.fromkeys(out))


def _native_chebi_map(model) -> dict[str, list[Any]]:
    mapping: dict[str, list[Any]] = defaultdict(list)
    for metabolite in model.metabolites:
        if str(getattr(metabolite, "compartment", "")) != "c":
            continue
        for cid in _chebi_values(getattr(metabolite, "annotation", {}) or {}):
            mapping[cid].append(metabolite)
    return dict(mapping)


def _select_native_metabolite(candidates: list[str], mapping: dict[str, list[Any]]):
    for cid in candidates:
        rows = mapping.get(str(cid).upper()) or []
        if rows:
            # Stable preference if an annotation maps to more than one cytosolic entry.
            return sorted(rows, key=lambda m: m.id)[0], str(cid).upper()
    return None, None


def _route_model(base_model, route: dict[str, Any], native_map: dict[str, list[Any]]):
    model = base_model.copy()
    # Objects from base_model are not the same objects after copy; rebuild lookup by ID.
    native_by_chebi: dict[str, list[Any]] = defaultdict(list)
    for cid, rows in native_map.items():
        for row in rows:
            try:
                native_by_chebi[cid].append(model.metabolites.get_by_id(row.id))
            except KeyError:
                pass

    created: dict[str, Any] = {}
    chosen_by_smiles: dict[str, Any] = {}
    native_participants: set[str] = set()
    all_participants: set[str] = set()

    def resolve_participant(participant: dict[str, Any]):
        smiles = str(participant.get("smiles") or "")
        candidates = [str(x).upper() for x in participant.get("chebi_candidates") or [] if str(x).strip()]
        if smiles in chosen_by_smiles:
            return chosen_by_smiles[smiles]
        native, matched = _select_native_metabolite(candidates, native_by_chebi)
        all_participants.update(candidates)
        if native is not None:
            if matched:
                native_participants.add(matched)
            chosen_by_smiles[smiles] = native
            return native
        cid = candidates[0] if candidates else "CHEBI:UNMAPPED"
        key = cid
        if key in created:
            metabolite = created[key]
        else:
            safe = cid.lower().replace(":", "_").replace("-", "_")
            metabolite = Metabolite(
                id=f"cf_{safe}_c",
                name=cid,
                compartment="c",
            )
            if candidates:
                metabolite.annotation = {"chebi": candidates}
            model.add_metabolites([metabolite])
            created[key] = metabolite
        chosen_by_smiles[smiles] = metabolite
        return metabolite

    route_reactions = []
    incomplete_steps = []
    for step in route.get("steps", []):
        full = step.get("full_stoichiometry") or {}
        if full.get("status") != "complete":
            incomplete_steps.append(int(step.get("step_index") or len(route_reactions) + 1))
            continue
        stoich = {}
        for participant in full.get("participants") or []:
            metabolite = resolve_participant(participant)
            coeff = float(participant.get("coefficient") or 0.0)
            if abs(coeff) > 1e-12:
                stoich[metabolite] = stoich.get(metabolite, 0.0) + coeff
        if not stoich:
            incomplete_steps.append(int(step.get("step_index") or len(route_reactions) + 1))
            continue
        rid = f"CF_{route.get('route_id','route')}_{int(step.get('step_index') or len(route_reactions)+1)}"
        reaction = Reaction(rid)
        reaction.name = str(step.get("directed_rhea_id") or step.get("rhea_id") or rid)
        reaction.lower_bound = 0.0
        reaction.upper_bound = 1000.0
        reaction.add_metabolites(stoich)
        reaction.annotation = {
            "rhea": [x for x in [step.get("rhea_id"), step.get("directed_rhea_id")] if x]
        }
        model.add_reactions([reaction])
        route_reactions.append(reaction)

    if incomplete_steps or not route_reactions:
        return model, route_reactions, None, {
            "status": "stoichiometry_incomplete",
            "incomplete_steps": incomplete_steps,
            "native_participant_count": len(native_participants),
            "participant_count": len(all_participants),
            "created_metabolite_count": len(created),
        }

    source_id = str((route.get("compound_ids") or [""])[0])
    target_id = str((route.get("compound_ids") or [""])[-1])
    source_rows = native_by_chebi.get(source_id.upper()) or []
    target_rows = native_by_chebi.get(target_id.upper()) or []

    # Find the target metabolite actually used in the final full Rhea reaction. Prefer
    # a native ChEBI match; otherwise use the route-created metabolite with the target ID.
    target_met = sorted(target_rows, key=lambda m: m.id)[0] if target_rows else created.get(target_id.upper())
    if target_met is None:
        # Target IDs can be an alternate ChEBI form. Search all final-step participants.
        final_full = (route.get("steps") or [])[-1].get("full_stoichiometry") or {}
        for participant in final_full.get("participants") or []:
            if float(participant.get("coefficient") or 0.0) <= 0:
                continue
            candidates = [str(x).upper() for x in participant.get("chebi_candidates") or []]
            if target_id.upper() not in candidates:
                continue
            target_met = resolve_participant(participant)
            break
    if target_met is None:
        return model, route_reactions, None, {
            "status": "target_unmapped",
            "source_native": bool(source_rows),
            "target_native": bool(target_rows),
            "native_participant_count": len(native_participants),
            "participant_count": len(all_participants),
            "created_metabolite_count": len(created),
        }

    demand = Reaction(f"DM_CF_{route.get('route_id','route')}")
    demand.name = f"Catalyst Finder demand for {target_id}"
    demand.lower_bound = 0.0
    demand.upper_bound = 1000.0
    demand.add_metabolites({target_met: -1.0})
    model.add_reactions([demand])
    meta = {
        "status": "prepared",
        "source_native": bool(source_rows),
        "target_native": bool(target_rows),
        "native_participant_count": len(native_participants),
        "participant_count": len(all_participants),
        "created_metabolite_count": len(created),
        "route_reaction_count": len(route_reactions),
        "target_metabolite_id": target_met.id,
    }
    return model, route_reactions, demand, meta


def _single_route(base_model, route: dict[str, Any], native_map: dict[str, list[Any]], baseline_growth: float, growth_fractions: list[float]) -> dict[str, Any]:
    model, route_reactions, demand, meta = _route_model(base_model, route, native_map)
    result = {
        "route_id": route.get("route_id"),
        "engine": "COBRApy/iML1515",
        **meta,
        "baseline_growth": baseline_growth,
        "growth_fraction_results": [],
    }
    if meta.get("status") != "prepared" or demand is None:
        return result

    objective_reactions = [rxn for rxn in model.reactions if abs(float(rxn.objective_coefficient)) > 1e-12]
    if len(objective_reactions) != 1:
        result.update({"status": "unsupported_objective", "objective_reaction_count": len(objective_reactions)})
        return result
    biomass = objective_reactions[0]

    z = model.problem.Variable(f"z_{str(route.get('route_id') or 'route').replace('-', '_')}", lb=0.0)
    constraints = []
    for reaction in route_reactions:
        constraints.append(model.problem.Constraint(reaction.flux_expression - z, lb=0.0, name=f"minroute_{reaction.id}"))
    constraints.append(model.problem.Constraint(demand.flux_expression - z, lb=0.0, name=f"mintarget_{demand.id}"))
    model.add_cons_vars([z, *constraints])
    model.objective = model.problem.Objective(z, direction="max")

    capacities = []
    for fraction in sorted({max(0.0, min(1.0, float(x))) for x in growth_fractions}):
        biomass.lower_bound = max(float(biomass.lower_bound), baseline_growth * fraction)
        try:
            solution = model.optimize()
            status = str(solution.status)
            capacity = float(solution.objective_value or 0.0) if status == "optimal" else 0.0
        except Exception as exc:  # solver/runtime failure is evidence-unavailable, not zero biology
            status = "solver_error"
            capacity = 0.0
            result.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
        capacities.append(capacity)
        result["growth_fraction_results"].append({
            "minimum_growth_fraction": fraction,
            "minimum_growth": baseline_growth * fraction,
            "status": status,
            "route_flux_capacity": capacity,
        })
    result["status"] = "complete"
    result["max_route_flux_10pct_growth"] = next((r["route_flux_capacity"] for r in result["growth_fraction_results"] if abs(r["minimum_growth_fraction"] - 0.1) < 1e-9), None)
    result["max_route_flux_50pct_growth"] = next((r["route_flux_capacity"] for r in result["growth_fraction_results"] if abs(r["minimum_growth_fraction"] - 0.5) < 1e-9), None)
    result["host_feasible"] = bool(capacities and max(capacities) > 1e-8)
    return result


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(str(payload.get("model_path") or MODEL_PATH))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.exists():
        return {"status": "model_missing", "model_path": str(model_path), "routes": []}
    base_model = load_json_model(str(model_path))
    base_model.solver = "glpk"
    baseline_solution = base_model.optimize()
    if str(baseline_solution.status) != "optimal":
        return {"status": "baseline_infeasible", "routes": []}
    baseline_growth = float(baseline_solution.objective_value or 0.0)
    native_map = _native_chebi_map(base_model)
    growth_fractions = payload.get("growth_fractions") or [0.1, 0.5]
    rows = [
        _single_route(base_model, route, native_map, baseline_growth, list(growth_fractions))
        for route in payload.get("routes") or []
    ]
    return {
        "status": "complete",
        "engine": "COBRApy 0.32.1",
        "model": "iML1515",
        "baseline_growth": baseline_growth,
        "cytosolic_chebi_count": len(native_map),
        "routes": rows,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    sys.stdout.write(json.dumps(analyze(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
