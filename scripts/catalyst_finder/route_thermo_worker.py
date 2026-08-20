#!/usr/bin/env python3
"""Isolated eQuilibrator worker for route-level thermodynamics.

Input reactions must already be verified Rhea directed reactions with complete participant
stoichiometry. The worker never guesses a compound ID. It resolves exact Rhea/ChEBI
participants into eQuilibrator, reports per-step transformed Gibbs energies, and uses
`equilibrator-pathway` MDF constraints for whole-route Max-min Driving Force.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "results/catalyst_finder_runtime/route_feasibility/thermo_site"
CACHE = ROOT / "results/catalyst_finder_runtime/route_feasibility/thermo_cache"
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
sys.path.insert(0, str(SITE))

import cvxpy as cp  # type: ignore  # noqa: E402
import numpy as np  # type: ignore  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
from equilibrator_api import ComponentContribution, Q_, R, default_T  # type: ignore  # noqa: E402
from equilibrator_api.phased_reaction import PhasedReaction  # type: ignore  # noqa: E402
from equilibrator_pathway import ThermodynamicModel  # type: ignore  # noqa: E402


def _measurement(value) -> dict[str, float | None]:
    try:
        magnitude = value.to("kJ/mol").magnitude
    except Exception:
        magnitude = getattr(value, "magnitude", value)
    nominal = getattr(magnitude, "nominal_value", magnitude)
    std = getattr(magnitude, "std_dev", None)
    try:
        nominal_f = float(nominal)
    except Exception:
        nominal_f = None
    try:
        std_f = float(std) if std is not None else None
    except Exception:
        std_f = None
    if nominal_f is not None and not math.isfinite(nominal_f):
        nominal_f = None
    if std_f is not None and not math.isfinite(std_f):
        std_f = None
    return {"value_kj_mol": nominal_f, "uncertainty_kj_mol": std_f}


def _resolve_compound(cc: ComponentContribution, candidates: list[str]):
    errors = []
    for cid in candidates:
        accession = "chebi:" + str(cid).upper()
        try:
            compound = cc.get_compound(accession)
        except Exception as exc:
            errors.append(f"{accession}: {type(exc).__name__}")
            continue
        if compound is not None:
            return compound, str(cid).upper(), errors
    return None, None, errors


def _build_route_model(cc: ComponentContribution, route: dict[str, Any]):
    reaction_dict = {}
    compound_dict = {}
    reaction_rows = []
    failures = []

    for step in route.get("steps") or []:
        full = step.get("full_stoichiometry") or {}
        step_index = int(step.get("step_index") or len(reaction_rows) + 1)
        rid = str(step.get("directed_rhea_id") or step.get("rhea_id") or f"step-{step_index}")
        if full.get("status") != "complete":
            failures.append({"step_index": step_index, "rhea_id": rid, "reason": "stoichiometry_incomplete"})
            continue
        sparse = {}
        selected = []
        unresolved = []
        for participant in full.get("participants") or []:
            candidates = [str(x).upper() for x in participant.get("chebi_candidates") or [] if str(x).strip()]
            compound, selected_chebi, errors = _resolve_compound(cc, candidates)
            if compound is None:
                unresolved.append({"candidates": candidates, "smiles": participant.get("smiles"), "errors": errors[-3:]})
                continue
            coeff = float(participant.get("coefficient") or 0.0)
            sparse[compound] = sparse.get(compound, 0.0) + coeff
            key = f"eq:{compound.id}"
            compound_dict[key] = compound
            selected.append({"chebi_id": selected_chebi, "coefficient": coeff, "equilibrator_compound_id": compound.id})
        if unresolved or not sparse:
            failures.append({"step_index": step_index, "rhea_id": rid, "reason": "compound_unresolved", "unresolved": unresolved})
            continue
        reaction = PhasedReaction(sparse, arrow="->", rid=rid)
        balanced = bool(reaction.is_balanced())
        step_result = {
            "step_index": step_index,
            "rhea_id": str(step.get("rhea_id") or ""),
            "directed_rhea_id": rid,
            "balanced": balanced,
            "selected_participants": selected,
        }
        if not balanced:
            failures.append({"step_index": step_index, "rhea_id": rid, "reason": "reaction_unbalanced"})
        else:
            try:
                step_result["standard_dg_prime"] = _measurement(cc.standard_dg_prime(reaction))
                step_result["physiological_dg_prime"] = _measurement(cc.physiological_dg_prime(reaction))
            except Exception as exc:
                step_result["dg_error"] = f"{type(exc).__name__}: {exc}"
                failures.append({"step_index": step_index, "rhea_id": rid, "reason": "dg_failed"})
        reaction_dict[rid] = reaction
        reaction_rows.append(step_result)

    if failures or len(reaction_dict) != len(route.get("steps") or []):
        return None, reaction_rows, failures

    columns = list(reaction_dict)
    S = pd.DataFrame(0.0, index=list(compound_dict), columns=columns)
    for rid, reaction in reaction_dict.items():
        for compound, coeff in reaction.items():
            S.loc[f"eq:{compound.id}", rid] = float(coeff)
    model = ThermodynamicModel(
        S=S,
        compound_dict=compound_dict,
        reaction_dict=reaction_dict,
        fluxes=np.ones(len(columns)) * Q_(1),
        comp_contrib=cc,
        config_dict={"solver": "CLARABEL"},
    )
    return model, reaction_rows, failures


def _mdf_with_single_reaction_fallback(model: ThermodynamicModel) -> tuple[float, list[float], str]:
    """Run package MDF; bypass only the 0-D result-object bug in v0.7.1.

    Current equilibrator-pathway 0.7.1 can optimize a single-reaction MDF but raises
    while constructing PathwayMdfSolution because a scalar physiological dG is indexed
    as a vector. The fallback uses the package's own concentration and thermodynamic
    constraint builders and solver, returning the already-defined MDF objective only.
    """
    try:
        solution = model.mdf_analysis()
        driving = []
        for row in solution.reaction_df.itertuples(index=False):
            optimized = getattr(row, "optimized_dg_prime", None)
            m = _measurement(optimized) if optimized is not None else {"value_kj_mol": None}
            value = m.get("value_kj_mol")
            driving.append(float(-value) if value is not None else float("nan"))
        return float(solution.score), driving, "equilibrator_pathway.mdf_analysis"
    except IndexError:
        if model.Nr != 1:
            raise

    ln_conc = cp.Variable(shape=model.Nc, name="log concentrations")
    c_lbs, c_ubs = model._conc_constraints(ln_conc)
    B = cp.Variable(shape=1, name="minimum driving force")
    y, y_constraints, dg_constraints = model._thermo_constraints(ln_conc, B)
    problem = cp.Problem(cp.Maximize(B), y_constraints + dg_constraints + c_lbs + c_ubs)
    problem.solve(model._solver)
    if problem.status != "optimal":
        raise RuntimeError(f"MDF solver status: {problem.status}")

    # Reconstruct the optimized dG using the same formula as the package constraint
    # builder so the reported driving force corresponds exactly to the solved MDF.
    rt = (R * default_T).m_as("kJ/mol")
    dg = model.standard_dg_primes.m_as("kJ/mol") + rt * model.S.T.values @ ln_conc.value
    if model.dg_sigma is not None and y is not None and y.value is not None:
        dg = dg + model.dg_sigma.m_as("kJ/mol") @ y.value
    driving = [-float(x) for x in np.asarray(dg).reshape(-1)]
    return float(problem.value), driving, "equilibrator_pathway.constraints_fallback"


def _analyze_route(cc: ComponentContribution, route: dict[str, Any]) -> dict[str, Any]:
    result = {
        "route_id": route.get("route_id"),
        "engine": "eQuilibrator / equilibrator-pathway",
        "status": "unknown",
        "steps": [],
    }
    try:
        model, steps, failures = _build_route_model(cc, route)
        result["steps"] = steps
        if model is None:
            result.update({"status": "insufficient_evidence", "failures": failures})
            return result
        mdf, driving, method = _mdf_with_single_reaction_fallback(model)
        for row, force in zip(result["steps"], driving):
            if math.isfinite(force):
                row["mdf_optimized_driving_force_kj_mol"] = force
        result.update({
            "status": "complete",
            "mdf_kj_mol": mdf,
            "mdf_positive": mdf > 0.0,
            "method": method,
            "reaction_count": model.Nr,
            "compound_count": model.Nc,
        })
        return result
    except Exception as exc:
        result.update({"status": "calculation_failed", "error": f"{type(exc).__name__}: {exc}"})
        return result


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    cc = ComponentContribution()
    conditions = {
        "p_h": float(cc.p_h.magnitude),
        "p_mg": float(cc.p_mg.magnitude),
        "ionic_strength_m": float(cc.ionic_strength.to("M").magnitude),
        "temperature_k": float(cc.temperature.to("K").magnitude),
        "temperature_c": float(cc.temperature.to("degC").magnitude),
        "concentration_bounds": "eQuilibrator default bounds; typically 1 µM–10 mM with curated cofactor exceptions",
    }
    routes = [_analyze_route(cc, route) for route in payload.get("routes") or []]
    return {
        "status": "complete",
        "engine": "equilibrator-api 0.7.0 + equilibrator-pathway 0.7.1",
        "conditions": conditions,
        "routes": routes,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    sys.stdout.write(json.dumps(analyze(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
