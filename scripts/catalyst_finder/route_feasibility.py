from __future__ import annotations

import json
import math
import subprocess
import sys
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scripts.catalyst_finder.route_design import RheaRouteDesigner


def _is_ecoli(host: str) -> bool:
    text = str(host or "").strip().casefold()
    return bool(text and ("escherichia coli" in text or "e. coli" in text or "e coli" in text or "大肠杆菌" in text or text == "ecoli"))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _percentile_components(values: dict[str, float]) -> dict[str, float]:
    """Return deterministic within-candidate [0,1] rank percentiles, higher is better."""
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda row: (-float(row[1]), row[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 1.0}
    denominator = float(len(ordered) - 1)
    return {route_id: 1.0 - rank / denominator for rank, (route_id, _value) in enumerate(ordered)}


class RouteFeasibilityAnalyzer:
    """Failure-tolerant orchestration for route thermodynamics and host FBA.

    Heavy scientific libraries stay in subprocess-only runtime sites. A failed/missing
    worker produces `unknown` evidence and never becomes evidence of feasibility.
    """

    def __init__(self, root: Path, route_designer: RheaRouteDesigner) -> None:
        self.root = Path(root)
        self.route_designer = route_designer
        self.runtime = self.root / "results/catalyst_finder_runtime/route_feasibility"
        self.thermo_worker = self.root / "scripts/catalyst_finder/route_thermo_worker.py"
        self.fba_worker = self.root / "scripts/catalyst_finder/route_fba_worker.py"
        self.thermo_site = self.runtime / "thermo_site"
        self.thermo_cache = self.runtime / "thermo_cache/equilibrator"
        self.fba_site = self.runtime / "fba_site"
        self.ecoli_model = self.runtime / "iML1515.json"

    def status(self) -> dict[str, Any]:
        thermo_ready = (
            self.thermo_worker.is_file()
            and (self.thermo_site / "equilibrator_api").is_dir()
            and self.thermo_cache.is_dir()
            and any(self.thermo_cache.iterdir())
        )
        fba_ready = (
            self.fba_worker.is_file()
            and (self.fba_site / "cobra").is_dir()
            and self.ecoli_model.is_file()
        )
        return {
            "thermodynamics": {
                "ready": thermo_ready,
                "engine": "equilibrator-api 0.7.0 + equilibrator-pathway 0.7.1",
            },
            "ecoli_fba": {
                "ready": fba_ready,
                "engine": "COBRApy 0.32.1",
                "model": "iML1515",
            },
        }

    def _run_worker(self, worker: Path, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        if not worker.is_file():
            return {"status": "unavailable", "message": f"worker missing: {worker.name}", "routes": []}
        try:
            completed = subprocess.run(
                [sys.executable, str(worker)],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                cwd=str(self.root),
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": f"{worker.name} exceeded {timeout}s", "routes": []}
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "worker failed")[-1600:]
            return {"status": "failed", "message": detail, "routes": []}
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {"status": "failed", "message": f"{worker.name} returned invalid JSON", "routes": []}
        return data if isinstance(data, dict) else {"status": "failed", "message": "worker result was not an object", "routes": []}

    @staticmethod
    def _weights(priority: str, *, host_expected: bool, thermo_available: bool, host_available: bool) -> dict[str, float]:
        if host_expected:
            if priority == "thermodynamic":
                weights = {"base": 0.30, "thermo": 0.50, "host": 0.20}
            elif priority == "host_flux":
                weights = {"base": 0.25, "thermo": 0.15, "host": 0.60}
            elif priority in {"short", "enzyme_available", "project_covered"}:
                weights = {"base": 0.55, "thermo": 0.15, "host": 0.30}
            else:
                weights = {"base": 0.50, "thermo": 0.20, "host": 0.30}
        else:
            if priority == "thermodynamic":
                weights = {"base": 0.35, "thermo": 0.65, "host": 0.0}
            else:
                weights = {"base": 0.75 if priority == "balanced" else 0.80, "thermo": 0.25 if priority == "balanced" else 0.20, "host": 0.0}
        # A globally unavailable layer must not silently lower every score. Route-level
        # unknown inside an available layer receives no bonus, preserving unknown != feasible.
        if not thermo_available:
            weights["base"] += weights["thermo"]
            weights["thermo"] = 0.0
        if not host_available:
            weights["base"] += weights["host"]
            weights["host"] = 0.0
        total = sum(weights.values()) or 1.0
        return {key: value / total for key, value in weights.items()}

    def evaluate(
        self,
        routes: list[dict[str, Any]],
        *,
        host: str = "",
        priority: str = "balanced",
        requested_count: int = 10,
        run_thermodynamics: bool = True,
        run_host_flux: bool = True,
    ) -> dict[str, Any]:
        if not routes:
            return {
                "routes": [],
                "summary": {
                    "preliminary_route_count": 0,
                    "returned_route_count": 0,
                    "thermo_complete_count": 0,
                    "host_fba_complete_count": 0,
                    "host_infeasible_filtered_count": 0,
                },
                "thermo_run": {"status": "not_run"},
                "host_run": {"status": "not_run"},
            }

        need_stoichiometry = bool(run_thermodynamics or run_host_flux)
        enriched = [
            self.route_designer.enrich_route_stoichiometry(route) if need_stoichiometry else deepcopy(route)
            for route in routes
        ]
        host_expected = _is_ecoli(host) and bool(run_host_flux)
        state = self.status() if need_stoichiometry else {"thermodynamics": {"ready": False}, "ecoli_fba": {"ready": False}}
        thermo_ready = bool(run_thermodynamics and state["thermodynamics"]["ready"])
        fba_ready = bool(run_host_flux and state["ecoli_fba"]["ready"] and host_expected)

        def thermo_call() -> dict[str, Any]:
            if not run_thermodynamics:
                return {"status": "not_requested", "routes": []}
            if not thermo_ready:
                return {"status": "unavailable", "message": "eQuilibrator runtime/cache not ready", "routes": []}
            return self._run_worker(self.thermo_worker, {"routes": enriched}, timeout=110)

        def fba_call() -> dict[str, Any]:
            if not run_host_flux:
                return {"status": "not_requested", "routes": []}
            if not host_expected:
                return {"status": "not_applicable", "routes": []}
            if not fba_ready:
                return {"status": "unavailable", "message": "COBRApy/iML1515 runtime not ready", "routes": []}
            return self._run_worker(
                self.fba_worker,
                {"routes": enriched, "growth_fractions": [0.1, 0.5]},
                timeout=110,
            )

        # Run only the analyses requested for this turn.
        if run_thermodynamics and run_host_flux:
            with ThreadPoolExecutor(max_workers=2) as pool:
                thermo_future = pool.submit(thermo_call)
                fba_future = pool.submit(fba_call)
                thermo_run = thermo_future.result()
                host_run = fba_future.result()
        else:
            thermo_run = thermo_call()
            host_run = fba_call()

        thermo_by_id = {str(row.get("route_id")): row for row in thermo_run.get("routes") or []}
        fba_by_id = {str(row.get("route_id")): row for row in host_run.get("routes") or []}
        thermo_values: dict[str, float] = {}
        fba_values: dict[str, float] = {}
        for route in enriched:
            route_id = str(route.get("route_id") or "")
            thermo = thermo_by_id.get(route_id) or {"status": "unknown"}
            host_result = fba_by_id.get(route_id) or ({"status": "unknown"} if host_expected else {"status": "not_applicable"})
            route["thermodynamics"] = thermo
            route["host_feasibility"] = host_result
            if thermo.get("status") == "complete" and thermo.get("mdf_kj_mol") is not None:
                thermo_values[route_id] = float(thermo["mdf_kj_mol"])
            if host_result.get("status") == "complete" and host_result.get("max_route_flux_50pct_growth") is not None:
                fba_values[route_id] = float(host_result["max_route_flux_50pct_growth"])

        thermo_component = _percentile_components(thermo_values)
        fba_component = _percentile_components({rid: value for rid, value in fba_values.items() if value > 1e-10})
        thermo_available = bool(thermo_values)
        host_available = bool(fba_values) if host_expected else False
        weights = self._weights(
            priority,
            host_expected=host_expected,
            thermo_available=thermo_available,
            host_available=host_available,
        )

        eligible: list[dict[str, Any]] = []
        filtered_host_infeasible = 0
        thermo_negative_count = 0
        for route in enriched:
            route_id = str(route.get("route_id") or "")
            base_score = float(route.get("base_route_score", route.get("score", 0.0)) or 0.0)
            thermo = route["thermodynamics"]
            host_result = route["host_feasibility"]
            mdf = thermo_values.get(route_id)
            t_component = thermo_component.get(route_id, 0.0)
            if mdf is not None and mdf < 0.0:
                # Negative MDF means no favorable assignment was found within the stated
                # concentration bounds. Keep it auditable but strongly demote it.
                t_component *= 0.10
                thermo_negative_count += 1
            h_component = fba_component.get(route_id, 0.0)
            host_complete = host_result.get("status") == "complete"
            host_flux = fba_values.get(route_id)
            if host_expected and host_complete and (host_flux is None or host_flux <= 1e-10):
                route["host_feasibility_gate"] = "infeasible"
                filtered_host_infeasible += 1
                continue
            route["host_feasibility_gate"] = "pass" if host_expected and host_complete else "unknown" if host_expected else "not_applicable"
            final_fraction = (
                weights["base"] * _clip01(base_score / 100.0)
                + weights["thermo"] * _clip01(t_component)
                + weights["host"] * _clip01(h_component)
            )
            route["base_route_score"] = round(base_score, 2)
            route["final_score"] = round(100.0 * final_fraction, 2)
            route["score"] = route["final_score"]
            route["ranking_components"] = {
                "base": round(_clip01(base_score / 100.0), 4),
                "thermodynamics": round(_clip01(t_component), 4) if route_id in thermo_values else None,
                "host_flux": round(_clip01(h_component), 4) if route_id in fba_values else None,
                "weights": {key: round(value, 4) for key, value in weights.items()},
                "thermodynamics_value": "within-candidate MDF percentile; negative MDF strongly demoted",
                "host_flux_value": "within-candidate percentile of route-supported flux at >=50% wild-type growth",
            }
            eligible.append(route)

        # Thermodynamic sign is a first-class scientific distinction. Within the host
        # feasible set, routes with complete negative MDF are placed after nonnegative/
        # unknown routes even if their legacy base score was high.
        def sort_key(route: dict[str, Any]):
            thermo = route.get("thermodynamics") or {}
            mdf = thermo.get("mdf_kj_mol") if thermo.get("status") == "complete" else None
            thermo_tier = 1 if mdf is not None and float(mdf) < 0.0 else 0
            return (thermo_tier, -float(route.get("final_score") or 0.0), int((route.get("metrics") or {}).get("step_count") or 999), str(route.get("route_id") or ""))

        eligible.sort(key=sort_key)
        requested_count = max(1, min(int(requested_count or 10), 20))
        returned = eligible[:requested_count]
        for rank, route in enumerate(returned, start=1):
            route["rank"] = rank

        # Keep detailed stoichiometry server-internal. Frontend receives coverage status,
        # real thermo/FBA evidence, and Rhea IDs without duplicating every participant.
        for route in returned:
            for step in route.get("steps") or []:
                full = step.pop("full_stoichiometry", None) or {}
                step["full_stoichiometry_status"] = full.get("status") or "unknown"

        return {
            "routes": returned,
            "summary": {
                "preliminary_route_count": len(enriched),
                "eligible_route_count": len(eligible),
                "returned_route_count": len(returned),
                "thermo_complete_count": len(thermo_values),
                "thermo_negative_count": thermo_negative_count,
                "host_fba_complete_count": len(fba_values),
                "host_infeasible_filtered_count": filtered_host_infeasible,
                "host_expected": host_expected,
                "requested_layers": [
                    *(["thermodynamics"] if run_thermodynamics else []),
                    *(["host_flux"] if run_host_flux else []),
                ],
                "ranking_weights": {key: round(value, 4) for key, value in weights.items()},
            },
            "thermo_run": {
                "status": thermo_run.get("status"),
                "engine": thermo_run.get("engine"),
                "conditions": thermo_run.get("conditions"),
                "message": thermo_run.get("message"),
            },
            "host_run": {
                "status": host_run.get("status"),
                "engine": host_run.get("engine"),
                "model": host_run.get("model"),
                "baseline_growth": host_run.get("baseline_growth"),
                "message": host_run.get("message"),
            },
        }
