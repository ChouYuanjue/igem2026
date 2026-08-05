from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable
import pandas as pd
from .routing import RouteProvenance

def identifier_set_hash(values: Iterable[str]) -> str:
    payload = "\n".join(sorted({str(value) for value in values}))
    return hashlib.sha256(payload.encode()).hexdigest()

def apply_route_provenance(result: pd.DataFrame, route: RouteProvenance, *,
                           candidate_ids: Iterable[str], registry_version: str = "legacy") -> pd.DataFrame:
    result["route_id"] = route.route_id
    result["route_version"] = route.route_version
    result["candidate_universe_version"] = route.candidate_universe_version
    result["candidate_universe_hash"] = identifier_set_hash(candidate_ids)
    result["model_bundle_version"] = route.model_bundle_version
    result["registry_version"] = registry_version
    return result

def write_query_audit(result: pd.DataFrame, output: Path) -> Path:
    if result.empty: raise ValueError("Cannot write a query audit for an empty result")
    row = result.iloc[0]
    keys = ["query_id","direction","ranking_objective","route_id","route_version",
            "candidate_universe_version","candidate_universe_hash","model_bundle_version",
            "registry_version","score_source","model_directory","secondary_model_directory",
            "auxiliary_score_directory","empirical_reliability_score",
            "empirical_reliability_tier","empirical_reliability_status"]
    query = {key: row.get(key) for key in keys if key in result.columns}
    input_columns = [c for c in result.columns if c.startswith(("protein_input_", "reaction_input_"))]
    query["input_audit"] = {c: row.get(c) for c in input_columns}
    query["n_results"] = len(result)
    cols = [c for c in ["rank","candidate_id","score","selection_source","is_external_candidate"] if c in result]
    query["candidates"] = result[cols].to_dict("records")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(query, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output
