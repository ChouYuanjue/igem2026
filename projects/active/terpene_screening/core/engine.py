from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COMMON_FIELDS = {
    "top_k",
    "ranking_objective",
    "reliability_policy",
    "retrieval_mode",
    "hybrid_direct_weight",
    "topk_neighbor_reactions",
    "topk_neighbor_proteins",
    "query_id",
    "device",
    "reaction_feature_policy",
    "protein_input_policy",
}
COMMAND_FIELDS = {
    "rank-enzymes": {
        "reaction_id",
        "reaction_smiles",
        "known_enzyme_ids",
        "cage_rescue_slots",
    },
    "rank-reactions": {
        "enzyme_id",
        "enzyme_sequence",
        "known_reaction_ids",
        "mask_reaction_ids",
    },
}


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def payload_to_argv(command: str, payload: dict[str, Any], *, allow_overrides: bool = False) -> list[str]:
    if command not in COMMAND_FIELDS:
        raise ValueError(f"Unsupported retrieval command: {command}")
    allowed = COMMON_FIELDS | COMMAND_FIELDS[command]
    if allow_overrides:
        allowed |= {
            "model_dir",
            "dual_tower_dir",
            "protein_dir",
            "registered_protein_dir",
            "registered_reactions_csv",
            "dual_kernel_dir",
            "calibrators",
            "route_manifest",
            "feature_cache_dir",
            "positives",
            "external_enzymes_csv",
            "external_reactions_csv",
        }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported request fields: {unknown}")
    argv = [command]
    for key, value in payload.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
            continue
        argv.extend([flag, str(value)])
    return argv


@dataclass
class RetrievalEngine:
    allow_overrides: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    def rank_frame(self, command: str, payload: dict[str, Any]) -> pd.DataFrame:
        from projects.active.terpene_screening.rank_open_world import (
            build_parser,
            execute_ranking,
        )

        argv = payload_to_argv(command, payload, allow_overrides=self.allow_overrides)
        args = build_parser().parse_args(argv)
        with self._lock:
            return execute_ranking(args)

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self.rank_frame(command, payload)
        if frame.empty:
            return {"query": {}, "candidates": []}
        row = frame.iloc[0]
        query_columns = [
            "query_id",
            "direction",
            "ranking_objective",
            "route_id",
            "route_version",
            "candidate_universe_version",
            "candidate_universe_hash",
            "model_bundle_version",
            "registry_version",
            "score_source",
            "model_directory",
            "secondary_model_directory",
            "auxiliary_score_directory",
            "query_nearest_library_id",
            "query_nearest_library_similarity",
            "empirical_reliability_score",
            "empirical_reliability_tier",
            "empirical_reliability_status",
            "empirical_reliability_binding_status",
            "reliability_recommendation",
            "evidence_passport_version",
            "applicability_model_version",
            "query_applicability_score",
            "query_applicability_tier",
            "query_applicability_recommendation",
            "query_applicability_components",
            "query_applicability_interpretation",
        ]
        query = {
            column: _json_value(row[column])
            for column in query_columns
            if column in frame.columns
        }
        input_columns = [
            column
            for column in frame.columns
            if column.startswith(("protein_input_", "reaction_input_"))
        ]
        query["input_audit"] = {
            column: _json_value(row[column]) for column in input_columns
        }
        query["evidence_passport"] = {
            "version": query.pop("evidence_passport_version", None),
            "applicability_model_version": query.pop("applicability_model_version", None),
            "applicability_score": query.pop("query_applicability_score", None),
            "applicability_tier": query.pop("query_applicability_tier", None),
            "recommendation": query.pop("query_applicability_recommendation", None),
            "components": json.loads(query.pop("query_applicability_components", "{}") or "{}"),
            "interpretation": query.pop("query_applicability_interpretation", None),
        }
        candidate_exclude = set(query_columns) | set(input_columns)
        candidate_columns = [column for column in frame.columns if column not in candidate_exclude]
        candidates = []
        for record in frame[candidate_columns].to_dict("records"):
            candidate = {key: _json_value(value) for key, value in record.items()}
            candidate["evidence_passport"] = {
                "score": candidate.pop("candidate_evidence_score", None),
                "tier": candidate.pop("candidate_evidence_tier", None),
                "paths": [
                    value
                    for value in str(candidate.pop("candidate_evidence_paths", "") or "").split(";")
                    if value
                ],
                "warnings": [
                    value
                    for value in str(candidate.pop("candidate_evidence_warnings", "") or "").split(";")
                    if value
                ],
                "interpretation": candidate.pop("candidate_evidence_interpretation", None),
            }
            candidates.append(candidate)
        return {"query": query, "candidates": candidates}

    def dumps(self, command: str, payload: dict[str, Any]) -> str:
        return json.dumps(self.rank(command, payload), ensure_ascii=False)
