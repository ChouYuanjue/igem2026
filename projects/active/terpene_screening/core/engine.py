from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from projects.active.terpene_screening.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    TPS_SPECIALIZED_UNIVERSE,
    resolve_candidate_universe,
)

ROOT = Path(__file__).resolve().parents[4]


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
    "conformal_mode",
    "conformal_alpha",
    "candidate_universe",
    "candidate_ids",
    "mask_semantics",
}
COMMAND_FIELDS = {
    "rank-enzymes": {
        "reaction_id",
        "reaction_smiles",
        "known_enzyme_ids",
        "mask_enzyme_ids",
        "cage_rescue_slots",
        "enzyme_taxonomy_scope",
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
    payload = dict(payload)
    # Direct research/core callers retain the historical TPS universe unless they
    # opt in. The Catalyst product layer always supplies its product-level default.
    # Request serialization itself is deliberately asset-independent; strict
    # candidate-universe validation happens immediately before actual execution in
    # RetrievalEngine.rank_frame(). This keeps portable CI/parser tests meaningful
    # without weakening the production execution boundary.
    universe_key = str(payload.pop("candidate_universe", TPS_SPECIALIZED_UNIVERSE))
    universe = resolve_candidate_universe(ROOT, universe_key, validate=False)
    allowed = COMMON_FIELDS | COMMAND_FIELDS[command]
    if allow_overrides:
        allowed |= {
            "model_dir",
            "dual_tower_dir",
            "internal_expert_override",
            "protein_dir",
            "registered_protein_dir",
            "registered_reactions_csv",
            "registered_reaction_feature_dir",
            "dual_kernel_dir",
            "calibrators",
            "conformal_calibrators",
            "route_manifest",
            "feature_cache_dir",
            "positives",
            "external_enzymes_csv",
            "external_reactions_csv",
        }
        if command == "rank-enzymes":
            allowed.add("taxonomy_scope_registry")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unsupported request fields: {unknown}")
    argv = [command]
    if universe.key != TPS_SPECIALIZED_UNIVERSE:
        argv.extend(["--protein-dir", str(universe.protein_dir)])
        # The merged universe already contains the historical registered protein
        # library. Point the optional extension hook back at the same directory so
        # rank_open_world does not append stale duplicates from the old registry.
        argv.extend(["--registered-protein-dir", str(universe.protein_dir)])
        argv.extend(["--registered-reactions-csv", str(universe.registered_reactions_csv)])
        if command == "rank-reactions" and universe.reaction_feature_dir is not None:
            argv.extend(["--registered-reaction-feature-dir", str(universe.reaction_feature_dir)])
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

    def prewarm_protein_encoder(self) -> dict[str, str]:
        """Load the fixed production ESM-C encoder before a raw-sequence ranking.

        This intentionally exposes no user-selectable model name, path, or device.
        It is a latency optimization for the same encoder already used by
        ``rank-reactions --enzyme-sequence``.
        """
        from projects.active.terpene_screening.rank_open_world import prewarm_esmc_model

        return prewarm_esmc_model()

    def rank_frame(self, command: str, payload: dict[str, Any]) -> pd.DataFrame:
        from projects.active.terpene_screening.rank_open_world import (
            build_parser,
            execute_ranking,
        )

        # Keep strict provenance/runtime checks at the execution boundary. Missing
        # TPS-specialist assets may not poison general request parsing, but selecting
        # an incomplete universe for an actual ranking must still fail before model IO.
        resolve_candidate_universe(
            ROOT,
            str(payload.get("candidate_universe", TPS_SPECIALIZED_UNIVERSE)),
            validate=True,
        )
        argv = payload_to_argv(command, payload, allow_overrides=self.allow_overrides)
        args = build_parser().parse_args(argv)
        with self._lock:
            return execute_ranking(args)

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        # rank_frame performs the strict execution-time validation. This non-validating
        # resolution is only for stable metadata annotation and also keeps mocked unit
        # tests independent from server-only candidate-universe assets.
        universe = resolve_candidate_universe(
            ROOT,
            str(payload.get("candidate_universe", TPS_SPECIALIZED_UNIVERSE)),
            validate=False,
        )
        frame = self.rank_frame(command, payload)
        if not frame.empty:
            frame = frame.copy()
            frame["candidate_universe_version"] = universe.version
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
            "candidate_universe_size",
            "model_bundle_version",
            "registry_version",
            "score_source",
            "model_directory",
            "model_feature_directory",
            "secondary_model_directory",
            "auxiliary_score_directory",
            "query_nearest_library_id",
            "query_nearest_library_similarity",
            "query_is_current_entity",
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
            "conformal_retrieval_version",
            "conformal_method",
            "conformal_mode",
            "conformal_alpha",
            "conformal_target_coverage",
            "conformal_calibrator",
            "conformal_binding_status",
            "conformal_status",
            "conformal_group",
            "conformal_group_source",
            "conformal_qhat",
            "conformal_set_size",
            "conformal_set_fraction",
            "conformal_set_truncated",
            "conformal_validation_coverage",
            "conformal_validation_n",
            "conformal_guarantee_scope",
            "conformal_interpretation",
            "conformal_recommendation",
            "requested_top_k",
            "conformal_expanded_output",
            "taxonomy_scope_version",
            "enzyme_taxonomy_scope",
            "taxonomy_scope_mode",
            "candidate_universe_pre_taxonomy_size",
            "candidate_universe_post_taxonomy_size",
            "candidate_subset_applied",
            "candidate_subset_requested_count",
            "candidate_subset_effective_count",
            "candidate_subset_missing_count",
            "taxonomy_eukaryote_count",
            "taxonomy_prokaryote_count",
            "taxonomy_other_count",
            "taxonomy_unknown_count",
            "taxonomy_excluded_count",
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
        query["conformal_retrieval_set"] = {
            "version": query.pop("conformal_retrieval_version", None),
            "method": query.pop("conformal_method", None),
            "mode": query.pop("conformal_mode", None),
            "alpha": query.pop("conformal_alpha", None),
            "target_coverage": query.pop("conformal_target_coverage", None),
            "calibrator": query.pop("conformal_calibrator", None),
            "binding_status": query.pop("conformal_binding_status", None),
            "status": query.pop("conformal_status", None),
            "applicability_group": query.pop("conformal_group", None),
            "group_source": query.pop("conformal_group_source", None),
            "qhat": query.pop("conformal_qhat", None),
            "set_size": query.pop("conformal_set_size", None),
            "set_fraction": query.pop("conformal_set_fraction", None),
            "truncated": query.pop("conformal_set_truncated", None),
            "validation_coverage": query.pop("conformal_validation_coverage", None),
            "validation_n": query.pop("conformal_validation_n", None),
            "guarantee_scope": query.pop("conformal_guarantee_scope", None),
            "interpretation": query.pop("conformal_interpretation", None),
            "recommendation": query.pop("conformal_recommendation", None),
            "requested_top_k": query.pop("requested_top_k", None),
            "expanded_output": query.pop("conformal_expanded_output", None),
        }
        seed_key = "known_enzyme_ids" if command == "rank-enzymes" else "known_reaction_ids"
        requested_seed_ids = [str(value) for value in payload.get(seed_key, []) or []]
        masked_ids = (
            [str(value) for value in payload.get("mask_reaction_ids", []) or []]
            if command == "rank-reactions"
            else []
        )
        effective_few_shot = (
            str(query.get("score_source", "")) == "seed"
            or "+fewshot" in str(query.get("route_id", ""))
        )
        query["scope"] = "current" if query.get("query_is_current_entity") else "external"
        query["candidate_universe"] = universe.key
        query["candidate_universe_description"] = universe.description
        if not universe.specialized:
            # Existing empirical/conformal calibrators were fitted on the historical
            # TPS production universe. Reusing those guarantees after expanding to a
            # general enzyme/reaction universe would be statistically invalid.
            query["empirical_reliability_status"] = "not_applicable_expanded_candidate_universe"
            query["empirical_reliability_tier"] = "uncalibrated"
            query["reliability_recommendation"] = "manual_review_general_universe"
            conformal = dict(query.get("conformal_retrieval_set") or {})
            conformal.update(
                {
                    "status": "not_applicable_expanded_candidate_universe",
                    "guarantee_scope": "historical_tps_universe_only",
                    "recommendation": "manual_review_general_universe",
                }
            )
            query["conformal_retrieval_set"] = conformal
        query["requested_shot_mode"] = "few_shot" if requested_seed_ids else "zero_shot"
        query["shot_mode"] = "few_shot" if effective_few_shot else "zero_shot"
        query["seed_ids"] = requested_seed_ids
        query["seed_count"] = len(requested_seed_ids)
        query["mask_ids"] = masked_ids
        query["mask_count"] = len(masked_ids)
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
