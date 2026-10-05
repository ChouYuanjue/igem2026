from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

TPS_SPECIALIZED_UNIVERSE = "tps_specialized"
GENERAL_UNIVERSE = "general_merged"


@dataclass(frozen=True)
class ExpertDecision:
    expert: str
    reason: str
    model_root: Path | None = None
    force_direct_zero_shot: bool = False
    ranking_objective: str | None = None

    @property
    def model_dir(self) -> Path | None:
        return self.model_root / "models" if self.model_root is not None else None


def _objective(payload: dict[str, Any]) -> str:
    requested = str(payload.get("ranking_objective") or "auto")
    if requested != "auto":
        return requested
    top_k = int(payload.get("top_k") or 10)
    if top_k <= 3:
        return "top3"
    if top_k <= 10:
        return "top10"
    return "top20"


def decide_expert(
    command: str,
    payload: dict[str, Any],
    *,
    adamerging_root: Path | None,
    full_root: Path | None,
) -> ExpertDecision:
    if command not in {"rank-enzymes", "rank-reactions"}:
        raise ValueError(f"Unsupported retrieval command: {command}")
    if payload.get("model_dir") is not None or payload.get("dual_tower_dir") is not None:
        return ExpertDecision("internal_override", "preexisting_server_model_override")

    universe = str(payload.get("candidate_universe") or TPS_SPECIALIZED_UNIVERSE)
    objective = _objective(payload)
    if universe == TPS_SPECIALIZED_UNIVERSE:
        return ExpertDecision("tps_legacy", "tps_specialized_candidate_universe", ranking_objective=objective)

    # The complete general-universe model is now owned by the canonical
    # production manifest (and, for registered entities, by FinalBridgeRuntime in
    # ModelGateway). Do not inject the historical 2115-d AdaMerging/full model
    # directories: they predate the current general reaction feature registry and
    # can silently change both model semantics and feature dimensionality.
    return ExpertDecision(
        "production_manifest",
        "general_candidate_universe_uses_current_production_manifest",
        ranking_objective=objective,
    )


def route_payload(command: str, payload: dict[str, Any]) -> tuple[dict[str, Any], ExpertDecision]:
    decision = decide_expert(
        command,
        payload,
        adamerging_root=None,
        full_root=None,
    )
    return dict(payload), decision
