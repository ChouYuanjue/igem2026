from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TPS_SPECIALIZED_UNIVERSE = "tps_specialized"
GENERAL_UNIVERSE = "general_merged"
DEFAULT_ADAMERGING_EXPERT = ROOT / "results/terpene_production_models/general_broad_adamerging_v1"
DEFAULT_FULL_EXPERT = ROOT / "results/terpene_production_models/general_broad_full_v1"


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


def _read_ids(path: Path, columns: tuple[str, ...]) -> set[str]:
    if not path.is_file():
        return set()
    values: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        column = next((candidate for candidate in columns if candidate in fieldnames), None)
        if column is None:
            raise ValueError(f"No identifier column {columns} found in {path}")
        for row in reader:
            value = str(row.get(column) or "").strip()
            if value:
                values.add(value)
    return values


@lru_cache(maxsize=1)
def specialist_query_ids() -> tuple[frozenset[str], frozenset[str]]:
    proteins = _read_ids(ROOT / "data/terpene_embeddings/esmc600m_mean/entries.csv", ("Entry", "protein_id"))
    proteins.update(_read_ids(ROOT / "data/terpene_open_world_registry/proteins/entries.csv", ("Entry", "protein_id")))
    reactions = _read_ids(
        ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/reaction_registry.csv",
        ("reaction_id", "rhea_id"),
    )
    return frozenset(proteins), frozenset(reactions)


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


def _has_seed(command: str, payload: dict[str, Any]) -> bool:
    key = "known_enzyme_ids" if command == "rank-enzymes" else "known_reaction_ids"
    return bool([value for value in (payload.get(key) or []) if str(value).strip()])


def decide_expert(
    command: str,
    payload: dict[str, Any],
    *,
    specialist_proteins: set[str] | frozenset[str],
    specialist_reactions: set[str] | frozenset[str],
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

    query_id = str(payload.get("reaction_id") or "") if command == "rank-enzymes" else str(payload.get("enzyme_id") or "")
    if query_id:
        specialist = query_id in specialist_reactions if command == "rank-enzymes" else query_id in specialist_proteins
        if specialist:
            return ExpertDecision("tps_legacy", "query_is_tps_specialist_entity", ranking_objective=objective)

    # Existing few-shot semantics are intentionally preserved. R2E seed routing is
    # model-independent protein-space retrieval; E2R seed routing has not yet been
    # re-audited for the broad expert and therefore stays on the legacy bundle.
    if _has_seed(command, payload):
        return ExpertDecision("tps_legacy", "few_shot_seed_route_preserved", ranking_objective=objective)

    retrieval_mode = str(payload.get("retrieval_mode") or "auto")
    force_direct = retrieval_mode == "auto"
    # Full broad continuation had slightly better R2E Hit@20 than AdaMerging; all
    # other audited zero-shot broad objectives use the layer-wise AdaMerging expert.
    if command == "rank-enzymes" and objective == "top20":
        if full_root is not None and (full_root / "models").is_dir():
            return ExpertDecision(
                "general_full_directional",
                "general_zero_shot_r2e_top20",
                model_root=full_root,
                force_direct_zero_shot=force_direct,
                ranking_objective=objective,
            )
    else:
        if adamerging_root is not None and (adamerging_root / "models").is_dir():
            return ExpertDecision(
                "general_adamerging",
                "general_zero_shot_budget_route",
                model_root=adamerging_root,
                force_direct_zero_shot=force_direct,
                ranking_objective=objective,
            )

    return ExpertDecision("tps_legacy", "general_expert_unavailable", ranking_objective=objective)


def configured_expert_roots() -> tuple[Path | None, Path | None]:
    ada_raw = os.environ.get("CATALYST_GENERAL_ADAMERGING_DIR", str(DEFAULT_ADAMERGING_EXPERT)).strip()
    full_raw = os.environ.get("CATALYST_GENERAL_FULL_DIR", str(DEFAULT_FULL_EXPERT)).strip()
    ada = Path(ada_raw).resolve() if ada_raw else None
    full = Path(full_raw).resolve() if full_raw else None
    return ada, full


def route_payload(command: str, payload: dict[str, Any]) -> tuple[dict[str, Any], ExpertDecision]:
    proteins, reactions = specialist_query_ids()
    ada, full = configured_expert_roots()
    decision = decide_expert(
        command,
        payload,
        specialist_proteins=proteins,
        specialist_reactions=reactions,
        adamerging_root=ada,
        full_root=full,
    )
    routed = dict(payload)
    if decision.model_dir is not None:
        routed["model_dir"] = decision.model_dir
        if decision.force_direct_zero_shot:
            routed["retrieval_mode"] = "direct"
    return routed, decision
