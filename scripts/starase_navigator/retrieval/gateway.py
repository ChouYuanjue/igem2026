from __future__ import annotations

import os
import threading
import time
from typing import Any

from projects.active.bridge.core.engine import RetrievalEngine
from projects.active.bridge.core.candidate_universes import (
    DEFAULT_CANDIDATE_UNIVERSE,
    MARTS_CORRESPONDENCE_UNIVERSE,
)
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from scripts.starase_navigator.retrieval.focused import CorrespondenceGeometryService
from scripts.starase_navigator.retrieval.backend_router import route_payload


class ModelGateway:
    """Own the production retrieval engine and model warm-up lifecycle.

    The application layer can ask for rankings or pre-warm the fixed raw-sequence
    encoder without knowing how model instances, locks, or accelerator state are
    managed. User requests never control model paths through this gateway.
    """

    def __init__(self) -> None:
        self._engine: RetrievalEngine | None = None
        self._engine_lock = threading.Lock()
        self._protein_encoder_warmup_lock = threading.Lock()
        self._protein_encoder_warmup: dict[str, Any] = {"status": "idle"}
        self._correspondence_lock = threading.RLock()
        self._correspondence_service: CorrespondenceGeometryService | None = None
        self._final_bridge_lock = threading.RLock()
        self._final_bridge: FinalBridgeRuntime | None = None
        self._final_bridge_error: str | None = None

    def engine(self) -> RetrievalEngine:
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:
                    # Overrides are required only for server-created temporary seed
                    # files. The HTTP API never accepts arbitrary model/deployment paths.
                    self._engine = RetrievalEngine(allow_overrides=True)
        return self._engine

    def final_bridge(self) -> FinalBridgeRuntime:
        with self._final_bridge_lock:
            if self._final_bridge is None:
                try:
                    self._final_bridge = FinalBridgeRuntime()
                except Exception as exc:
                    self._final_bridge_error = f"{type(exc).__name__}: {exc}"
                    raise
                else:
                    self._final_bridge_error = None
            return self._final_bridge

    def final_bridge_status(self) -> dict[str, Any]:
        try:
            status = dict(self.final_bridge().status())
        except Exception:
            return {
                "status": "failed",
                "load_error": self._final_bridge_error or "unknown final BRIDGE load failure",
            }
        status.update(
            {
                "routing_role": "authoritative_registered_general_retrieval",
                "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
                "integrity_verified": True,
            }
        )
        return status

    def correspondence_service(self) -> CorrespondenceGeometryService:
        with self._correspondence_lock:
            if self._correspondence_service is None:
                self._correspondence_service = CorrespondenceGeometryService()
            return self._correspondence_service

    def correspondence_contains_protein(self, value: str) -> bool:
        return self.correspondence_service().contains_protein(value)

    def correspondence_contains_reaction(self, value: str) -> bool:
        return self.correspondence_service().contains_reaction(value)

    def application_profile_status(self) -> dict[str, Any]:
        return self.correspondence_service().application_profile_status()

    def enzymology_evidence_status(self) -> dict[str, Any]:
        return self.correspondence_service().enzymology_evidence_status()

    @staticmethod
    def _registered_query_id(command: str, payload: dict[str, Any]) -> str:
        key = "reaction_id" if command == "rank-enzymes" else "enzyme_id"
        return str(payload.get(key) or "").strip()

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command not in {"rank-enzymes", "rank-reactions"}:
            raise ValueError(f"unsupported ranking command: {command}")
        universe = str(payload.get("candidate_universe") or DEFAULT_CANDIDATE_UNIVERSE)

        # The MARTS correspondence universe is retained only as an explicit
        # historical/compatibility route. Normal COMPASS planning no longer selects
        # it, because application-domain intent must not shrink the ranking pool.
        if universe == MARTS_CORRESPONDENCE_UNIVERSE:
            result = self.correspondence_service().rank(command, dict(payload))
            query = result.setdefault("query", {})
            query.update({
                "candidate_universe": MARTS_CORRESPONDENCE_UNIVERSE,
                "candidate_universe_description": (
                    "COMPASS full-information application domain: canonical BRIDGE correspondence "
                    "with catalytic/mechanistic resolutions and TPS-domain application refinement"
                ),
                "candidate_universe_specialized": True,
                "model_expert": "bridge",
                "model_expert_reason": "COMPASS application-profile BRIDGE over the explicit MARTS/TPS candidate scope",
                "model_expert_objective": str(payload.get("ranking_objective") or "top10"),
                "model_expert_policy": "candidate_scope_contract_v1",
            })
            return result

        # Registered entities in the full general universe use the frozen final
        # BRIDGE composition directly. Raw SMILES/sequence queries remain on the
        # open-world production manifest until every final expert defines an
        # external-input representation.
        query_id = self._registered_query_id(command, payload)
        if universe == DEFAULT_CANDIDATE_UNIVERSE and query_id:
            runtime = self.final_bridge()
            if runtime.contains(command, query_id):
                result = runtime.rank(command, dict(payload))
                query = result.setdefault("query", {})
                query.update(
                    {
                        "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
                        "candidate_universe_description": (
                            "Full BRIDGE deployment universe with Broad retrieval and "
                            "inference-driven gated experts"
                        ),
                        "candidate_universe_specialized": False,
                        "model_expert": "bridge_final",
                        "model_expert_reason": (
                            "registered entity scored by the frozen complete BRIDGE system"
                        ),
                        "model_expert_objective": str(
                            payload.get("ranking_objective") or "top10"
                        ),
                        "model_expert_policy": "broad_gated_experts_v1",
                    }
                )
                return result

        routed_payload, decision = route_payload(command, payload)
        result = self.engine().rank(command, routed_payload)
        query = result.setdefault("query", {})
        query["model_expert"] = decision.expert
        query["model_expert_reason"] = decision.reason
        query["model_expert_objective"] = decision.ranking_objective
        query["model_expert_policy"] = "domain_and_budget_post_hoc_v1"
        return result

    def _run_protein_encoder_warmup(self) -> dict[str, Any]:
        started = time.time()
        try:
            details = self.engine().prewarm_protein_encoder()
        except Exception as exc:
            result: dict[str, Any] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.time() - started) * 1000, 2),
            }
        else:
            result = {
                **details,
                "status": "ready",
                "latency_ms": round((time.time() - started) * 1000, 2),
            }
        with self._protein_encoder_warmup_lock:
            policy = self._protein_encoder_warmup.get("policy")
            if policy:
                result["policy"] = policy
            self._protein_encoder_warmup = result
        return dict(result)

    def prewarm_protein_encoder(self, *, background: bool = True) -> dict[str, Any]:
        """Warm the fixed raw-sequence encoder once, optionally in a daemon thread."""
        with self._protein_encoder_warmup_lock:
            status = str(self._protein_encoder_warmup.get("status") or "idle")
            if status in {"warming", "ready"}:
                return dict(self._protein_encoder_warmup)
            self._protein_encoder_warmup = {
                "status": "warming",
                "started_at_unix": time.time(),
            }
        if not background:
            return self._run_protein_encoder_warmup()
        threading.Thread(
            target=self._run_protein_encoder_warmup,
            name="starase-navigator-esmc-prewarm",
            daemon=True,
        ).start()
        return {"status": "warming"}

    def startup_prewarm_protein_encoder(
        self,
        *,
        mode: str | None = None,
        cuda_available: bool | None = None,
    ) -> dict[str, Any]:
        """Apply the service-startup warm-up policy without blocking HTTP readiness.

        ``auto`` (the default) starts a background warm-up only when CUDA is
        available. ``on``/``force`` always starts it and ``off`` disables it.
        The model name, path, and device are still fixed by the production engine;
        this setting controls lifecycle only.
        """

        configured = str(
            mode
            if mode is not None
            else os.environ.get("STARASE_NAVIGATOR_PROTEIN_ENCODER_PREWARM", "auto")
        ).strip().lower()
        if configured in {"0", "off", "false", "disabled", "none"}:
            with self._protein_encoder_warmup_lock:
                current = str(self._protein_encoder_warmup.get("status") or "idle")
                if current not in {"warming", "ready"}:
                    self._protein_encoder_warmup = {
                        "status": "disabled",
                        "policy": configured or "off",
                    }
                return dict(self._protein_encoder_warmup)

        if configured not in {"auto", "1", "on", "true", "force"}:
            with self._protein_encoder_warmup_lock:
                self._protein_encoder_warmup = {
                    "status": "deferred",
                    "reason": "invalid_startup_policy",
                    "policy": configured,
                }
                return dict(self._protein_encoder_warmup)

        if configured == "auto":
            if cuda_available is None:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            if not cuda_available:
                with self._protein_encoder_warmup_lock:
                    current = str(self._protein_encoder_warmup.get("status") or "idle")
                    if current not in {"warming", "ready"}:
                        self._protein_encoder_warmup = {
                            "status": "deferred",
                            "reason": "cuda_unavailable",
                            "policy": "auto",
                        }
                    return dict(self._protein_encoder_warmup)

        result = self.prewarm_protein_encoder(background=True)
        with self._protein_encoder_warmup_lock:
            self._protein_encoder_warmup.setdefault("policy", configured)
            return dict(self._protein_encoder_warmup)

    def protein_encoder_status(self) -> dict[str, Any]:
        with self._protein_encoder_warmup_lock:
            return dict(self._protein_encoder_warmup)
