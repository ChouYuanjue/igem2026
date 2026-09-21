from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.core.candidate_universes import (  # noqa: E402
    DEFAULT_CANDIDATE_UNIVERSE,
)
from scripts.database_bridge.model_catalog import ModelDataCatalog  # noqa: E402
from scripts.starase_navigator.agent.resolution import AgentResolutionService  # noqa: E402
from scripts.starase_navigator.agent_harness.capabilities import public_capabilities  # noqa: E402
from scripts.starase_navigator.agent_harness.harness import ScientificAgentHarness  # noqa: E402
from scripts.starase_navigator.agent_harness.session_store import AgentSessionStore  # noqa: E402
from scripts.starase_navigator.agent_harness.tool_registry import ScientificToolRegistry  # noqa: E402
from scripts.starase_navigator.evidence_catalog import IntegratedEvidenceCatalog  # noqa: E402
from scripts.starase_navigator.evidence_query_service import AssociationEvidenceQueryService  # noqa: E402
from scripts.starase_navigator.errors import AppError  # noqa: E402
from scripts.starase_navigator.routing.enzyme_to_reaction import E2RRoutePlanner  # noqa: E402
from scripts.starase_navigator.homology import ProteinHomologyIndex  # noqa: E402
from scripts.starase_navigator.http_transport import Handler  # noqa: E402
from scripts.starase_navigator.routing.language import DeepSeekResolver  # noqa: E402
from scripts.starase_navigator.retrieval.gateway import ModelGateway  # noqa: E402
from scripts.starase_navigator.open_world_inputs import ProteinSequenceInput  # noqa: E402
from scripts.starase_navigator.protein_resolution import ProteinResolver  # noqa: E402
from scripts.starase_navigator.runtime_store import RuntimeStore  # noqa: E402
from scripts.starase_navigator.pathway_compatibility import PathwayCompatibilityAnalyzer  # noqa: E402
from scripts.starase_navigator.protein_family_catalog import ProteinFamilyCatalog  # noqa: E402
from scripts.starase_navigator.protein_family_service import ProteinFamilyEvidenceService  # noqa: E402
from scripts.starase_navigator.retrieval.service import RetrievalApplicationService  # noqa: E402
from scripts.starase_navigator.scientific_research_service import ScientificResearchService  # noqa: E402
from scripts.starase_navigator.rhea_client import RheaClient, canonical_rhea_id  # noqa: E402,F401
from scripts.starase_navigator.route_design import RheaRouteDesigner  # noqa: E402
from scripts.starase_navigator.resolution_helpers import (  # noqa: E402,F401
    candidate_match as _candidate_match,
    explicit_uniprot_accession as _explicit_uniprot_accession,
    fallback_queries as _fallback_queries,
)
from scripts.starase_navigator.route_feasibility import RouteFeasibilityAnalyzer  # noqa: E402
from scripts.starase_navigator.route_pathway_service import RoutePathwayService  # noqa: E402
from scripts.starase_navigator.route_view import system_route_catalog  # noqa: E402
from scripts.starase_navigator.routing.reaction_to_enzyme import RoutePlanner  # noqa: E402

STATIC_ROOT = ROOT / "frontend/starase_navigator"
RUNTIME_ROOT = ROOT / "results/starase_navigator_runtime"
CACHE_ROOT = RUNTIME_ROOT / "cache"
FEEDBACK_PATH = RUNTIME_ROOT / "feedback.jsonl"
RUN_EVENTS_PATH = RUNTIME_ROOT / "run_events.jsonl"

DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
USER_AGENT = "NJU-iGEM-2026-StaraseNavigator/1.0"


def _runtime_source_fingerprint() -> str:
    hasher = hashlib.sha256()
    files: list[Path] = []
    navigator_root = ROOT / "scripts/starase_navigator"
    for path in navigator_root.rglob("*.py"):
        if "__pycache__" in path.parts or path.name.startswith("test_"):
            continue
        files.append(path)
    files.extend([
        ROOT / "frontend/starase_navigator/index.html",
        ROOT / "frontend/starase_navigator/app.js",
        ROOT / "frontend/starase_navigator/styles.css",
        ROOT / "projects/active/fibre/core/candidate_universes.py",
        ROOT / "projects/active/fibre/geometry/extension.py",
        ROOT / "projects/active/fibre/geometry/multiscale.py",
        ROOT / "projects/active/fibre/runtime/entities.py",
    ])
    for path in sorted({value.resolve() for value in files if value.is_file()}, key=str):
        relative = path.relative_to(ROOT.resolve())
        hasher.update(str(relative).encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(block)
        hasher.update(b"\0")
    return hasher.hexdigest()[:12]


def _build_revision() -> str:
    configured = str(os.environ.get("STARASE_NAVIGATOR_BUILD_REVISION") or "").strip()
    if configured:
        return configured[:64]
    source = _runtime_source_fingerprint()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        head = completed.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        head = "unknown"
    return f"{head}+src-{source}"


class ProductionHTTPServer(ThreadingHTTPServer):
    """Small production wrapper around the standard threaded HTTP server."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

class NavigatorRuntime:
    def __init__(self) -> None:
        self.started_at_unix = time.time()
        self.source_fingerprint = _runtime_source_fingerprint()
        self.build_revision = _build_revision()
        self.catalog = ModelDataCatalog(ROOT)
        self.evidence = IntegratedEvidenceCatalog(ROOT)
        self.rhea = RheaClient(CACHE_ROOT)
        self.deepseek = DeepSeekResolver()
        self.proteins = ProteinResolver(self.catalog, user_agent=USER_AGENT)
        self.families = ProteinFamilyCatalog(ROOT, self.evidence.candidate_protein_ids())
        self.family_evidence = ProteinFamilyEvidenceService(
            families=self.families,
            evidence=self.evidence,
            rhea=self.rhea,
            proteins=self.proteins,
        )
        self.evidence_queries = AssociationEvidenceQueryService(
            evidence=self.evidence,
            families=self.families,
            proteins=self.proteins,
            rhea=self.rhea,
            deepseek=self.deepseek,
            catalog=self.catalog,
        )
        self.route_planner = RoutePlanner(
            proposal_fn=self.deepseek.select_route,
            protein_ids=self.evidence.candidate_protein_ids(),
        )
        self.e2r_planner = E2RRoutePlanner(proposal_fn=self.deepseek.select_e2r_route)
        self.homology = ProteinHomologyIndex()
        self.pathway = PathwayCompatibilityAnalyzer(
            root=ROOT,
            catalog=self.catalog,
            rank_reaction=self.rank,
            user_agent=USER_AGENT,
            cache_root=CACHE_ROOT,
        )
        self.route_designer = RheaRouteDesigner(
            root=ROOT,
            user_agent=USER_AGENT,
            cache_root=CACHE_ROOT,
        )
        self.family_evidence.official_rhea = self.route_designer
        self.route_feasibility = RouteFeasibilityAnalyzer(ROOT, self.route_designer)
        self._route_catalog = system_route_catalog()
        self.route_pathway = RoutePathwayService(
            catalog=self.catalog,
            deepseek=self.deepseek,
            proteins=self.proteins,
            route_designer=self.route_designer,
            route_feasibility=self.route_feasibility,
            pathway=self.pathway,
            resolve_reaction=lambda text: self.agent_resolution.resolve(text),
            resolve_reaction_from_terms=lambda **kwargs: self.agent_resolution._resolve_reaction_from_terms(**kwargs),
            rank_model=lambda command, payload: self.model_gateway.rank(command, payload),
        )
        self.agent_resolution = AgentResolutionService(
            catalog=self.catalog,
            evidence=self.evidence,
            rhea=self.rhea,
            deepseek=self.deepseek,
            proteins=self.proteins,
            families=self.families,
            family_evidence=self.family_evidence,
            evidence_queries=self.evidence_queries,
            route_design_resolve=self.route_pathway.route_design_resolve,
            pathway_resolve=self.route_pathway.pathway_resolve,
        )
        self.model_gateway = ModelGateway()
        require_application = str(
            os.environ.get("STARASE_NAVIGATOR_REQUIRE_APPLICATION_PROFILE","")
        ).strip().lower() in {"1","true","yes","on"}
        if require_application:
            application_status = self.model_gateway.application_profile_status()
            if (
                str(application_status.get("status") or "") != "ready"
                or not bool(application_status.get("integrity_verified"))
                or not bool(
                    application_status.get(
                        "primary_runtime_verified_against_promoted_geometry"
                    )
                )
            ):
                raise RuntimeError(
                    "Starase full-information application profile is required but not "
                    "strictly verified: "
                    + str(application_status.get("load_error") or application_status)
                )
            enzymology_status=self.model_gateway.enzymology_evidence_status()
            if str(enzymology_status.get("status") or "")!="ready":
                raise RuntimeError(
                    "Starase full-information application profile requires scoped "
                    "enzymology evidence but it is unavailable: "
                    + str(enzymology_status.get("load_error") or enzymology_status)
                )
        self.retrieval_service = RetrievalApplicationService(
            catalog=self.catalog,
            evidence=self.evidence,
            proteins=self.proteins,
            rhea=self.rhea,
            route_planner=self.route_planner,
            e2r_planner=self.e2r_planner,
            homology=self.homology,
            route_designer=self.route_designer,
            model_gateway=self.model_gateway,
        )
        self.research_service = ScientificResearchService(
            evidence=self.evidence,
            evidence_queries=self.evidence_queries,
            proteins=self.proteins,
            rhea=self.rhea,
            route_designer=self.route_designer,
            model_gateway=self.model_gateway,
            catalog=self.catalog,
            user_agent=USER_AGENT,
            deepseek=self.deepseek,
            retrieval_service=self.retrieval_service,
            cache_root=CACHE_ROOT,
        )
        self.agent_sessions = AgentSessionStore(ttl_seconds=7200, max_sessions=512)
        self.agent_tools = ScientificToolRegistry(
            agent_resolution=self.agent_resolution,
            deepseek=self.deepseek,
            families=self.families,
            family_evidence=self.family_evidence,
            evidence_queries=self.evidence_queries,
            route_design_resolve=self.route_pathway.route_design_resolve,
            pathway_resolve=self.route_pathway.pathway_resolve,
            compound_resolve=self.route_designer.resolve_compound,
            research_service=self.research_service,
            candidate_execute=self._execute_prepared_candidate_search,
            route_execute=self._execute_prepared_route_design,
            route_patch_execute=self._execute_route_segment_patch,
            pathway_execute=self._execute_prepared_pathway_analysis,
        )
        self.agent_harness = ScientificAgentHarness(
            deepseek=self.deepseek,
            tools=self.agent_tools,
            sessions=self.agent_sessions,
            max_turns=8,
        )
        self.runtime_store = RuntimeStore(
            feedback_path=FEEDBACK_PATH,
            run_events_path=RUN_EVENTS_PATH,
        )

    @property
    def feedback_path(self) -> Path:
        return self.runtime_store.feedback_path

    @feedback_path.setter
    def feedback_path(self, value: Path) -> None:
        self.runtime_store.feedback_path = Path(value)

    @property
    def run_events_path(self) -> Path:
        return self.runtime_store.run_events_path

    @run_events_path.setter
    def run_events_path(self, value: Path) -> None:
        self.runtime_store.run_events_path = Path(value)

    def record_run_event(self, **kwargs: Any) -> dict[str, Any]:
        return self.runtime_store.record_run_event(**kwargs)

    def hold_run_step(self, run_id: str, step: dict[str, Any]) -> None:
        self.runtime_store.hold_run_step(run_id, step)

    def take_run_steps(self, run_id: str) -> list[dict[str, Any]]:
        return self.runtime_store.take_run_steps(run_id)


    def engine(self):
        """Compatibility accessor; application code should use ``model_gateway``."""
        return self.model_gateway.engine()

    def prewarm_protein_encoder(self, *, background: bool = True) -> dict[str, Any]:
        return self.model_gateway.prewarm_protein_encoder(background=background)

    def startup_prewarm_protein_encoder(self) -> dict[str, Any]:
        return self.model_gateway.startup_prewarm_protein_encoder()

    def capabilities(self) -> dict[str, Any]:
        payload = public_capabilities()
        tool_catalog = self.agent_tools.catalog()
        payload["tool_count"] = len(tool_catalog)
        payload["tools"] = [
            {
                "name": str(item.get("name") or ""),
                "purpose": str(item.get("purpose") or ""),
            }
            for item in tool_catalog
        ]
        return payload

    def status(self) -> dict[str, Any]:
        project_summary = self.catalog.summary()
        evidence_summary = self.evidence.summary()
        return {
            "status": "ready",
            "service": "starase_navigator",
            "build_revision": self.build_revision,
            "source_fingerprint": self.source_fingerprint,
            "process_id": os.getpid(),
            "uptime_seconds": round(max(0.0, time.time() - self.started_at_unix), 2),
            "deepseek_configured": self.deepseek.configured,
            "deepseek_model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            "deepseek": self.deepseek.provenance(),
            "route_planner": "langgraph",
            "agent_controller": "model_led_scientific_harness",
            "agent_entrypoint": "/api/agent/resolve",
            "agent_capabilities_version": str(public_capabilities().get("version") or "unknown"),
            "agent_directions": ["reaction_to_enzyme", "enzyme_to_reaction", "route_design", "pathway_compatibility"],
            "natural_language_resolution": ["reaction", "protein", "positive_enzyme"],
            "default_route": {
                "top_k": 10,
                "enzyme_taxonomy_scope": "all",
                "shot_mode": "few_shot_if_database_positive_else_zero_shot",
                "homology_policy": "allow",
                "known_association_policy": "separate_known",
                "verified_application_domain": "starase-application",
                "outside_application_domain": "broad_general",
                "explicit_broad_request": "broad_general",
            },
            "result_scopes": ["separate_known", "rank_with_known", "known_only", "exclude_known"],
            "homology_definition": "MMseqs2 50% sequence identity, >=80% coverage",
            "homology_index_cached": self.homology.ready,
            "route_catalog": self._route_catalog["counts"],
            # Backward-compatible broad-universe summary. This is not the
            # default for a verified current Starase-domain entity; see
            # default_route and application_profile above.
            "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
            "candidate_universe_role": "broad_general_fallback_and_out_of_domain_search",
            "candidate_enzymes": evidence_summary["candidate_proteins"],
            "candidate_reactions": evidence_summary["candidate_reactions"],
            "recorded_associations": evidence_summary["recorded_associations"],
            "project_catalog": {
                "proteins": project_summary["proteins"],
                "reactions": project_summary["reactions"],
            },
            # Backward compatibility for clients that still read this field. It now
            # reports the active product reaction candidate universe, not the old
            # 753-reaction project catalog.
            "model_reactions": evidence_summary["candidate_reactions"],
            "open_world_protein_encoder": self.model_gateway.protein_encoder_status(),
            "application_profile": self.model_gateway.application_profile_status(),
            "enzymology_evidence": self.model_gateway.enzymology_evidence_status(),
            "feedback_enabled": True,
            "route_feasibility": self.route_feasibility.status(),
        }

    def update_view_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "").strip()
        entity_kind = str(payload.get("entity_kind") or "").strip()
        entity_ids = [str(value).strip() for value in payload.get("entity_ids") or [] if str(value).strip()]
        page_index = max(0, int(payload.get("page_index") or 0))
        return self.agent_sessions.mark_visible_entities(
            session_id, entity_kind=entity_kind, entity_ids=entity_ids, page_index=page_index,
        )

    def literature_page(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = str(payload.get("query") or "").strip()
        cursor = str(payload.get("cursor") or "*").strip() or "*"
        page_size = max(1, min(int(payload.get("page_size") or 10), 20))
        page_index = max(0, int(payload.get("page_index") or 0))
        provider = str(payload.get("provider") or "europe_pmc").strip().lower()
        panel = self.research_service.literature_page(
            query, cursor_mark=cursor, page_size=page_size, provider=provider,
        )
        session_id = str(payload.get("session_id") or "").strip()
        if session_id:
            page_items = [dict(row) for row in panel.get("items") or [] if isinstance(row, dict)]
            self.agent_sessions.remember_literature_items(
                session_id, page_items, start_index=page_index * page_size,
            )
            visible_ids = []
            for row in page_items:
                pmid = str(row.get("pmid") or "").strip()
                raw_id = str(row.get("id") or row.get("pmcid") or "").strip()
                source = str(row.get("source") or "").strip().upper()
                if pmid:
                    visible_ids.append(f"MED:{pmid}")
                elif raw_id:
                    visible_ids.append(raw_id if ":" in raw_id else f"{source}:{raw_id}" if source in {"MED", "PMC"} else raw_id)
            self.agent_sessions.mark_visible_entities(
                session_id, entity_kind="literature", entity_ids=visible_ids, page_index=page_index,
            )
        return panel

    def suggest_followups(self, payload: dict[str, Any]) -> dict[str, Any]:
        context = payload.get("result_context") if isinstance(payload.get("result_context"), dict) else {}
        session_id = str(payload.get("session_id") or "").strip()
        ui_language = str(payload.get("ui_language") or "en")
        session_facts = self.agent_sessions.model_snapshot(session_id) if session_id else {}
        items = self.deepseek.suggest_next_steps(
            result_context=context,
            session_facts=session_facts,
            tool_catalog=self.agent_tools.catalog(),
            ui_language=ui_language,
            limit=3,
        )
        return {"status": "ok", "items": items}

    def submit_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime_store.submit_feedback(payload)


    def _resolve_reaction_from_terms(
        self,
        *,
        substrate_terms: list[str],
        product_terms: list[str],
        interpreted_reaction: str = "",
        assumptions: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.agent_resolution._resolve_reaction_from_terms(
            substrate_terms=substrate_terms,
            product_terms=product_terms,
            interpreted_reaction=interpreted_reaction,
            assumptions=assumptions,
        )


    def resolve_protein(self, text: str) -> dict[str, Any]:
        return self.agent_resolution.resolve_protein(text)


    def route_design_resolve(self, text: str, ui_language: str = "en") -> dict[str, Any]:
        return self.route_pathway.route_design_resolve(text, ui_language=ui_language)


    def design_routes(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.route_pathway.design_routes(payload)


    def pathway_resolve(self, text: str, ui_language: str = "en") -> dict[str, Any]:
        return self.route_pathway.pathway_resolve(text, ui_language=ui_language)


    def analyze_pathway(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.route_pathway.analyze_pathway(payload)


    def _sequence_candidate_payload(self, item: ProteinSequenceInput) -> dict[str, Any]:
        return self.agent_resolution._sequence_candidate_payload(item)


    def session_audit_snapshot(self, session_id: str) -> dict[str, Any]:
        snapshot = self.agent_sessions.model_snapshot(session_id)
        entities = (snapshot.get("session_entities") or {}) if isinstance(snapshot, dict) else {}
        def compact(rows: Any) -> list[dict[str, Any]]:
            result = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                result.append({
                    "kind": str(row.get("kind") or ""),
                    "id": str(row.get("id") or ""),
                    "role": str(row.get("role") or ""),
                    "focus": bool(row.get("focus")),
                    "active": bool(row.get("active")),
                    "related_index": row.get("related_index"),
                })
            return result[:40]
        return {
            "last_direction": str(snapshot.get("last_direction") or ""),
            "last_target": str(snapshot.get("last_target") or ""),
            "active": compact(entities.get("active")),
            "history": compact(entities.get("history")),
            "related": compact(entities.get("related")),
        }

    def agent_resolve(
        self,
        text: str,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        result = self.agent_harness.run(
            text,
            conversation_context=dict(conversation_context or {}),
            ui_language=ui_language,
            session_id=session_id,
        )
        self.agent_sessions.remember_dialogue_turn(
            session_id,
            user_text=text,
            assistant_text=str(
                result.get("assistant_response")
                or result.get("summary")
                or ""
            ),
            response_type=str(result.get("response_type") or ""),
        )
        return result

    def _execute_prepared_candidate_search(
        self,
        *,
        resolution: dict[str, Any],
        user_text: str,
        session_id: str,
        ui_language: str,
    ) -> dict[str, Any]:
        direction = str(resolution.get("direction") or "").strip()
        target_conditions = (
            dict(resolution.get("target_conditions") or {})
            if isinstance(resolution.get("target_conditions"), dict)
            else {}
        )
        retrieval_plan = (
            dict(resolution.get("retrieval_plan") or {})
            if isinstance(resolution.get("retrieval_plan"), dict)
            else {}
        )
        if direction == "reaction_to_enzyme":
            reaction = (
                dict(resolution.get("reaction_resolution") or {})
                if isinstance(resolution.get("reaction_resolution"), dict)
                else {}
            )
            target_id = str(reaction.get("recommended_id") or "").strip()
            candidates = [row for row in reaction.get("candidates") or [] if isinstance(row, dict)]
            selected = next(
                (
                    row for row in candidates
                    if str(row.get("rhea_id") or row.get("id") or row.get("query_id") or "").strip() == target_id
                ),
                candidates[0] if len(candidates) == 1 else {},
            )
            reaction_smiles = str(
                selected.get("reaction_smiles")
                or reaction.get("reaction_smiles")
                or ""
            ).strip()
            orientation = str(selected.get("orientation") or "forward").strip() or "forward"
            return self.rank(
                "" if reaction_smiles else target_id,
                reaction_smiles=reaction_smiles,
                query_id=target_id if reaction_smiles else "",
                orientation=orientation,
                user_text=user_text,
                target_conditions=target_conditions,
                retrieval_plan=retrieval_plan,
                confirmed_seed_ids=[],
                confirmed_seed_inputs=[],
                ui_language=ui_language,
                session_id=session_id,
            )

        if direction == "enzyme_to_reaction":
            protein = (
                dict(resolution.get("protein_resolution") or {})
                if isinstance(resolution.get("protein_resolution"), dict)
                else {}
            )
            target_id = str(protein.get("recommended_id") or "").strip()
            candidates = [row for row in protein.get("candidates") or [] if isinstance(row, dict)]
            selected = next(
                (
                    row for row in candidates
                    if str(row.get("id") or row.get("accession") or row.get("query_id") or "").strip() == target_id
                ),
                candidates[0] if len(candidates) == 1 else {},
            )
            enzyme_sequence = str(selected.get("sequence") or "").strip()
            reaction_constraints = (
                dict(resolution.get("reaction_constraints") or {})
                if isinstance(resolution.get("reaction_constraints"), dict)
                else {}
            )
            return self.rank_reactions(
                "" if enzyme_sequence else target_id,
                enzyme_sequence=enzyme_sequence,
                query_id=target_id if enzyme_sequence else "",
                user_text=user_text,
                target_conditions=target_conditions,
                reaction_constraints=reaction_constraints,
                retrieval_plan=retrieval_plan,
                confirmed_reaction_seed_ids=[],
                ui_language=ui_language,
                session_id=session_id,
            )

        raise AppError(
            "candidate_direction_invalid",
            "The prepared candidate workflow has no executable direction.",
            HTTPStatus.UNPROCESSABLE_ENTITY,
        )

    def _execute_prepared_route_design(
        self,
        *,
        resolution: dict[str, Any],
        user_text: str,
        session_id: str,
        ui_language: str,
    ) -> dict[str, Any]:
        route = (
            dict(resolution.get("route_design_resolution") or {})
            if isinstance(resolution.get("route_design_resolution"), dict)
            else {}
        )
        payload = {
            "source_chebi_id": str(route.get("recommended_source_id") or ""),
            "target_chebi_id": str(route.get("recommended_target_id") or ""),
            "target_terms": list(route.get("target_terms") or []),
            "host": str(route.get("host") or ""),
            "max_steps": int(route.get("max_steps") or 6),
            "route_count": int(route.get("route_count") or 10),
            "priority": str(route.get("priority") or "balanced"),
            "exploration_policy": str(route.get("exploration_policy") or "known_first"),
            "analysis_layers": list(route.get("analysis_layers") or []),
            "user_text": str(user_text or ""),
            "session_id": str(session_id or ""),
            "ui_language": str(ui_language or "en"),
        }
        return self.design_routes(payload)

    def _execute_route_segment_patch(
        self,
        *,
        route: dict[str, Any],
        segment_steps: list[dict[str, Any]],
        replacement_count: int,
        max_replacement_steps: int | None,
        session_id: str,
        ui_language: str,
    ) -> dict[str, Any]:
        parent_route_id = str(route.get("route_id") or "").strip()
        parent_steps = sorted(
            [dict(step) for step in route.get("steps") or [] if isinstance(step, dict)],
            key=lambda step: int(step.get("step_index") or 0),
        )
        if not parent_route_id or not parent_steps:
            raise AppError("route_patch_parent_invalid", "The selected route has no reusable route steps.", HTTPStatus.UNPROCESSABLE_ENTITY)

        selected_indices = sorted({int(row.get("step_index") or 0) for row in segment_steps})
        if not selected_indices or selected_indices != list(range(selected_indices[0], selected_indices[-1] + 1)):
            raise AppError("route_patch_segment_invalid", "The selected route segment is not contiguous.", HTTPStatus.UNPROCESSABLE_ENTITY)
        parent_by_index = {int(step.get("step_index") or 0): step for step in parent_steps}
        for row in segment_steps:
            index = int(row.get("step_index") or 0)
            selected_step = row.get("step") if isinstance(row.get("step"), dict) else {}
            parent_step = parent_by_index.get(index)
            if parent_step is None:
                raise AppError("route_patch_step_missing", "A selected step is no longer present in the parent route.", HTTPStatus.UNPROCESSABLE_ENTITY)
            if selected_step:
                for key in ("rhea_id", "source", "target"):
                    if str(selected_step.get(key) or "") != str(parent_step.get(key) or ""):
                        raise AppError("route_patch_step_stale", "The selected route-step handle no longer matches the parent route.", HTTPStatus.CONFLICT)

        start_index, end_index = selected_indices[0], selected_indices[-1]
        original_segment = [parent_by_index[index] for index in selected_indices]
        prefix = [step for step in parent_steps if int(step.get("step_index") or 0) < start_index]
        suffix = [step for step in parent_steps if int(step.get("step_index") or 0) > end_index]
        boundary_source = str(original_segment[0].get("source") or "").strip()
        boundary_target = str(original_segment[-1].get("target") or "").strip()
        if not boundary_source or not boundary_target:
            raise AppError("route_patch_boundary_missing", "The selected route segment has no fixed compound boundaries.", HTTPStatus.UNPROCESSABLE_ENTITY)

        search_context = route.get("search_context") if isinstance(route.get("search_context"), dict) else {}
        priority = str(search_context.get("priority") or "balanced")
        if priority not in {"balanced", "short", "enzyme_available", "project_covered", "thermodynamic", "host_flux"}:
            priority = "balanced"
        host = str(search_context.get("host") or "")
        whole_route_max_steps = max(
            len(parent_steps),
            min(8, max(1, int(search_context.get("max_steps") or len(parent_steps)))),
        )
        preserved_step_count = len(prefix) + len(suffix)
        available_segment_steps = max(1, whole_route_max_steps - preserved_step_count)
        requested_segment_steps = (
            max(1, min(8, int(max_replacement_steps)))
            if max_replacement_steps is not None
            else available_segment_steps
        )
        segment_step_budget = min(available_segment_steps, requested_segment_steps)
        replaced_rhea_ids = sorted({
            str(step.get("rhea_id") or "").strip()
            for step in original_segment
            if str(step.get("rhea_id") or "").strip()
        })
        excluded_rhea_ids = sorted({
            str(step.get("rhea_id") or "").strip()
            for step in parent_steps
            if str(step.get("rhea_id") or "").strip()
        })
        parent_compound_ids = [str(value).strip() for value in route.get("compound_ids") or [] if str(value).strip()]
        forbidden_internal_compounds = set(parent_compound_ids) - {boundary_source, boundary_target}
        replacement_count = max(1, min(int(replacement_count or 3), 10))
        candidate_limit = min(40, max(12, replacement_count * 4))

        segment_search = self.route_designer.design(
            source_terms=[boundary_source],
            target_terms=[boundary_target],
            host="",
            max_steps=segment_step_budget,
            limit=replacement_count,
            candidate_limit=candidate_limit,
            priority=priority,
            local_reaction_ids=self.catalog.reaction_by_id.keys(),
            excluded_reaction_ids=excluded_rhea_ids,
        )

        patched_routes: list[dict[str, Any]] = []
        for replacement in segment_search.get("routes") or []:
            replacement_steps = [dict(step) for step in replacement.get("steps") or [] if isinstance(step, dict)]
            if not replacement_steps:
                continue
            replacement_compounds = [str(value).strip() for value in replacement.get("compound_ids") or [] if str(value).strip()]
            if any(compound in forbidden_internal_compounds for compound in replacement_compounds[1:-1]):
                continue
            merged_steps = [dict(step) for step in prefix] + replacement_steps + [dict(step) for step in suffix]
            if len(merged_steps) > whole_route_max_steps:
                continue
            merged_nodes = [str(merged_steps[0].get("source") or "")] + [
                str(step.get("target") or "") for step in merged_steps
            ]
            if any(not node for node in merged_nodes) or len(merged_nodes) != len(set(merged_nodes)):
                continue
            patched = self.route_designer.materialize_route(
                merged_steps,
                max_steps=whole_route_max_steps,
                priority=priority,
                local_reaction_ids=self.catalog.reaction_by_id.keys(),
                route_id_prefix="RP",
                route_id_seed=f"{parent_route_id}:{start_index}-{end_index}",
            )
            patched["route_type"] = "patched_known_rhea"
            patched["parent_route_id"] = parent_route_id
            patched["root_route_id"] = str(
                route.get("root_route_id")
                or route.get("parent_route_id")
                or parent_route_id
            )
            patched["generation"] = int(route.get("generation") or 0) + 1
            patched["search_context"] = {
                "priority": priority,
                "host": host,
                "max_steps": whole_route_max_steps,
                "analysis_layers": [],
                "exploration_policy": str(search_context.get("exploration_policy") or "known_first"),
            }
            patched["patch"] = {
                "start_step_index": start_index,
                "end_step_index": end_index,
                "original_rhea_ids": replaced_rhea_ids,
                "replacement_segment_route_id": str(replacement.get("route_id") or ""),
                "replacement_segment_score": replacement.get("score"),
                "replacement_step_count": len(replacement_steps),
                "preserved_prefix_step_count": len(prefix),
                "preserved_suffix_step_count": len(suffix),
                "boundary_source": boundary_source,
                "boundary_target": boundary_target,
            }
            patched["evidence_note"] = (
                "该路线由既有已执行路线做局部替换得到：选中区间之外的步骤保持不变，"
                "替换区间使用 Rhea 已知反应重新规划并排除原区间 Rhea 反应。整路基础分按与普通路线相同的公式重新计算；"
                "原路线的热力学/宿主通量附加层没有沿用，需要时应重新评估。"
            )
            patched_routes.append(patched)

        patched_routes.sort(
            key=lambda row: (
                -float(row.get("score") or 0.0),
                int((row.get("metrics") or {}).get("step_count") or len(row.get("steps") or [])),
                str(row.get("route_id") or ""),
            )
        )
        patched_routes = patched_routes[:replacement_count]
        for rank, patched in enumerate(patched_routes, start=1):
            patched["rank"] = rank
            patched["base_rank"] = rank

        parent_compound_names = list(route.get("compound_names") or [])
        final_target_id = str(parent_compound_ids[-1] if parent_compound_ids else boundary_target)
        final_target_name = str(parent_compound_names[-1] if parent_compound_names else final_target_id)
        return {
            "direction": "route_design",
            "answer_mode": "route_patch",
            "engine": "rhea_local_segment_patch_v1",
            "source_mode": "workspace_route_patch",
            "host": host,
            "priority": priority,
            "max_steps": whole_route_max_steps,
            "route_count": len(patched_routes),
            "requested_route_count": replacement_count,
            "routes": patched_routes,
            "exploratory_routes": [],
            "analysis_layers": [],
            "selected_target": {"chebi_id": final_target_id, "name": final_target_name},
            "graph_stats": dict(segment_search.get("graph_stats") or {}),
            "patch_context": {
                "parent_route_id": parent_route_id,
                "start_step_index": start_index,
                "end_step_index": end_index,
                "boundary_source": boundary_source,
                "boundary_target": boundary_target,
                "replaced_rhea_ids": replaced_rhea_ids,
                "excluded_rhea_ids": excluded_rhea_ids,
                "segment_step_budget": segment_step_budget,
                "preserved_prefix_step_count": len(prefix),
                "preserved_suffix_step_count": len(suffix),
            },
            "feasibility": {
                "preliminary_route_count": len(segment_search.get("routes") or []),
                "eligible_route_count": len(patched_routes),
                "returned_route_count": len(patched_routes),
                "requested_layers": [],
            },
            "score_note": (
                "局部替换后的整条路线基础分使用普通 route design 的同一公式重新计算；"
                "replacement_segment_score 只描述替换片段在局部搜索中的相对排序。"
            ),
            "session_id": str(session_id or ""),
            "ui_language": str(ui_language or "en"),
        }

    def _execute_prepared_pathway_analysis(
        self,
        *,
        resolution: dict[str, Any],
        user_text: str,
        session_id: str,
        ui_language: str,
    ) -> dict[str, Any]:
        pathway = (
            dict(resolution.get("pathway_resolution") or {})
            if isinstance(resolution.get("pathway_resolution"), dict)
            else {}
        )
        steps: list[dict[str, Any]] = []
        for step in pathway.get("steps") or []:
            if not isinstance(step, dict):
                continue
            reaction = (
                dict(step.get("reaction_resolution") or {})
                if isinstance(step.get("reaction_resolution"), dict)
                else {}
            )
            reaction_id = str(reaction.get("recommended_id") or "").strip()
            candidates = [row for row in reaction.get("candidates") or [] if isinstance(row, dict)]
            selected = next(
                (
                    row for row in candidates
                    if str(row.get("rhea_id") or row.get("id") or "").strip() == reaction_id
                ),
                candidates[0] if len(candidates) == 1 else {},
            )
            enzyme = (
                dict(step.get("enzyme_resolution") or {})
                if isinstance(step.get("enzyme_resolution"), dict)
                else {}
            )
            enzyme_id = str(enzyme.get("recommended_id") or "").strip() if bool(enzyme.get("specified")) else ""
            steps.append({
                "rhea_id": reaction_id,
                "orientation": str(selected.get("orientation") or "forward") or "forward",
                "equation": str(selected.get("equation") or reaction.get("interpreted_reaction") or ""),
                "enzyme_id": enzyme_id,
            })
        payload = {
            "steps": steps,
            "user_text": str(user_text or ""),
            "execution_mode": str(pathway.get("execution_mode") or "auto"),
            "host": str(pathway.get("host") or ""),
            "target_conditions": dict(pathway.get("target_conditions") or {}),
            "evidence_dimensions": list(pathway.get("evidence_dimensions") or []),
            "session_id": str(session_id or ""),
            "ui_language": str(ui_language or "en"),
        }
        return self.analyze_pathway(payload)


    def _prepare_seed_inputs(
        self,
        identifiers: list[str],
        sequence_inputs: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], Path | None, list[dict[str, Any]]]:
        return self.retrieval_service._prepare_seed_inputs(identifiers, sequence_inputs)


    def _validate_confirmed_positive_selection(
        self,
        *,
        session_id: str,
        direction: str,
        target_id: str,
        positive_ids: list[str] | None = None,
        positive_sequence_inputs: list[dict[str, Any]] | None = None,
        ui_language: str = "en",
    ) -> None:
        validation = self.agent_sessions.validate_pending_confirmation(
            session_id,
            direction=direction,
            target_id=target_id,
            positive_ids=positive_ids,
            positive_sequence_inputs=positive_sequence_inputs,
        )
        if bool(validation.get("valid")):
            return
        code = str(validation.get("error_code") or "confirmation_context_invalid")
        zh = str(ui_language or "").lower().startswith("zh")
        message = (
            "已确认正例必须来自当前服务器核对卡，请重新核对目标和正例后再执行。"
            if zh else
            "Confirmed positive seeds must come from the current server-verified card. Re-verify the target and positives before running."
        )
        safe_detail = {
            key: value for key, value in validation.items()
            if key in {"error_code", "allowed_target_count", "unknown_count", "candidate_id"}
        }
        raise AppError(code, message, HTTPStatus.UNPROCESSABLE_ENTITY, json.dumps(safe_detail, ensure_ascii=False))

    def rank_reactions(
        self,
        protein_id: str = "",
        *,
        enzyme_sequence: str = "",
        query_id: str = "",
        user_text: str = "",
        route_mode: str = "intelligent",
        observation_mode: str = "standard",
        target_conditions: dict[str, Any] | None = None,
        reaction_constraints: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
        confirmed_reaction_seed_ids: list[str] | None = None,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        confirmation_target = str(protein_id or query_id or "").strip()
        self._validate_confirmed_positive_selection(
            session_id=session_id,
            direction="enzyme_to_reaction",
            target_id=confirmation_target,
            positive_ids=confirmed_reaction_seed_ids,
            ui_language=ui_language,
        )
        result = self.retrieval_service.rank_reactions(
            protein_id,
            enzyme_sequence=enzyme_sequence,
            query_id=query_id,
            user_text=user_text,
            route_mode=route_mode,
            observation_mode=observation_mode,
            target_conditions=target_conditions,
            reaction_constraints=reaction_constraints,
            retrieval_plan=retrieval_plan,
            confirmed_reaction_seed_ids=confirmed_reaction_seed_ids,
            conversation_context=self.agent_sessions.execution_context(session_id, ui_language=ui_language),
            ui_language=ui_language,
        )
        self.agent_sessions.confirm_protein(
            session_id,
            protein_id=protein_id,
            sequence=enzyme_sequence,
            query_id=query_id,
        )
        self.agent_sessions.remember_execution_result(session_id, result, direction="enzyme_to_reaction")
        self.agent_sessions.consume_pending_confirmation(
            session_id, direction="enzyme_to_reaction", target_id=confirmation_target,
        )
        return result

    def rank_family_reactions(
        self,
        family_id: str,
        *,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        result = self.family_evidence.summarize(family_id, ui_language=ui_language)
        family = self.families.resolve(family_id)
        if family is not None:
            self.agent_sessions.confirm_protein_scope(session_id, {
                "kind": "family",
                "id": family.family_id,
                "family_id": family.family_id,
                "label": family.label,
            })
        return result


    def resolve(self, text: str) -> dict[str, Any]:
        return self.agent_resolution.resolve(text)


    def rank(
        self,
        rhea_id: str = "",
        *,
        reaction_smiles: str = "",
        query_id: str = "",
        orientation: str = "forward",
        user_text: str = "",
        route_mode: str = "intelligent",
        observation_mode: str = "standard",
        target_conditions: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
        top_k: int | None = None,
        confirmed_seed_ids: list[str] | None = None,
        confirmed_seed_inputs: list[dict[str, Any]] | None = None,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
        session_id: str = "",
    ) -> dict[str, Any]:
        confirmation_target = str(rhea_id or query_id or "").strip()
        self._validate_confirmed_positive_selection(
            session_id=session_id,
            direction="reaction_to_enzyme",
            target_id=confirmation_target,
            positive_ids=confirmed_seed_ids,
            positive_sequence_inputs=confirmed_seed_inputs,
            ui_language=ui_language,
        )
        result = self.retrieval_service.rank(
            rhea_id,
            reaction_smiles=reaction_smiles,
            query_id=query_id,
            orientation=orientation,
            user_text=user_text,
            route_mode=route_mode,
            observation_mode=observation_mode,
            target_conditions=target_conditions,
            retrieval_plan=retrieval_plan,
            top_k=top_k,
            confirmed_seed_ids=confirmed_seed_ids,
            confirmed_seed_inputs=confirmed_seed_inputs,
            conversation_context=self.agent_sessions.execution_context(session_id, ui_language=ui_language),
            ui_language=ui_language,
        )
        self.agent_sessions.confirm_reaction(
            session_id,
            reaction_id=rhea_id,
            reaction_smiles=reaction_smiles,
            query_id=query_id,
            orientation=orientation,
        )
        self.agent_sessions.remember_execution_result(session_id, result, direction="reaction_to_enzyme")
        self.agent_sessions.consume_pending_confirmation(
            session_id, direction="reaction_to_enzyme", target_id=confirmation_target,
        )
        return result





Handler.runtime = NavigatorRuntime()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the isolated Starase Navigator interface.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    args = parser.parse_args()
    if not STATIC_ROOT.is_dir():
        raise SystemExit(f"Static frontend not found: {STATIC_ROOT}")
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    server = ProductionHTTPServer((args.host, args.port), Handler)
    Handler.runtime.startup_prewarm_protein_encoder()
    print(json.dumps({"url": f"http://{args.host}:{args.port}/", **Handler.runtime.status()}, ensure_ascii=False, indent=2))
    stopping = threading.Event()

    def request_shutdown(_signum: int, _frame: Any) -> None:
        if stopping.is_set():
            return
        stopping.set()
        # socketserver.shutdown() must be called from a thread other than the one
        # currently running serve_forever().
        threading.Thread(target=server.shutdown, name="starase-navigator-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
