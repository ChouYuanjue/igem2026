from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.candidate_universes import (  # noqa: E402
    DEFAULT_CANDIDATE_UNIVERSE,
)
from scripts.database_bridge.model_catalog import ModelDataCatalog  # noqa: E402
from scripts.catalyst_finder.agent_resolution_service import AgentResolutionService  # noqa: E402
from scripts.catalyst_finder.evidence_catalog import IntegratedEvidenceCatalog  # noqa: E402
from scripts.catalyst_finder.e2r_routing_graph import E2RRoutePlanner  # noqa: E402
from scripts.catalyst_finder.homology import ProteinHomologyIndex  # noqa: E402
from scripts.catalyst_finder.http_transport import Handler  # noqa: E402
from scripts.catalyst_finder.legacy_intent_compat import (  # noqa: E402,F401
    FOLLOWUP_ENZYME_ONLY_RE,
    FOLLOWUP_REACTION_ONLY_RE,
    PATHWAY_ARROW_RE,
    PATHWAY_INTENT_RE,
    ROUTE_DESIGN_INTENT_RE,
    ROUTE_ROLE_PAIR_RE,
    SINGLE_REACTION_INTENT_RE,
    classify_task_intent,
)
from scripts.catalyst_finder.language_resolver import DeepSeekResolver  # noqa: E402
from scripts.catalyst_finder.model_gateway import ModelGateway  # noqa: E402
from scripts.catalyst_finder.open_world_inputs import ProteinSequenceInput  # noqa: E402
from scripts.catalyst_finder.protein_resolution import ProteinResolver  # noqa: E402
from scripts.catalyst_finder.runtime_store import RuntimeStore  # noqa: E402
from scripts.catalyst_finder.pathway_compatibility import PathwayCompatibilityAnalyzer  # noqa: E402
from scripts.catalyst_finder.retrieval_service import RetrievalApplicationService  # noqa: E402
from scripts.catalyst_finder.rhea_client import RheaClient, canonical_rhea_id  # noqa: E402,F401
from scripts.catalyst_finder.route_design import RheaRouteDesigner  # noqa: E402
from scripts.catalyst_finder.resolution_helpers import (  # noqa: E402,F401
    candidate_match as _candidate_match,
    explicit_uniprot_accession as _explicit_uniprot_accession,
    fallback_queries as _fallback_queries,
)
from scripts.catalyst_finder.route_feasibility import RouteFeasibilityAnalyzer  # noqa: E402
from scripts.catalyst_finder.route_pathway_service import RoutePathwayService  # noqa: E402
from scripts.catalyst_finder.route_view import system_route_catalog  # noqa: E402
from scripts.catalyst_finder.routing_graph import RoutePlanner  # noqa: E402

STATIC_ROOT = ROOT / "frontend/catalyst_finder"
RUNTIME_ROOT = ROOT / "results/catalyst_finder_runtime"
CACHE_ROOT = RUNTIME_ROOT / "cache"
FEEDBACK_PATH = RUNTIME_ROOT / "feedback.jsonl"
RUN_EVENTS_PATH = RUNTIME_ROOT / "run_events.jsonl"

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
USER_AGENT = "NJU-iGEM-2026-CatalystFinder/1.0"


def _build_revision() -> str:
    configured = str(os.environ.get("CATALYST_FINDER_BUILD_REVISION") or "").strip()
    if configured:
        return configured[:40]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


class ProductionHTTPServer(ThreadingHTTPServer):
    """Small production wrapper around the standard threaded HTTP server."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

class CatalystFinderRuntime:
    def __init__(self) -> None:
        self.started_at_unix = time.time()
        self.build_revision = _build_revision()
        self.catalog = ModelDataCatalog(ROOT)
        self.evidence = IntegratedEvidenceCatalog(ROOT)
        self.rhea = RheaClient(CACHE_ROOT)
        self.deepseek = DeepSeekResolver()
        self.proteins = ProteinResolver(self.catalog, user_agent=USER_AGENT)
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
        )
        self.agent_resolution = AgentResolutionService(
            catalog=self.catalog,
            evidence=self.evidence,
            rhea=self.rhea,
            deepseek=self.deepseek,
            proteins=self.proteins,
            route_design_resolve=self.route_pathway.route_design_resolve,
            pathway_resolve=self.route_pathway.pathway_resolve,
        )
        self.model_gateway = ModelGateway()
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

    def status(self) -> dict[str, Any]:
        project_summary = self.catalog.summary()
        evidence_summary = self.evidence.summary()
        return {
            "status": "ready",
            "service": "catalyst_finder",
            "build_revision": self.build_revision,
            "process_id": os.getpid(),
            "uptime_seconds": round(max(0.0, time.time() - self.started_at_unix), 2),
            "deepseek_configured": self.deepseek.configured,
            "deepseek_model": os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            "deepseek": self.deepseek.provenance(),
            "route_planner": "langgraph",
            "agent_directions": ["reaction_to_enzyme", "enzyme_to_reaction", "route_design", "pathway_compatibility"],
            "natural_language_resolution": ["reaction", "protein", "positive_enzyme"],
            "default_route": {"top_k": 10, "enzyme_taxonomy_scope": "all", "shot_mode": "zero_shot", "homology_policy": "allow", "known_association_policy": "allow_known"},
            "result_scopes": ["allow_known", "known_only", "exclude_known"],
            "homology_definition": "MMseqs2 50% sequence identity, >=80% coverage",
            "homology_index_cached": self.homology.ready,
            "route_catalog": self._route_catalog["counts"],
            "candidate_universe": DEFAULT_CANDIDATE_UNIVERSE,
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
            "feedback_enabled": True,
            "route_feasibility": self.route_feasibility.status(),
        }

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


    def _direct_open_world_resolution(
        self,
        text: str,
        direction_hint: str,
        ui_language: str,
    ) -> dict[str, Any] | None:
        return self.agent_resolution._direct_open_world_resolution(
            text, direction_hint, ui_language
        )


    def agent_resolve(
        self,
        text: str,
        direction_hint: str = "auto",
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        return self.agent_resolution.agent_resolve(
            text,
            direction_hint=direction_hint,
            conversation_context=conversation_context,
            ui_language=ui_language,
            resolve_reaction=self.resolve,
        )


    def _prepare_seed_inputs(
        self,
        identifiers: list[str],
        sequence_inputs: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], Path | None, list[dict[str, Any]]]:
        return self.retrieval_service._prepare_seed_inputs(identifiers, sequence_inputs)


    def rank_reactions(
        self,
        protein_id: str = "",
        *,
        enzyme_sequence: str = "",
        query_id: str = "",
        user_text: str = "",
        route_mode: str = "intelligent",
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        return self.retrieval_service.rank_reactions(
            protein_id,
            enzyme_sequence=enzyme_sequence,
            query_id=query_id,
            user_text=user_text,
            route_mode=route_mode,
            conversation_context=conversation_context,
            ui_language=ui_language,
        )


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
        top_k: int | None = None,
        confirmed_seed_ids: list[str] | None = None,
        confirmed_seed_inputs: list[dict[str, Any]] | None = None,
        conversation_context: dict[str, Any] | None = None,
        ui_language: str = "en",
    ) -> dict[str, Any]:
        return self.retrieval_service.rank(
            rhea_id,
            reaction_smiles=reaction_smiles,
            query_id=query_id,
            orientation=orientation,
            user_text=user_text,
            route_mode=route_mode,
            top_k=top_k,
            confirmed_seed_ids=confirmed_seed_ids,
            confirmed_seed_inputs=confirmed_seed_inputs,
            conversation_context=conversation_context,
            ui_language=ui_language,
        )





Handler.runtime = CatalystFinderRuntime()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the isolated Catalyst Finder interface.")
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
        threading.Thread(target=server.shutdown, name="catalyst-shutdown", daemon=True).start()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
