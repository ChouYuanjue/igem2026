from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _SessionState:
    updated_at: float
    verified_reaction_ids: list[str] = field(default_factory=list)
    verified_protein_ids: list[str] = field(default_factory=list)
    verified_family_ids: list[str] = field(default_factory=list)
    verified_protein_scopes: list[dict[str, Any]] = field(default_factory=list)
    verified_compound_ids: list[str] = field(default_factory=list)
    last_direction: str = ""
    last_target: str = ""
    last_result_mode: str = ""
    last_route_id: str = ""
    recent_evidence_ids: list[str] = field(default_factory=list)


class AgentSessionStore:
    """Short-lived trusted facts for conversational follow-ups.

    Only identifiers coming from deterministic resolvers/database results are stored.
    LLM summaries and free-form claims are intentionally excluded.
    """

    def __init__(self, *, ttl_seconds: int = 7200, max_sessions: int = 512) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_sessions = max(16, int(max_sessions))
        self._lock = threading.RLock()
        self._states: dict[str, _SessionState] = {}

    @staticmethod
    def _unique(values: list[str], limit: int = 24) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = str(value or "").strip()
            if text and text not in seen:
                seen.add(text)
                result.append(text)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _unique_scopes(values: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for value in values:
            if not isinstance(value, dict):
                continue
            kind = str(value.get("kind") or "").strip()
            scope_id = str(value.get("id") or value.get("family_id") or value.get("scope_id") or "").strip()
            if kind not in {"family", "functional_class"} or not scope_id:
                continue
            marker = (kind, scope_id)
            if marker in seen:
                continue
            seen.add(marker)
            result.append(deepcopy(value))
            if len(result) >= limit:
                break
        return result

    def _prune(self, now: float) -> None:
        expired = [key for key, value in self._states.items() if now - value.updated_at > self.ttl_seconds]
        for key in expired:
            self._states.pop(key, None)
        if len(self._states) <= self.max_sessions:
            return
        oldest = sorted(self._states.items(), key=lambda item: item[1].updated_at)
        for key, _value in oldest[: len(self._states) - self.max_sessions]:
            self._states.pop(key, None)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        key = str(session_id or "").strip()
        if not key:
            return {}
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._states.get(key)
            if state is None:
                return {}
            return deepcopy({
                "verified_reaction_ids": state.verified_reaction_ids,
                "verified_protein_ids": state.verified_protein_ids,
                "verified_family_ids": state.verified_family_ids,
                "verified_protein_scopes": state.verified_protein_scopes,
                "verified_compound_ids": state.verified_compound_ids,
                "last_direction": state.last_direction,
                "last_target": state.last_target,
                "last_result_mode": state.last_result_mode,
                "last_route_id": state.last_route_id,
                "recent_evidence_ids": state.recent_evidence_ids,
            })

    def remember_resolution(self, session_id: str, resolution: dict[str, Any]) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._states.get(key) or _SessionState(updated_at=now)
            state.updated_at = now
            direction = str(resolution.get("direction") or "").strip()
            if direction:
                state.last_direction = direction

            reaction_resolution = resolution.get("reaction_resolution") if isinstance(resolution.get("reaction_resolution"), dict) else {}
            reaction_ids = [str(row.get("rhea_id") or "") for row in reaction_resolution.get("candidates") or [] if isinstance(row, dict)]
            recommended_reaction = str(reaction_resolution.get("recommended_id") or "").strip()
            if recommended_reaction:
                reaction_ids.insert(0, recommended_reaction)

            protein_resolution = resolution.get("protein_resolution") if isinstance(resolution.get("protein_resolution"), dict) else {}
            mode = str(protein_resolution.get("mode") or "")
            recommended_protein = str(protein_resolution.get("recommended_id") or "").strip()
            if mode in {"protein_id", "natural_language", "general_merged_sequence_match"} and recommended_protein:
                state.verified_protein_ids = self._unique([recommended_protein] + state.verified_protein_ids)
            if mode in {"protein_family", "protein_functional_class"} and recommended_protein:
                state.verified_family_ids = self._unique([recommended_protein] + state.verified_family_ids)

            immediate = resolution.get("immediate_result") if isinstance(resolution.get("immediate_result"), dict) else {}
            family_payload = immediate.get("family") if isinstance(immediate.get("family"), dict) else {}
            if not family_payload and isinstance(protein_resolution.get("family"), dict):
                family_payload = dict(protein_resolution.get("family") or {})
            if mode in {"protein_family", "protein_functional_class"} and recommended_protein:
                label = str((immediate.get("protein") or {}).get("name") if isinstance(immediate.get("protein"), dict) else "").strip() or str(protein_resolution.get("interpreted_protein") or recommended_protein)
                if mode == "protein_family":
                    scope_snapshot = {
                        "kind": "family",
                        "id": recommended_protein,
                        "family_id": str(family_payload.get("family_id") or recommended_protein),
                        "label": label,
                    }
                else:
                    terms = [str(x).strip() for x in family_payload.get("normalized_terms") or [] if str(x).strip()]
                    strict = [str(x).strip() for x in family_payload.get("strict_terms") or [] if str(x).strip()]
                    broader = [str(x).strip() for x in family_payload.get("broader_terms") or [] if str(x).strip()]
                    scope_snapshot = {
                        "kind": "functional_class",
                        "id": recommended_protein,
                        "scope_id": str(family_payload.get("scope_id") or family_payload.get("family_id") or recommended_protein),
                        "label": label,
                        "enzyme_spec": {
                            "raw_text": label,
                            "protein_terms": terms or strict or [label],
                            "strict_terms": strict or terms,
                            "broader_terms": broader,
                            "organism_terms": [str(x).strip() for x in family_payload.get("organism_terms") or [] if str(x).strip()],
                            "gene_terms": [str(x).strip() for x in family_payload.get("gene_terms") or [] if str(x).strip()],
                            "accession_terms": [],
                            "scope_broadened": bool(family_payload.get("scope_broadened")),
                        },
                    }
                state.verified_protein_scopes = self._unique_scopes([scope_snapshot] + state.verified_protein_scopes)
            reaction = immediate.get("reaction") if isinstance(immediate.get("reaction"), dict) else {}
            if reaction.get("rhea_id"):
                reaction_ids.insert(0, str(reaction["rhea_id"]))
            protein = immediate.get("protein") if isinstance(immediate.get("protein"), dict) else {}
            if protein.get("id"):
                pid = str(protein["id"])
                if str(protein.get("input_mode") or "") in {"protein_family", "protein_functional_class"}:
                    state.verified_family_ids = self._unique([pid] + state.verified_family_ids)
                else:
                    state.verified_protein_ids = self._unique([pid] + state.verified_protein_ids)
            if str(immediate.get("answer_mode") or "") == "entity_list" and str(immediate.get("entity_kind") or "") == "compound":
                compound_ids = [
                    str(row.get("id") or "").strip()
                    for row in immediate.get("entities") or []
                    if isinstance(row, dict) and str(row.get("id") or "").strip().upper().startswith("CHEBI:")
                ]
                state.verified_compound_ids = self._unique(compound_ids + state.verified_compound_ids)
            known = immediate.get("known_associations") if isinstance(immediate.get("known_associations"), dict) else {}
            evidence_ids = [str(row.get("candidate_id") or "") for row in known.get("items") or [] if isinstance(row, dict)]
            state.recent_evidence_ids = self._unique(evidence_ids + state.recent_evidence_ids)
            state.verified_reaction_ids = self._unique(reaction_ids + state.verified_reaction_ids)

            ranking = immediate.get("ranking") if isinstance(immediate.get("ranking"), dict) else {}
            discovery = immediate.get("discovery_filter") if isinstance(immediate.get("discovery_filter"), dict) else {}
            if ranking.get("route_id"):
                state.last_route_id = str(ranking["route_id"])
            if discovery.get("result_mode"):
                state.last_result_mode = str(discovery["result_mode"])
            state.last_target = (
                str(reaction.get("rhea_id") or "")
                or str(protein.get("name") or protein.get("id") or "")
                or recommended_reaction
                or recommended_protein
                or state.last_target
            )
            self._states[key] = state
