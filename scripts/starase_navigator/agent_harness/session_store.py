from __future__ import annotations

import hashlib
import threading
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


_ENTITY_KINDS = {"reaction", "protein", "protein_scope", "compound", "literature", "route", "route_step"}
_ROLE_PRIORITY = {"model_candidate": 0, "related_evidence": 1, "resolved_target": 2, "confirmed_target": 3}
_NON_FOCUS_ROLES = {"model_candidate", "related_evidence"}


@dataclass
class _SessionState:
    updated_at: float
    # Legacy identity lists remain in snapshots for compatibility, but they now contain
    # only actual query targets/scopes. Related evidence rows are never promoted here.
    verified_reaction_ids: list[str] = field(default_factory=list)
    verified_protein_ids: list[str] = field(default_factory=list)
    verified_family_ids: list[str] = field(default_factory=list)
    verified_protein_scopes: list[dict[str, Any]] = field(default_factory=list)
    verified_compound_ids: list[str] = field(default_factory=list)
    # Canonical reusable entities. Payloads are kept server-side and are not sent to the
    # controller model by model_snapshot().
    entities: list[dict[str, Any]] = field(default_factory=list)
    active_entity_keys: dict[str, str] = field(default_factory=dict)
    last_direction: str = ""
    last_target: str = ""
    last_result_mode: str = ""
    last_association_policy: str = ""
    last_route_id: str = ""
    recent_evidence_ids: list[str] = field(default_factory=list)
    # Natural chronological conversation history. Provenance is carried by the role
    # (user/assistant); scientific workspace state remains separately source-tagged.
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    # Compatibility alias retained temporarily for older callers/tests. New controller
    # code consumes conversation_history directly.
    recent_dialogue: list[dict[str, Any]] = field(default_factory=list)
    # Compact summary of the last successful structured result. This contains only
    # server-produced result fields safe for follow-up reference.
    last_result_context: dict[str, Any] = field(default_factory=dict)
    # Recent server-executed artifacts, kept append-only so a later turn can compare
    # more than just the immediately previous ranking/result.
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    # Current client-visible slice of an already verified paginated result. This is view
    # state only; it never creates trusted entities or changes conversational focus.
    visible_entity_keys: list[str] = field(default_factory=list)
    visible_entity_kind: str = ""
    visible_page_index: int = 0
    # One verification card worth of server-verified selectable targets/positive anchors.
    # This is intentionally server-only and is never exposed to the controller model.
    pending_confirmation: dict[str, Any] = field(default_factory=dict)


class AgentSessionStore:
    """Short-lived trusted facts for conversational follow-ups.

    The store distinguishes three concepts that must not be conflated:
    - a query target resolved in a prior turn;
    - a target the user actually confirmed before execution;
    - entities merely shown as related database evidence.

    Only deterministic resolver/database output or a user's confirmed selection enters
    this store. Free-form model claims never become session facts.
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

    @staticmethod
    def _entity_key(kind: str, entity_id: str) -> str:
        return f"{kind}:{entity_id}"

    @classmethod
    def _upsert_entity(
        cls,
        state: _SessionState,
        entity: dict[str, Any],
        *,
        activate: bool = False,
        limit: int = 40,
    ) -> None:
        kind = str(entity.get("kind") or "").strip()
        entity_id = str(entity.get("id") or "").strip()
        if kind not in _ENTITY_KINDS or not entity_id:
            return
        item = deepcopy(entity)
        item["kind"] = kind
        item["id"] = entity_id
        item["label"] = str(item.get("label") or entity_id).strip() or entity_id
        role = str(item.get("role") or "resolved_target").strip()
        if role not in _ROLE_PRIORITY:
            role = "resolved_target"
        item["role"] = role
        key = cls._entity_key(kind, entity_id)

        def literature_aliases(row: dict[str, Any]) -> set[str]:
            if str(row.get("kind") or "") != "literature":
                return set()
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            aliases = {f"id:{str(row.get('id') or '').strip().casefold()}"}
            pmid = str(payload.get("pmid") or "").strip()
            doi = str(payload.get("doi") or "").strip().casefold()
            pmcid = str(payload.get("pmcid") or "").strip().upper()
            if pmid:
                aliases.add(f"pmid:{pmid}")
            if doi:
                aliases.add(f"doi:{doi}")
            if pmcid:
                aliases.add(f"pmcid:{pmcid}")
            return aliases

        existing = next((row for row in state.entities if cls._entity_key(str(row.get("kind") or ""), str(row.get("id") or "")) == key), None)
        if existing is None and kind == "literature":
            aliases = literature_aliases(item)
            existing = next((
                row for row in state.entities
                if str(row.get("kind") or "") == "literature" and aliases.intersection(literature_aliases(row))
            ), None)
        old_key = ""
        if existing is not None:
            old_id = str(existing.get("id") or "").strip()
            old_key = cls._entity_key(kind, old_id)
            old_role = str(existing.get("role") or "related_evidence")
            if _ROLE_PRIORITY.get(old_role, 0) > _ROLE_PRIORITY.get(role, 0):
                item["role"] = old_role
            old_payload = existing.get("payload") if isinstance(existing.get("payload"), dict) else {}
            new_payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            merged_payload = deepcopy(old_payload)
            for payload_key, payload_value in new_payload.items():
                if payload_value not in (None, "", [], {}):
                    merged_payload[payload_key] = deepcopy(payload_value)
            providers = []
            for source_payload in (old_payload, new_payload):
                for provider in list(source_payload.get("evidence_providers") or []) + [source_payload.get("provider")]:
                    provider = str(provider or "").strip()
                    if provider and provider not in providers:
                        providers.append(provider)
            if providers:
                merged_payload["evidence_providers"] = providers
            item["payload"] = merged_payload
            if kind == "literature":
                def identity_priority(value: str) -> int:
                    upper = value.upper()
                    return 4 if upper.startswith("MED:") else 3 if upper.startswith("PMC") else 2 if upper.startswith("DOI:") else 1
                if identity_priority(old_id) > identity_priority(entity_id):
                    item["id"] = old_id
                    entity_id = old_id
                    key = old_key
        remove_keys = {key}
        if old_key:
            remove_keys.add(old_key)
        state.entities = [
            row for row in state.entities
            if cls._entity_key(str(row.get("kind") or ""), str(row.get("id") or "")) not in remove_keys
        ]
        state.entities.insert(0, item)
        del state.entities[limit:]
        if old_key and old_key != key:
            state.visible_entity_keys = [key if value == old_key else value for value in state.visible_entity_keys]
            for active_kind, active_key in list(state.active_entity_keys.items()):
                if active_key == old_key:
                    state.active_entity_keys[active_kind] = key
        if activate:
            state.active_entity_keys[kind] = key

    @staticmethod
    def _sequence_digest(sequence: str) -> str:
        normalized = re.sub(r"\s+", "", str(sequence or "")).upper()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""

    @staticmethod
    def _candidate_id(row: dict[str, Any]) -> str:
        for key in ("id", "rhea_id", "accession", "query_id", "recommended_id"):
            value = str(row.get(key) or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def _resolution_selectable_ids(cls, resolution: dict[str, Any]) -> list[str]:
        values: list[str] = []
        recommended = str(resolution.get("recommended_id") or "").strip()
        if recommended:
            values.append(recommended)
        for row in resolution.get("candidates") or []:
            if not isinstance(row, dict):
                continue
            value = cls._candidate_id(row)
            if value:
                values.append(value)
        return cls._unique(values, limit=40)

    @classmethod
    def _pending_confirmation_from_resolution(cls, resolution: dict[str, Any]) -> dict[str, Any] | None:
        if "positive_enzyme_resolutions" not in resolution and "positive_reaction_resolutions" not in resolution:
            return None
        direction = str(resolution.get("direction") or "").strip()
        if direction not in {"reaction_to_enzyme", "enzyme_to_reaction"}:
            return None
        reaction = resolution.get("reaction_resolution") if isinstance(resolution.get("reaction_resolution"), dict) else {}
        protein = resolution.get("protein_resolution") if isinstance(resolution.get("protein_resolution"), dict) else {}
        target_resolution = reaction if direction == "reaction_to_enzyme" else protein
        target_ids = cls._resolution_selectable_ids(target_resolution)

        positive_enzyme_ids: list[str] = []
        positive_enzyme_sequence_digests: dict[str, str] = {}
        for group in resolution.get("positive_enzyme_resolutions") or []:
            if not isinstance(group, dict):
                continue
            rows = [row for row in group.get("candidates") or [] if isinstance(row, dict)]
            for row in rows:
                candidate_id = cls._candidate_id(row)
                if not candidate_id:
                    continue
                positive_enzyme_ids.append(candidate_id)
                digest = cls._sequence_digest(str(row.get("sequence") or ""))
                if digest:
                    positive_enzyme_sequence_digests[candidate_id] = digest

        positive_reaction_ids: list[str] = []
        for group in resolution.get("positive_reaction_resolutions") or []:
            if not isinstance(group, dict):
                continue
            for row in group.get("candidates") or []:
                if not isinstance(row, dict):
                    continue
                candidate_id = cls._candidate_id(row)
                if candidate_id:
                    positive_reaction_ids.append(candidate_id)

        return {
            "direction": direction,
            "target_ids": cls._unique(target_ids, limit=40),
            "positive_enzyme_ids": cls._unique(positive_enzyme_ids, limit=40),
            "positive_enzyme_sequence_digests": positive_enzyme_sequence_digests,
            "positive_reaction_ids": cls._unique(positive_reaction_ids, limit=40),
        }

    @staticmethod
    def _candidate_for_id(resolution: dict[str, Any], target_id: str) -> dict[str, Any]:
        candidates = [row for row in resolution.get("candidates") or [] if isinstance(row, dict)]
        id_keys = ("id", "rhea_id", "chebi_id", "query_id")
        for row in candidates:
            if any(str(row.get(key) or "").strip() == target_id for key in id_keys):
                return deepcopy(row)
        if len(candidates) == 1:
            return deepcopy(candidates[0])
        return {}

    @classmethod
    def _reaction_entity(cls, resolution: dict[str, Any], *, role: str) -> dict[str, Any] | None:
        target_id = str(resolution.get("recommended_id") or "").strip()
        if not target_id:
            return None
        candidate = cls._candidate_for_id(resolution, target_id)
        interpreted = str(resolution.get("interpreted_reaction") or "").strip()
        payload = {
            "mode": str(resolution.get("mode") or "session_verified_reaction"),
            "interpreted_reaction": interpreted or target_id,
            "assumptions": [],
            "normalized": deepcopy(resolution.get("normalized") or {}),
            "candidates": [candidate] if candidate else [],
            "recommended_id": target_id,
        }
        for key in ("matched_reaction_ids", "reaction_smiles"):
            if resolution.get(key):
                payload[key] = deepcopy(resolution[key])
        if candidate.get("reaction_smiles") and not payload.get("reaction_smiles"):
            payload["reaction_smiles"] = str(candidate["reaction_smiles"])
        label = str(candidate.get("equation") or interpreted or target_id)
        return {"kind": "reaction", "id": target_id, "label": label, "role": role, "payload": payload}

    @classmethod
    def _protein_entity(cls, resolution: dict[str, Any], *, role: str) -> dict[str, Any] | None:
        target_id = str(resolution.get("recommended_id") or "").strip()
        if not target_id:
            return None
        candidate = cls._candidate_for_id(resolution, target_id)
        interpreted = str(resolution.get("interpreted_protein") or "").strip()
        if not candidate:
            candidate = {
                "id": target_id,
                "name": interpreted or target_id,
                "input_mode": str(resolution.get("mode") or "protein_id"),
            }
        payload = {
            "mode": str(resolution.get("mode") or candidate.get("input_mode") or "session_verified_protein"),
            "interpreted_protein": interpreted or str(candidate.get("name") or target_id),
            "assumptions": [],
            "normalized": deepcopy(resolution.get("normalized") or {}),
            "candidates": [deepcopy(candidate)],
            "recommended_id": target_id,
        }
        label = str(candidate.get("name") or interpreted or target_id)
        entity = {"kind": "protein", "id": target_id, "label": label, "role": role, "payload": payload}
        if candidate.get("organism"):
            entity["subtitle"] = str(candidate["organism"])
        return entity

    @classmethod
    def _model_candidate_entity(
        cls,
        row: dict[str, Any],
        *,
        direction: str,
        target: dict[str, Any],
    ) -> dict[str, Any] | None:
        candidate_id = str(row.get("candidate_id") or row.get("id") or "").strip()
        if not candidate_id:
            return None
        rank = row.get("rank")
        hypothesis = {
            "source": "executed_model_candidate",
            "direction": str(direction or ""),
            "rank": rank,
            "model_support_index": row.get("model_support_index"),
            "correspondence_defect": row.get("correspondence_defect"),
            "target": deepcopy(target),
            "association_status": "model_hypothesis_not_verified_relation",
        }
        if direction == "enzyme_to_reaction":
            label = str(
                row.get("equation")
                or row.get("name")
                or (
                    f"{row.get('substrate_name')} → {row.get('product_name')}"
                    if row.get("substrate_name") or row.get("product_name") else ""
                )
                or candidate_id
            ).strip()
            candidate = deepcopy(row)
            if candidate_id.startswith("RHEA:"):
                candidate.setdefault("rhea_id", candidate_id)
            candidate.setdefault("equation", label)
            payload = {
                "mode": "executed_model_candidate_reaction",
                "interpreted_reaction": label,
                "assumptions": [],
                "normalized": {},
                "candidates": [candidate],
                "recommended_id": candidate_id,
                "model_candidate_context": hypothesis,
            }
            return {
                "kind": "reaction",
                "id": candidate_id,
                "label": label,
                "role": "model_candidate",
                "payload": payload,
                "source": "executed_model_candidate",
                "hypothesis": True,
                "rank": rank,
                "target_id": str(target.get("id") or ""),
            }
        if direction == "reaction_to_enzyme":
            label = str(row.get("name") or candidate_id).strip()
            candidate = deepcopy(row)
            candidate.setdefault("id", candidate_id)
            payload = {
                "mode": "executed_model_candidate_protein",
                "interpreted_protein": label,
                "assumptions": [],
                "normalized": {},
                "candidates": [candidate],
                "recommended_id": candidate_id,
                "model_candidate_context": hypothesis,
            }
            entity = {
                "kind": "protein",
                "id": candidate_id,
                "label": label,
                "role": "model_candidate",
                "payload": payload,
                "source": "executed_model_candidate",
                "hypothesis": True,
                "rank": rank,
                "target_id": str(target.get("id") or ""),
            }
            if row.get("organism"):
                entity["subtitle"] = str(row.get("organism") or "")
            return entity
        return None

    @staticmethod
    def _scope_entity(scope: dict[str, Any], *, role: str) -> dict[str, Any] | None:
        kind = str(scope.get("kind") or "").strip()
        scope_id = str(scope.get("id") or scope.get("family_id") or scope.get("scope_id") or "").strip()
        if kind not in {"family", "functional_class"} or not scope_id:
            return None
        return {
            "kind": "protein_scope",
            "id": scope_id,
            "label": str(scope.get("label") or scope_id),
            "role": role,
            "payload": deepcopy(scope),
            "scope_kind": kind,
        }

    @staticmethod
    def _literature_entity(row: dict[str, Any], *, role: str) -> dict[str, Any] | None:
        source = str(row.get("source") or "").strip().upper()
        pmid = str(row.get("pmid") or "").strip()
        pmcid = str(row.get("pmcid") or "").strip().upper()
        doi = str(row.get("doi") or "").strip()
        raw_id = str(row.get("id") or "").strip()
        if pmid:
            entity_id = f"MED:{pmid}"
        elif pmcid:
            entity_id = pmcid if pmcid.startswith("PMC") else f"PMC:{pmcid}"
        elif doi:
            entity_id = f"DOI:{doi}"
        elif re.match(r"^[A-Za-z][A-Za-z0-9_-]*:", raw_id):
            entity_id = raw_id
        elif raw_id:
            stable_source = source if source in {"MED", "PMC"} else ""
            entity_id = f"{stable_source}:{raw_id}" if stable_source else raw_id
        else:
            return None
        title = str(row.get("title") or entity_id).strip() or entity_id
        subtitle = " · ".join(str(x).strip() for x in [row.get("authors"), row.get("journal"), row.get("year")] if str(x or "").strip())
        return {
            "kind": "literature",
            "id": entity_id,
            "label": title,
            "subtitle": subtitle,
            "role": role,
            "payload": deepcopy(row),
        }

    @staticmethod
    def _compound_entity(row: dict[str, Any], *, role: str) -> dict[str, Any] | None:
        cid = str(row.get("chebi_id") or row.get("id") or "").strip().upper()
        if not cid.startswith("CHEBI:"):
            return None
        payload = deepcopy(row)
        payload["chebi_id"] = cid
        return {
            "kind": "compound",
            "id": cid,
            "label": str(row.get("name") or cid),
            "role": role,
            "payload": payload,
        }

    @staticmethod
    def _route_entity(
        route: dict[str, Any],
        *,
        role: str,
        result_section: str = "routes",
    ) -> dict[str, Any] | None:
        route_id = str(route.get("route_id") or "").strip()
        if not route_id:
            return None
        names = [str(value).strip() for value in route.get("compound_names") or [] if str(value).strip()]
        label = " → ".join(names) if names else route_id
        payload = deepcopy(route)
        payload["result_section"] = str(result_section or "routes")
        patch = route.get("patch") if isinstance(route.get("patch"), dict) else {}
        parent_route_id = str(route.get("parent_route_id") or "").strip()
        root_route_id = str(route.get("root_route_id") or "").strip() or (parent_route_id if parent_route_id else route_id)
        return {
            "kind": "route",
            "id": route_id,
            "label": label,
            "role": role,
            "payload": payload,
            "rank": route.get("rank") or route.get("base_rank"),
            "route_type": str(route.get("route_type") or ""),
            "parent_route_id": parent_route_id,
            "root_route_id": root_route_id,
            "generation": int(route.get("generation") or (1 if parent_route_id else 0)),
            "patch_start_step_index": patch.get("start_step_index"),
            "patch_end_step_index": patch.get("end_step_index"),
        }

    @staticmethod
    def _route_step_entity(
        route: dict[str, Any],
        step: dict[str, Any],
        *,
        role: str,
        route_rank: int | None = None,
    ) -> dict[str, Any] | None:
        route_id = str(route.get("route_id") or "").strip()
        step_index = int(step.get("step_index") or 0)
        if not route_id or step_index <= 0:
            return None
        step_id = f"{route_id}::step:{step_index}"
        rhea_id = str(step.get("rhea_id") or "").strip()
        source_name = str(step.get("source_name") or step.get("source") or "").strip()
        target_name = str(step.get("target_name") or step.get("target") or "").strip()
        label = f"{source_name} → {target_name}".strip(" →") or rhea_id or step_id
        payload = {
            "route_id": route_id,
            "route_rank": route_rank,
            "step_index": step_index,
            "step": deepcopy(step),
            "route_compound_ids": list(route.get("compound_ids") or []),
            "route_compound_names": list(route.get("compound_names") or []),
        }
        return {
            "kind": "route_step",
            "id": step_id,
            "label": label,
            "role": role,
            "payload": payload,
            "parent_route_id": route_id,
            "route_rank": route_rank,
            "step_index": step_index,
            "reaction_id": rhea_id,
            "source_compound_id": str(step.get("source") or ""),
            "target_compound_id": str(step.get("target") or ""),
        }

    @classmethod
    def _listed_entity(
        cls,
        row: dict[str, Any],
        *,
        entity_kind: str,
        role: str,
    ) -> dict[str, Any] | None:
        kind = str(entity_kind or "").strip().lower()
        if kind == "literature":
            return cls._literature_entity(row, role=role)
        if kind == "compound":
            return cls._compound_entity(row, role=role)
        if kind == "protein":
            entity_id = str(row.get("id") or row.get("candidate_id") or row.get("accession") or "").strip()
            if not entity_id:
                return None
            label = str(row.get("name") or entity_id).strip() or entity_id
            candidate = deepcopy(row)
            candidate.setdefault("id", entity_id)
            return {
                "kind": "protein",
                "id": entity_id,
                "label": label,
                "subtitle": str(row.get("organism") or row.get("subtitle") or ""),
                "role": role,
                "payload": {
                    "mode": "session_listed_protein",
                    "interpreted_protein": label,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": entity_id,
                },
            }
        if kind == "reaction":
            entity_id = str(row.get("rhea_id") or row.get("id") or row.get("candidate_id") or "").strip()
            if not entity_id:
                return None
            label = str(
                row.get("equation")
                or row.get("name")
                or (
                    f"{row.get('substrate_name')} → {row.get('product_name')}"
                    if row.get("substrate_name") or row.get("product_name") else ""
                )
                or entity_id
            ).strip()
            candidate = deepcopy(row)
            if entity_id.startswith("RHEA:"):
                candidate.setdefault("rhea_id", entity_id)
            return {
                "kind": "reaction",
                "id": entity_id,
                "label": label,
                "role": role,
                "payload": {
                    "mode": "session_listed_reaction",
                    "interpreted_reaction": label,
                    "assumptions": [],
                    "normalized": {},
                    "candidates": [candidate],
                    "recommended_id": entity_id,
                },
            }
        if kind == "route":
            route_id = str(row.get("id") or row.get("route_id") or "").strip()
            if not route_id:
                return None
            return {
                "kind": "route",
                "id": route_id,
                "label": str(row.get("name") or route_id),
                "role": role,
                "payload": deepcopy(row),
                "rank": row.get("rank"),
            }
        if kind == "route_step":
            step_id = str(row.get("id") or "").strip()
            if not step_id:
                return None
            return {
                "kind": "route_step",
                "id": step_id,
                "label": str(row.get("name") or step_id),
                "role": role,
                "payload": deepcopy(row),
                "parent_route_id": str(row.get("route_id") or ""),
                "route_rank": row.get("route_rank"),
                "step_index": row.get("step_index"),
                "reaction_id": str(row.get("rhea_id") or ""),
                "source_compound_id": str(row.get("source_compound_id") or ""),
                "target_compound_id": str(row.get("target_compound_id") or ""),
            }
        return None

    def _prune(self, now: float) -> None:
        expired = [key for key, value in self._states.items() if now - value.updated_at > self.ttl_seconds]
        for key in expired:
            self._states.pop(key, None)
        if len(self._states) <= self.max_sessions:
            return
        oldest = sorted(self._states.items(), key=lambda item: item[1].updated_at)
        for key, _value in oldest[: len(self._states) - self.max_sessions]:
            self._states.pop(key, None)

    @classmethod
    def _session_entities_snapshot(cls, state: _SessionState, *, include_payload: bool) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        active_keys = set(state.active_entity_keys.values())
        focus_keys: set[str] = set()
        focused_kinds: set[str] = set()
        for raw in state.entities:
            kind = str(raw.get("kind") or "")
            role = str(raw.get("role") or "")
            if kind and kind not in focused_kinds and role not in _NON_FOCUS_ROLES:
                focus_keys.add(cls._entity_key(kind, str(raw.get("id") or "")))
                focused_kinds.add(kind)
        visible_positions = {key: index + 1 for index, key in enumerate(state.visible_entity_keys)}
        for index, raw in enumerate(state.entities):
            row = deepcopy(raw)
            key = cls._entity_key(str(row.get("kind") or ""), str(row.get("id") or ""))
            row["active"] = key in active_keys
            row["focus"] = key in focus_keys
            row["recency_index"] = index
            if key in visible_positions:
                row["visible"] = True
                row["visible_index"] = visible_positions[key]
                row["visible_page_index"] = state.visible_page_index
            else:
                row["visible"] = False
            if not include_payload:
                row.pop("payload", None)
            rows.append(row)
        return {
            "focus": [row for row in rows if row.get("focus")],
            "active": [row for row in rows if row.get("active")],
            "visible": sorted([row for row in rows if row.get("visible")], key=lambda row: int(row.get("visible_index") or 10**6)),
            "history": [row for row in rows if str(row.get("role") or "") not in _NON_FOCUS_ROLES],
            "related": [row for row in rows if str(row.get("role") or "") == "related_evidence"],
            "candidates": [row for row in rows if str(row.get("role") or "") == "model_candidate"],
            "all": rows,
            "reuse_rule": (
                "These are server-verified workspace entities from prior turns. The harness mounts them "
                "as current-run refs before planning. focus marks recent conversational focus; active "
                "marks the last explicitly confirmed/executed target; visible marks the current UI page. "
                "The latest user instruction still determines which handle, if any, is relevant."
            ),
        }

    def snapshot(self, session_id: str) -> dict[str, Any]:
        """Return server-side session facts, including canonical reusable payloads."""
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
                "last_association_policy": state.last_association_policy,
                "last_route_id": state.last_route_id,
                "recent_evidence_ids": state.recent_evidence_ids,
                "conversation_history": state.conversation_history,
                "recent_dialogue": state.recent_dialogue,
                "last_result_context": state.last_result_context,
                "execution_history": state.execution_history,
                "session_entities": self._session_entities_snapshot(state, include_payload=True),
            })

    def model_snapshot(self, session_id: str) -> dict[str, Any]:
        """Return compact facts safe to send to the controller (no sequences/payload blobs)."""
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
                "last_direction": state.last_direction,
                "last_target": state.last_target,
                "last_result_mode": state.last_result_mode,
                "last_association_policy": state.last_association_policy,
                "last_route_id": state.last_route_id,
                "conversation_history": state.conversation_history,
                "last_result_context": state.last_result_context,
                "execution_history": state.execution_history,
                "session_entities": self._session_entities_snapshot(state, include_payload=False),
            })

    def _state(self, key: str, now: float) -> _SessionState:
        state = self._states.get(key) or _SessionState(updated_at=now)
        state.updated_at = now
        return state

    def remember_resolution(self, session_id: str, resolution: dict[str, Any]) -> None:
        """Remember verified outputs without treating related evidence as user-selected targets."""
        key = str(session_id or "").strip()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            direction = str(resolution.get("direction") or "").strip()
            if direction:
                state.last_direction = direction

            pending_confirmation = self._pending_confirmation_from_resolution(resolution)
            if pending_confirmation is not None:
                state.pending_confirmation = pending_confirmation

            immediate = resolution.get("immediate_result") if isinstance(resolution.get("immediate_result"), dict) else {}
            # Resolved/inspected evidence changes the conversational focus only.
            # The active target is reserved for an explicit user-confirmed execution
            # and is updated exclusively by confirm_reaction/confirm_protein/confirm_protein_scope.
            completed_without_confirmation = False
            reaction_resolution = resolution.get("reaction_resolution") if isinstance(resolution.get("reaction_resolution"), dict) else {}
            protein_resolution = resolution.get("protein_resolution") if isinstance(resolution.get("protein_resolution"), dict) else {}

            primary_reaction = self._reaction_entity(reaction_resolution, role="resolved_target") if reaction_resolution else None
            if primary_reaction:
                self._upsert_entity(state, primary_reaction, activate=completed_without_confirmation)
                rid = primary_reaction["id"]
                state.verified_reaction_ids = self._unique([rid] + state.verified_reaction_ids)

            mode = str(protein_resolution.get("mode") or "")
            primary_protein: dict[str, Any] | None = None
            if mode in {"protein_id", "natural_language", "general_merged_sequence_match", "session_verified_protein", "raw_protein_sequence"}:
                primary_protein = self._protein_entity(protein_resolution, role="resolved_target")
                if mode == "raw_protein_sequence" and primary_protein:
                    payload = primary_protein.get("payload") if isinstance(primary_protein.get("payload"), dict) else {}
                    candidate = self._candidate_for_id(payload, str(primary_protein.get("id") or ""))
                    if not str(candidate.get("sequence") or "").strip():
                        primary_protein = None
                if primary_protein:
                    self._upsert_entity(state, primary_protein, activate=completed_without_confirmation)
                    state.verified_protein_ids = self._unique([primary_protein["id"]] + state.verified_protein_ids)

            family_payload = immediate.get("family") if isinstance(immediate.get("family"), dict) else {}
            if not family_payload and isinstance(protein_resolution.get("family"), dict):
                family_payload = dict(protein_resolution.get("family") or {})
            recommended_scope = str(protein_resolution.get("recommended_id") or "").strip()
            if mode in {"protein_family", "protein_functional_class"} and recommended_scope:
                label = str((immediate.get("protein") or {}).get("name") if isinstance(immediate.get("protein"), dict) else "").strip() or str(protein_resolution.get("interpreted_protein") or recommended_scope)
                if mode == "protein_family":
                    scope_snapshot = {
                        "kind": "family",
                        "id": recommended_scope,
                        "family_id": str(family_payload.get("family_id") or recommended_scope),
                        "label": label,
                    }
                else:
                    terms = [str(x).strip() for x in family_payload.get("normalized_terms") or [] if str(x).strip()]
                    strict = [str(x).strip() for x in family_payload.get("strict_terms") or [] if str(x).strip()]
                    broader = [str(x).strip() for x in family_payload.get("broader_terms") or [] if str(x).strip()]
                    scope_snapshot = {
                        "kind": "functional_class",
                        "id": recommended_scope,
                        "scope_id": str(family_payload.get("scope_id") or family_payload.get("family_id") or recommended_scope),
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
                state.verified_family_ids = self._unique([recommended_scope] + state.verified_family_ids)
                state.verified_protein_scopes = self._unique_scopes([scope_snapshot] + state.verified_protein_scopes)
                scope_entity = self._scope_entity(scope_snapshot, role="resolved_target")
                if scope_entity:
                    self._upsert_entity(state, scope_entity, activate=completed_without_confirmation)

            # A compound resolver result has one deterministic recommended ChEBI target.
            # Keep that target for follow-ups, but do not promote every alternative.
            operation = str(resolution.get("operation") or "")
            compound_resolution = resolution.get("compound_resolution") if isinstance(resolution.get("compound_resolution"), dict) else {}
            if operation == "resolve_compound" and compound_resolution:
                compound_id = str(compound_resolution.get("recommended_id") or "").strip().upper()
                candidate = self._candidate_for_id(compound_resolution, compound_id) if compound_id else {}
                if not candidate and compound_id:
                    candidate = {"chebi_id": compound_id, "name": compound_id, "smiles": ""}
                entity = self._compound_entity(candidate, role="resolved_target") if candidate else None
                if entity:
                    self._upsert_entity(state, entity, activate=completed_without_confirmation)
                    state.verified_compound_ids = self._unique([entity["id"]] + state.verified_compound_ids)

            # Identity inspection returns one explicitly requested entity. Lists of family
            # members/evidence are not silently converted into future query targets.
            if operation == "inspect_entity" and str(immediate.get("answer_mode") or "") == "entity_list":
                rows = [row for row in immediate.get("entities") or [] if isinstance(row, dict)]
                if len(rows) == 1:
                    entity_kind = str(immediate.get("entity_kind") or "")
                    row = rows[0]
                    if entity_kind == "compound":
                        entity = self._compound_entity(row, role="resolved_target")
                        if entity:
                            self._upsert_entity(state, entity, activate=False)
                            state.verified_compound_ids = self._unique([entity["id"]] + state.verified_compound_ids)
                    elif entity_kind == "literature":
                        # Explicitly inspecting a paper promotes that paper from related
                        # evidence to the conversational focus. This lets later anaphora
                        # such as “this paper” resolve independently from another ID named
                        # in the same user message.
                        entity = self._literature_entity(row, role="resolved_target")
                        if entity:
                            self._upsert_entity(state, entity, activate=False)
                    elif entity_kind in {"route", "route_step"}:
                        entity = self._listed_entity(
                            row,
                            entity_kind=entity_kind,
                            role="resolved_target",
                        )
                        if entity:
                            self._upsert_entity(state, entity, activate=False)

            # Any entity list shown to the user becomes an ordered workspace list.
            # The list items are related/reusable objects, not confirmed targets. This
            # gives ordinal references such as "the third paper/member" a stable object
            # meaning independent of how the LLM phrases its prose summary.
            if (
                operation != "inspect_entity"
                and str(immediate.get("answer_mode") or "") == "entity_list"
            ):
                entity_kind = str(immediate.get("entity_kind") or "").strip().lower()
                listed_entities: list[dict[str, Any]] = []
                for list_index, row in enumerate(immediate.get("entities") or [], start=1):
                    if not isinstance(row, dict):
                        continue
                    entity = self._listed_entity(
                        row,
                        entity_kind=entity_kind,
                        role="related_evidence",
                    )
                    if entity:
                        entity["related_index"] = list_index
                        self._upsert_entity(state, entity, activate=False)
                        listed_entities.append(entity)
                if listed_entities:
                    state.visible_entity_keys = [
                        self._entity_key(str(entity.get("kind") or ""), str(entity.get("id") or ""))
                        for entity in listed_entities
                    ]
                    state.visible_entity_kind = entity_kind
                    state.visible_page_index = 0

            # Literature returned by the research workspace is related evidence, not a new
            # active biochemical target. Preserve order for follow-ups such as "the second paper".
            if str(immediate.get("answer_mode") or "") == "research_workspace":
                literature_panels = [
                    row for row in immediate.get("source_panels") or []
                    if isinstance(row, dict) and (
                        str(row.get("section") or "") == "literature"
                        or str(row.get("entity_kind") or "") == "literature"
                        or str(row.get("id") or "") == "literature"
                        or str(row.get("id") or "").startswith("literature_")
                    )
                ]
                global_index = 0
                first_panel_entities: list[dict[str, Any]] = []
                first_panel_page_size = 10
                for panel_index, literature_panel in enumerate(literature_panels):
                    panel_entities: list[dict[str, Any]] = []
                    for row in literature_panel.get("items") or []:
                        if not isinstance(row, dict):
                            continue
                        entity = self._literature_entity(row, role="related_evidence")
                        if entity:
                            global_index += 1
                            entity["related_index"] = global_index
                            self._upsert_entity(state, entity, activate=False)
                            panel_entities.append(entity)
                    if not first_panel_entities and panel_entities:
                        first_panel_entities = panel_entities
                        first_panel_page_size = max(1, min(int(((literature_panel.get("pagination") or {}).get("page_size") or 10)), 30))
                if first_panel_entities:
                    first_page = first_panel_entities[:first_panel_page_size]
                    state.visible_entity_keys = [self._entity_key("literature", str(entity.get("id") or "")) for entity in first_page]
                    state.visible_entity_kind = "literature"
                    state.visible_page_index = 0

            if direction == "route_design" and immediate:
                route_entities: list[dict[str, Any]] = []
                route_rows: list[tuple[str, dict[str, Any]]] = []
                route_rows.extend(("routes", row) for row in immediate.get("routes") or [] if isinstance(row, dict))
                route_rows.extend(("exploratory_routes", row) for row in immediate.get("exploratory_routes") or [] if isinstance(row, dict))
                for route_index, (section, route) in enumerate(route_rows, start=1):
                    route_payload = deepcopy(route)
                    route_payload["search_context"] = {
                        "priority": str(immediate.get("priority") or "balanced"),
                        "host": str(immediate.get("host") or ""),
                        "max_steps": int(immediate.get("max_steps") or max(1, len(route.get("steps") or []))),
                        "analysis_layers": list(immediate.get("analysis_layers") or []),
                        "exploration_policy": str(immediate.get("exploration_policy") or "known_first"),
                    }
                    route_entity = self._route_entity(
                        route_payload,
                        role="related_evidence",
                        result_section=section,
                    )
                    if not route_entity:
                        continue
                    route_entity["related_index"] = route_index
                    if not route_entity.get("rank"):
                        route_entity["rank"] = route_index
                    self._upsert_entity(state, route_entity, activate=False)
                    route_entities.append(route_entity)
                    route_rank = int(route_entity.get("rank") or route_index)
                    for step in route.get("steps") or []:
                        if not isinstance(step, dict):
                            continue
                        step_entity = self._route_step_entity(
                            route,
                            step,
                            role="related_evidence",
                            route_rank=route_rank,
                        )
                        if step_entity:
                            self._upsert_entity(state, step_entity, activate=False)
                if route_entities:
                    state.visible_entity_keys = [
                        self._entity_key("route", str(entity.get("id") or ""))
                        for entity in route_entities
                    ]
                    state.visible_entity_kind = "route"
                    state.visible_page_index = 0
                    state.last_route_id = str(route_entities[0].get("id") or state.last_route_id)

            # Related association rows stay related. Preserve ordering so follow-ups such as
            # Page-local and historical ordering stays available on the verified
            # workspace handles so the primary agent can resolve follow-up references.
            known = immediate.get("known_associations") if isinstance(immediate.get("known_associations"), dict) else {}
            evidence_ids: list[str] = []
            for index, row in enumerate(known.get("items") or []):
                if not isinstance(row, dict):
                    continue
                item_id = str(row.get("candidate_id") or row.get("id") or "").strip()
                if not item_id:
                    continue
                evidence_ids.append(item_id)
                if direction == "reaction_to_enzyme":
                    resolution_stub = {
                        "mode": "session_verified_protein",
                        "interpreted_protein": str(row.get("name") or item_id),
                        "recommended_id": item_id,
                        "candidates": [{
                            "id": item_id,
                            "name": str(row.get("name") or item_id),
                            "organism": str(row.get("organism") or ""),
                            "input_mode": "protein_id",
                        }],
                    }
                    entity = self._protein_entity(resolution_stub, role="related_evidence")
                elif direction == "enzyme_to_reaction" and item_id.startswith("RHEA:"):
                    resolution_stub = {
                        "mode": "session_verified_rhea",
                        "interpreted_reaction": str(row.get("equation") or item_id),
                        "recommended_id": item_id,
                        "candidates": [{"rhea_id": item_id, "equation": str(row.get("equation") or item_id), "orientation": "forward"}],
                    }
                    entity = self._reaction_entity(resolution_stub, role="related_evidence")
                else:
                    entity = None
                if entity:
                    entity["related_index"] = index + 1
                    self._upsert_entity(state, entity, activate=False)
            state.recent_evidence_ids = self._unique(evidence_ids + state.recent_evidence_ids)

            ranking = immediate.get("ranking") if isinstance(immediate.get("ranking"), dict) else {}
            discovery = immediate.get("discovery_filter") if isinstance(immediate.get("discovery_filter"), dict) else {}
            if ranking.get("route_id"):
                state.last_route_id = str(ranking["route_id"])
            if discovery.get("result_mode"):
                state.last_result_mode = str(discovery["result_mode"])
            primary_target = (
                (primary_reaction or {}).get("id")
                or (primary_protein or {}).get("id")
                or recommended_scope
            )
            if primary_target:
                state.last_target = str(primary_target)
            self._states[key] = state


    def mark_visible_entities(
        self,
        session_id: str,
        *,
        entity_kind: str,
        entity_ids: list[str],
        page_index: int = 0,
    ) -> dict[str, Any]:
        """Mark a client-visible page using only entities already trusted by the server."""
        key = str(session_id or "").strip()
        kind = str(entity_kind or "").strip()
        if not key or kind not in _ENTITY_KINDS:
            return {"entity_kind": kind, "visible_count": 0, "page_index": max(0, int(page_index or 0))}
        requested = [str(value or "").strip() for value in entity_ids if str(value or "").strip()][:30]
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._states.get(key)
            if state is None:
                return {"entity_kind": kind, "visible_count": 0, "page_index": max(0, int(page_index or 0))}
            allowed_keys = {
                self._entity_key(str(row.get("kind") or ""), str(row.get("id") or ""))
                for row in state.entities
                if str(row.get("kind") or "") == kind
            }
            visible_keys = []
            for entity_id in requested:
                candidate = self._entity_key(kind, entity_id)
                if candidate in allowed_keys and candidate not in visible_keys:
                    visible_keys.append(candidate)
            state.visible_entity_keys = visible_keys
            state.visible_entity_kind = kind
            state.visible_page_index = max(0, int(page_index or 0))
            state.updated_at = now
            self._states[key] = state
            return {
                "entity_kind": kind,
                "visible_count": len(visible_keys),
                "page_index": state.visible_page_index,
                "visible_ids": [value.split(":", 1)[1] for value in visible_keys],
            }

    def remember_literature_items(
        self,
        session_id: str,
        items: list[dict[str, Any]],
        *,
        start_index: int = 0,
    ) -> None:
        """Remember literature rows loaded by remote pagination as reusable evidence."""
        key = str(session_id or "").strip()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            for offset, row in enumerate(items):
                if not isinstance(row, dict):
                    continue
                entity = self._literature_entity(row, role="related_evidence")
                if entity:
                    entity["related_index"] = max(0, int(start_index)) + offset + 1
                    self._upsert_entity(state, entity, activate=False)
            self._states[key] = state

    def validate_pending_confirmation(
        self,
        session_id: str,
        *,
        direction: str,
        target_id: str,
        positive_ids: list[str] | None = None,
        positive_sequence_inputs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate client-declared confirmed positives against the current server verification card.

        No validation is required when the caller supplies no confirmed positives. The
        ordinary ranking APIs remain callable directly; only the stronger claim that a
        seed was user-confirmed is protected by this card-bound trust check.
        """
        ids = self._unique([str(value or "").strip() for value in (positive_ids or []) if str(value or "").strip()], limit=40)
        sequence_inputs = [dict(value) for value in (positive_sequence_inputs or []) if isinstance(value, dict)]
        if not ids and not sequence_inputs:
            return {"valid": True, "required": False, "error_code": ""}

        key = str(session_id or "").strip()
        target = str(target_id or "").strip()
        requested_direction = str(direction or "").strip()
        if not key:
            return {"valid": False, "required": True, "error_code": "confirmation_session_missing"}

        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._states.get(key)
            pending = dict(state.pending_confirmation) if state is not None else {}
            if not pending:
                return {"valid": False, "required": True, "error_code": "confirmation_context_missing"}
            if str(pending.get("direction") or "") != requested_direction:
                return {"valid": False, "required": True, "error_code": "confirmation_direction_mismatch"}
            allowed_targets = {str(value) for value in pending.get("target_ids") or []}
            if not target or target not in allowed_targets:
                return {
                    "valid": False, "required": True, "error_code": "confirmation_target_mismatch",
                    "allowed_target_count": len(allowed_targets),
                }

            if requested_direction == "enzyme_to_reaction":
                allowed_ids = {str(value) for value in pending.get("positive_reaction_ids") or []}
                unknown = [value for value in ids if value not in allowed_ids]
                if sequence_inputs:
                    return {"valid": False, "required": True, "error_code": "confirmation_positive_type_mismatch"}
            else:
                allowed_ids = {str(value) for value in pending.get("positive_enzyme_ids") or []}
                unknown = [value for value in ids if value not in allowed_ids]
                expected_digests = {str(k): str(v) for k, v in (pending.get("positive_enzyme_sequence_digests") or {}).items()}
                for row in sequence_inputs:
                    candidate_id = str(row.get("id") or row.get("query_id") or "").strip()
                    sequence = str(row.get("sequence") or "").strip()
                    if not candidate_id or candidate_id not in allowed_ids:
                        unknown.append(candidate_id or "<missing-id>")
                        continue
                    expected = expected_digests.get(candidate_id, "")
                    actual = self._sequence_digest(sequence)
                    if not expected or actual != expected:
                        return {
                            "valid": False, "required": True,
                            "error_code": "confirmation_sequence_mismatch",
                            "candidate_id": candidate_id,
                        }

            if unknown:
                return {
                    "valid": False, "required": True, "error_code": "confirmation_positive_not_verified",
                    "unknown_count": len(set(unknown)),
                }
            return {
                "valid": True, "required": True, "error_code": "",
                "verified_positive_count": len(ids) + len(sequence_inputs),
            }

    def consume_pending_confirmation(self, session_id: str, *, direction: str, target_id: str) -> None:
        key = str(session_id or "").strip()
        target = str(target_id or "").strip()
        if not key:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._states.get(key)
            if state is None or not state.pending_confirmation:
                return
            pending = state.pending_confirmation
            if str(pending.get("direction") or "") != str(direction or "").strip():
                return
            if target not in {str(value) for value in pending.get("target_ids") or []}:
                return
            state.pending_confirmation = {}
            state.updated_at = now
            self._states[key] = state

    def confirm_reaction(
        self,
        session_id: str,
        *,
        reaction_id: str = "",
        reaction_smiles: str = "",
        query_id: str = "",
        orientation: str = "forward",
    ) -> None:
        key = str(session_id or "").strip()
        target_id = str(reaction_id or query_id or "").strip()
        if not key or not target_id:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            candidate = {
                "rhea_id": target_id,
                "equation": reaction_smiles or target_id,
                "orientation": orientation or "forward",
            }
            if reaction_smiles:
                candidate["reaction_smiles"] = reaction_smiles
            resolution = {
                "mode": "raw_reaction_smiles" if reaction_smiles else "session_verified_rhea",
                "interpreted_reaction": reaction_smiles or target_id,
                "recommended_id": target_id,
                "candidates": [candidate],
            }
            entity = self._reaction_entity(resolution, role="confirmed_target")
            if entity:
                self._upsert_entity(state, entity, activate=True)
                state.verified_reaction_ids = self._unique([target_id] + state.verified_reaction_ids)
                state.last_target = target_id
            self._states[key] = state

    def confirm_protein(
        self,
        session_id: str,
        *,
        protein_id: str = "",
        sequence: str = "",
        query_id: str = "",
    ) -> None:
        key = str(session_id or "").strip()
        target_id = str(protein_id or query_id or "").strip()
        if not key or not target_id:
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            existing = next((row for row in state.entities if row.get("kind") == "protein" and str(row.get("id") or "") == target_id), None)
            candidate: dict[str, Any] = {}
            if isinstance(existing, dict):
                payload = existing.get("payload") if isinstance(existing.get("payload"), dict) else {}
                candidate = self._candidate_for_id(payload, target_id)
            if not candidate:
                candidate = {"id": target_id, "name": target_id, "input_mode": "raw_protein_sequence" if sequence else "protein_id"}
            if sequence:
                candidate["sequence"] = sequence
                candidate["input_mode"] = "raw_protein_sequence"
            resolution = {
                "mode": str(candidate.get("input_mode") or "protein_id"),
                "interpreted_protein": str(candidate.get("name") or target_id),
                "recommended_id": target_id,
                "candidates": [candidate],
            }
            entity = self._protein_entity(resolution, role="confirmed_target")
            if entity:
                self._upsert_entity(state, entity, activate=True)
                state.verified_protein_ids = self._unique([target_id] + state.verified_protein_ids)
                state.last_target = target_id
            self._states[key] = state

    def confirm_protein_scope(self, session_id: str, scope: dict[str, Any]) -> None:
        key = str(session_id or "").strip()
        if not key or not isinstance(scope, dict):
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            entity = self._scope_entity(scope, role="confirmed_target")
            if entity:
                self._upsert_entity(state, entity, activate=True)
                state.verified_family_ids = self._unique([entity["id"]] + state.verified_family_ids)
                state.verified_protein_scopes = self._unique_scopes([scope] + state.verified_protein_scopes)
                state.last_target = entity["id"]
            self._states[key] = state

    def remember_dialogue_turn(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str,
        response_type: str = "",
        limit: int = 8,
    ) -> None:
        """Persist the visible user/assistant exchange as chronological agent history."""
        key=str(session_id or "").strip()
        if not key:
            return
        user=str(user_text or "").strip()[:1200]
        assistant=str(assistant_text or "").strip()[:1800]
        if not user and not assistant:
            return
        now=time.time()
        with self._lock:
            self._prune(now)
            state=self._state(key,now)
            if user:
                state.conversation_history.append({"role":"user","content":user})
            if assistant:
                state.conversation_history.append({"role":"assistant","content":assistant})
            history_limit=max(4,min(int(limit)*2,32))
            state.conversation_history=state.conversation_history[-history_limit:]
            # Keep a lightweight compatibility view until downstream callers/tests are
            # migrated. It no longer carries a synthetic epistemic label.
            state.recent_dialogue.append({
                "user":user,
                "assistant":assistant,
                "response_type":str(response_type or "")[:80],
            })
            state.recent_dialogue=state.recent_dialogue[-max(2,min(int(limit),12)):]
            self._states[key]=state

    @staticmethod
    def _compact_result_context(result: dict[str,Any],direction: str) -> dict[str,Any]:
        ranking=result.get("ranking") if isinstance(result.get("ranking"),dict) else {}
        discovery=result.get("discovery_filter") if isinstance(result.get("discovery_filter"),dict) else {}
        routing=result.get("routing") if isinstance(result.get("routing"),dict) else {}
        candidates=[]
        for row in list(result.get("candidates") or [])[:24]:
            if not isinstance(row,dict):
                continue
            support=row.get("support_applicability") if isinstance(row.get("support_applicability"),dict) else {}
            mechanisms=(
                (row.get("fibre_resolution") or {}).get("mechanistic_coordinates")
                if isinstance(row.get("fibre_resolution"),dict) else []
            )
            candidates.append({
                "rank":row.get("rank"),
                "candidate_id":str(row.get("candidate_id") or "")[:160],
                "name":str(row.get("name") or "")[:240],
                "substrate_name":str(row.get("substrate_name") or "")[:240],
                "product_name":str(row.get("product_name") or "")[:240],
                "known_association":bool(row.get("known_association")),
                "model_support_index":row.get("model_support_index"),
                "correspondence_defect":row.get("correspondence_defect"),
                "nearest_joint_positive_distance":support.get("nearest_joint_positive_distance"),
                "candidate_support_distance":support.get("current_candidate_support_distance"),
                "query_support_distance":support.get("query_support_distance"),
                "mechanistic_coordinates":[str(x) for x in mechanisms or []][:8],
            })
        target={}
        for key in ("protein","reaction"):
            value=result.get(key)
            if isinstance(value,dict) and value:
                target={
                    "kind":key,
                    "id":str(value.get("id") or value.get("rhea_id") or "")[:160],
                    "name":str(value.get("name") or value.get("equation") or "")[:360],
                }
                break
        return {
            "direction":str(direction or ""),
            "target":target,
            "result_mode":str(discovery.get("result_mode") or ""),
            "route_id":str(ranking.get("route_id") or ""),
            "candidate_count":len(result.get("candidates") or []),
            "candidates":candidates,
            "known_association_count":int((result.get("known_associations") or {}).get("count") or 0)
                if isinstance(result.get("known_associations"),dict) else 0,
            "reaction_constraints":deepcopy(routing.get("reaction_constraints") or {}),
            "source":"verified_server_execution_result",
        }

    def remember_execution_result(self, session_id: str, result: dict[str, Any], *, direction: str = "") -> None:
        """Persist the actual executed retrieval scope for later relative follow-ups.

        This is execution state, not entity resolution. It is written only after a
        successful ranking/route request and replaces browser-maintained continuation
        flags as the trusted source for previous result mode and route.
        """
        key = str(session_id or "").strip()
        if not key or not isinstance(result, dict):
            return
        now = time.time()
        with self._lock:
            self._prune(now)
            state = self._state(key, now)
            if direction:
                state.last_direction = str(direction)
            discovery = result.get("discovery_filter") if isinstance(result.get("discovery_filter"), dict) else {}
            result_mode = str(discovery.get("result_mode") or "").strip()
            if result_mode:
                state.last_result_mode = result_mode
            raw_policy = str(discovery.get("policy") or "").strip()
            if raw_policy:
                state.last_association_policy = {
                    "retain_recorded_associations_only": "known_only",
                    "exclude_recorded_associations": "exclude_known",
                    "rank_recorded_and_unrecorded_together": "rank_with_known",
                    "separate_recorded_evidence": "separate_known",
                }.get(raw_policy, "separate_known")
            ranking = result.get("ranking") if isinstance(result.get("ranking"), dict) else {}
            route_view = result.get("route_view") if isinstance(result.get("route_view"), dict) else {}
            route_id = str(ranking.get("route_id") or route_view.get("route_id") or "").strip()
            if route_id:
                state.last_route_id = route_id
            compact=self._compact_result_context(
                result,str(direction or state.last_direction or "")
            )
            compact["recorded_at_unix"]=now
            state.last_result_context=compact
            state.execution_history.append(deepcopy(compact))
            state.execution_history=state.execution_history[-8:]

            target = compact.get("target") if isinstance(compact.get("target"), dict) else {}
            for row in list(result.get("candidates") or [])[:24]:
                if not isinstance(row, dict) or bool(row.get("known_association")):
                    continue
                entity = self._model_candidate_entity(
                    row,
                    direction=str(direction or state.last_direction or ""),
                    target=target,
                )
                if entity:
                    self._upsert_entity(state, entity, activate=False)

            self._states[key] = state

    def execution_context(self, session_id: str, *, ui_language: str = "en") -> dict[str, Any]:
        snapshot = self.snapshot(session_id)
        return {
            "previous_direction": str(snapshot.get("last_direction") or ""),
            "previous_result_mode": str(snapshot.get("last_result_mode") or ""),
            "previous_association_policy": str(snapshot.get("last_association_policy") or ""),
            "previous_route_id": str(snapshot.get("last_route_id") or ""),
            "previous_target": str(snapshot.get("last_target") or ""),
            "last_result_context": deepcopy(snapshot.get("last_result_context") or {}),
            "conversation_history": deepcopy(snapshot.get("conversation_history") or []),
            "ui_language": str(ui_language or "en"),
        }

    def model_history(self, session_id: str, *, limit: int = 24) -> list[dict[str,str]]:
        """Return chronological visible conversation items for the controller."""
        snapshot=self.model_snapshot(session_id)
        rows=[]
        for raw in list(snapshot.get("conversation_history") or [])[-max(2,min(int(limit),40)):]:
            if not isinstance(raw,dict):
                continue
            role=str(raw.get("role") or "").strip()
            content=str(raw.get("content") or "").strip()
            if role in {"user","assistant"} and content:
                rows.append({"role":role,"content":content})
        return rows

    def clear(self, session_id: str) -> None:
        key = str(session_id or "").strip()
        if not key:
            return
        with self._lock:
            self._states.pop(key, None)
