from __future__ import annotations

from typing import Any


DUAL_MEMORY_CONTRACT = "bridge_dual_memory_v1"


def canonical_memory_context(
    query: dict[str, Any], routing: dict[str, Any]
) -> dict[str, Any]:
    """Describe actual BRIDGE memory activation, independently of legacy labels.

    Training-graph memory is owned by the frozen model; a query-local episodic
    update is present only when the runtime confirms a nonzero gated update.
    Evidence lookup counts never qualify as episodic supports.
    """
    if str(query.get("model_expert") or "") != "bridge_final":
        return {}
    gated_update = bool(query.get("episodic_memory_applied"))
    source = str(routing.get("seed_source") or "")
    # BRIDGE has two long-term readouts. R2E and standard E2R expose a
    # per-query relation weight; inductive E2R exposes a separate authority
    # plus two-sided graph propagation. Missing readout fields are unknown,
    # never a fabricated zero relation count or zero activation strength.
    long_term: dict[str, Any] = {
        "source": "clean2023_frozen_training_graph",
        "gate": "query_conditioned",
    }
    if query.get("train_memory_relation_count") is not None:
        long_term["query_relation_count"] = int(
            query["train_memory_relation_count"]
        )
    if query.get("relation_context_weight") is not None:
        long_term["gate_weight"] = float(query["relation_context_weight"])
    elif isinstance(query.get("evidence_authority"), dict):
        relation = query["evidence_authority"].get("long_term_relation")
        if relation is not None:
            long_term["gate_weight"] = float(relation)
    if str(query.get("route_id") or "") == "bridge-e2r-inductive-bipartite-graph":
        long_term["readout"] = "inductive_two_sided_relation_graph"
    context: dict[str, Any] = {
        "contract": DUAL_MEMORY_CONTRACT,
        "mode": "long_term_plus_episodic" if gated_update else "long_term_only",
        "long_term_memory": long_term,
    }
    if gated_update:
        # Reject model-visible claims of catalog updates even if a legacy
        # caller incorrectly requested them: that would be a real provenance
        # violation, not another legitimate BRIDGE memory mode.
        if source != "user_confirmed":
            raise ValueError(
                "Runtime episodic memory was activated without confirmed "
                "new-relation provenance"
            )
        context["episodic_memory"] = {
            "source": "confirmed_new_relation",
            "support_count": int(query.get("episodic_support_count") or 0),
            "gate_weight": float(query.get("episodic_memory_weight") or 0.0),
            "training_graph_duplicates_ignored": int(
                query.get("training_seed_count") or 0
            ),
        }
    return context
