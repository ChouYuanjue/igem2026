"""Tests for the released E2R biochemical association evidence path."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from projects.active.bridge.runtime.final_system import (
    E2R_INDUCTIVE_CONTEXT, FinalBridgeRuntime,
)
from projects.active.bridge.runtime.inductive_relation import (
    InductiveE2RRelation, InductiveRelationConfig,
)


def test_inductive_e2r_asset_is_frozen_and_bounded():
    values = json.loads(E2R_INDUCTIVE_CONTEXT.read_text())
    assert values["schema"] == "bridge-e2r-inductive-bipartite-graph-production-v1"
    assert values["neighborhood"]["protein_neighbors"] == 16
    assert values["neighborhood"]["reaction_neighbors"] == 8
    assert values["coefficients"]["observed_links"] >= 0
    assert values["coefficients"]["reaction_analogy"] >= 0
    assert values["evaluation"]["frozen_query_separated_test_positive_relations"] == 21505


def test_missing_graph_relation_stays_neutral_for_seen_protein():
    model = object.__new__(InductiveE2RRelation)
    model.nr = 5
    model.rt = SimpleNamespace(index=SimpleNamespace(protein_index={"P": 0}))
    model.protein_seen = np.array([True])
    corr = model.correction("P", np.array([0, 2, 4]), broad_std=0.1)
    np.testing.assert_array_equal(corr, np.zeros(3))


def test_automatically_selects_validated_masked_discovery_route():
    rt = object.__new__(FinalBridgeRuntime)
    with patch.object(FinalBridgeRuntime, "_rank_reactions_inductive_relation",
                      return_value={"query": {"route_id": "inductive"}}) as call:
        value = rt.rank_reactions({"enzyme_id": "P", "top_k": 10, "mask_clean2023": True})
    assert value["query"]["route_id"] == "inductive"
    call.assert_called_once()


def test_explicit_inductive_route_is_supported():
    rt = object.__new__(FinalBridgeRuntime)
    with patch.object(FinalBridgeRuntime, "_rank_reactions_inductive_relation",
                      return_value={"query": {"route_id": "inductive"}}) as call:
        value = rt.rank_reactions({"e2r_gate_profile": "inductive-bipartite",
                                   "enzyme_id": "P", "mask_clean2023": True})
    assert value["query"]["route_id"] == "inductive"
    call.assert_called_once()


def test_original_query_router_profile_remains_compatible():
    rt = object.__new__(FinalBridgeRuntime)
    with patch.object(FinalBridgeRuntime, "_rank_reactions_joint_query_v4",
                      return_value={"query": {"route_id": "legacy"}}) as call:
        value = rt.rank_reactions({"e2r_gate_profile": "joint-query-v4",
                                   "enzyme_id": "P", "mask_clean2023": True})
    assert value["query"]["route_id"] == "legacy"
    call.assert_called_once()


def test_additional_database_known_reactions_allow_inductive_evidence_route():
    rt = object.__new__(FinalBridgeRuntime)
    with patch.object(FinalBridgeRuntime, "_rank_reactions_inductive_relation",
                      return_value={"query": {"route_id": "inductive"}}) as call:
        value = rt.rank_reactions({"enzyme_id": "P", "top_k": 10,
                                   "mask_clean2023": True,
                                   "mask_reaction_ids": ["RHEA:12345"]})
    assert value["query"]["route_id"] == "inductive"
    assert call.call_args.args[0]["mask_reaction_ids"] == ["RHEA:12345"]
