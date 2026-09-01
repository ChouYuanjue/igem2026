from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from projects.active.terpene_screening.rank_open_world import BoundedIdentityHiddenResidualReactionDualTower
from projects.active.terpene_screening.train_cleanroom_directional_identity_aux_residual import (
    _local_positive_rows,
    _sample_global_teacher_candidates,
    configure_identity_residual_trainables,
)
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig

ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = ROOT / "projects/active/terpene_screening/CATALYST_COMPREHENSIVE_DIRECTIONAL_CENTER_V1.json"


def test_protocol_is_frozen_before_performance_and_forbids_outer_selection() -> None:
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["status"] == "frozen_before_any_new_split_performance_materialization"
    assert protocol["global_rules"]["external_or_revealed_outer_metrics_used_for_selection"] is False
    assert protocol["global_rules"]["external_benchmarks_allowed_for_selection"] == []
    assert protocol["development"]["split_salt"] == "comprehensive_directional_center_v1_dev_20260901"
    assert protocol["development"]["development_folds"] == [0, 1, 2]
    assert protocol["confirmation"]["split_salt"] == "comprehensive_directional_center_v1_confirm_20260901"
    assert protocol["confirmation"]["dev_fold"] == 4
    assert protocol["model_family"]["max_residual_ratio"] == 0.1
    assert protocol["model_family"]["center_training_recipe"]["e2r_hard_candidates_per_step"] == 512


def test_only_auxiliary_projection_is_trainable() -> None:
    config = ModelConfig(protein_input_dim=7, reaction_input_dim=11, hidden_dim=13, embedding_dim=5, dropout=0.0)
    model = BoundedIdentityHiddenResidualReactionDualTower(config, aux_input_dim=3, max_residual_ratio=0.1)
    trainable = configure_identity_residual_trainables(model)
    assert trainable == [model.aux_to_hidden.weight]
    assert [name for name, parameter in model.named_parameters() if parameter.requires_grad] == ["aux_to_hidden.weight"]


def test_e2r_teacher_candidate_sampling_retains_all_positives_and_is_bounded() -> None:
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    candidates = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [-1.0, 0.0], [0.0, -1.0]],
        dtype=torch.float32,
    )
    positives = [np.asarray([4], dtype=np.int64), np.asarray([5], dtype=np.int64)]
    selected = _sample_global_teacher_candidates(
        queries,
        candidates,
        positives,
        hard_candidates=2,
        random_candidates=1,
        rng=__import__("random").Random(7),
    )
    assert {4, 5} <= set(selected.tolist())
    assert len(selected) <= 5  # two positives + at most two hard + one random
    local = _local_positive_rows(positives, selected)
    assert len(local) == 2
    assert all(len(rows) == 1 for rows in local)
