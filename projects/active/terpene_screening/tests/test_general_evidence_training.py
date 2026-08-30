from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.train_general_evidence_retriever import (
    _directional_full_candidate_loss,
    _loss_candidate_view,
    _query_positive_rows,
)


def test_query_positive_rows_builds_directional_multi_positive_targets():
    associations = pd.DataFrame([
        {"protein_id": "P1", "reaction_id": "R1"},
        {"protein_id": "P2", "reaction_id": "R1"},
        {"protein_id": "P1", "reaction_id": "R2"},
    ])
    q, positives = _query_positive_rows(
        associations, direction="r2e",
        query_index={"R1": 0, "R2": 1}, candidate_index={"P1": 0, "P2": 1},
    )
    mapping = dict(zip(q, positives))
    assert mapping["R1"].tolist() == [0, 1]
    assert mapping["R2"].tolist() == [0]


def test_training_entity_candidate_scope_excludes_unseen_candidates():
    associations = pd.DataFrame([
        {"protein_id": "P1", "reaction_id": "R1"},
        {"protein_id": "P2", "reaction_id": "R1"},
        {"protein_id": "P2", "reaction_id": "R2"},
    ])
    selected, rows, index = _loss_candidate_view(
        associations,
        direction="r2e",
        candidate_ids=["P0", "P1", "P2", "P_TEST"],
        scope="training_entities",
    )
    assert selected == ["P1", "P2"]
    assert rows.tolist() == [1, 2]
    assert index == {"P1": 0, "P2": 1}


def test_full_universe_candidate_scope_preserves_production_behavior():
    associations = pd.DataFrame([
        {"protein_id": "P1", "reaction_id": "R1"},
    ])
    selected, rows, index = _loss_candidate_view(
        associations,
        direction="e2r",
        candidate_ids=["R0", "R1", "R_TEST"],
        scope="full_universe",
    )
    assert selected == ["R0", "R1", "R_TEST"]
    assert rows.tolist() == [0, 1, 2]
    assert index == {"R0": 0, "R1": 1, "R_TEST": 2}


def test_directional_loss_rewards_positive_above_kth_negative():
    candidates = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]), dim=1)
    good = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    bad = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0]]), dim=1)
    good_loss, _ = _directional_full_candidate_loss(
        good, candidates, [np.asarray([0])], temperature=0.1,
        topk_k=1, topk_weight=0.1, topk_margin=0.0, all_positive_weight=0.05,
    )
    bad_loss, _ = _directional_full_candidate_loss(
        bad, candidates, [np.asarray([0])], temperature=0.1,
        topk_k=1, topk_weight=0.1, topk_margin=0.0, all_positive_weight=0.05,
    )
    assert float(good_loss) < float(bad_loss)


def test_positive_aware_filter_masks_teacher_suspected_false_negative():
    candidates = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]), dim=1
    )
    student = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    teacher = student.clone()
    unfiltered, unfiltered_parts = _directional_full_candidate_loss(
        student,
        candidates,
        [np.asarray([0])],
        temperature=0.1,
        topk_k=1,
        topk_weight=0.0,
        topk_margin=0.0,
        all_positive_weight=0.0,
    )
    filtered, filtered_parts = _directional_full_candidate_loss(
        student,
        candidates,
        [np.asarray([0])],
        temperature=0.1,
        topk_k=1,
        topk_weight=0.0,
        topk_margin=0.0,
        all_positive_weight=0.0,
        teacher_query_embeddings=teacher,
        false_negative_margin=0.05,
    )
    assert filtered < unfiltered
    assert unfiltered_parts["false_negative_filtered_fraction"] == 0.0
    assert filtered_parts["false_negative_filtered_fraction"] > 0.0


def test_positive_aware_filter_respects_max_filter_cap():
    candidates = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [0.97, 0.03], [0.0, 1.0]]), dim=1
    )
    query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0]]), dim=1)
    _, parts = _directional_full_candidate_loss(
        query,
        candidates,
        [np.asarray([0])],
        temperature=0.1,
        topk_k=1,
        topk_weight=0.0,
        topk_margin=0.0,
        all_positive_weight=0.0,
        teacher_query_embeddings=query,
        false_negative_margin=0.05,
        false_negative_max_filter=1,
    )
    assert parts["false_negative_filtered_fraction"] == 0.25


def test_recadam_optimizer_uses_frozen_source_parameters():
    from projects.active.terpene_screening.third_party.recadam import RecAdam

    current = torch.nn.Parameter(torch.tensor([1.0]))
    source = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
    optimizer = RecAdam(
        [current], lr=1e-3, anneal_fun="sigmoid", anneal_k=0.1, anneal_t0=100, anneal_w=1.0,
        pretrain_cof=100.0, pretrain_params=[source], weight_decay=0.0,
    )
    current.grad = torch.zeros_like(current)
    optimizer.step()
    assert current.item() < 1.0


def test_build_optimizer_selects_recadam_and_matches_trainable_parameters():
    from projects.active.terpene_screening.third_party.recadam import RecAdam
    from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig, TerpeneDualTower
    from projects.active.terpene_screening.train_general_evidence_retriever import _build_optimizer

    config = ModelConfig(protein_input_dim=4, reaction_input_dim=3, hidden_dim=8, embedding_dim=2, dropout=0.0)
    model = TerpeneDualTower(config)
    source = TerpeneDualTower(config)
    source.load_state_dict(model.state_dict())
    for parameter in model.protein_tower.parameters():
        parameter.requires_grad = False
    optimizer = _build_optimizer(
        model, source, optimizer_name="recadam", learning_rate=1e-4, weight_decay=0.0,
        total_training_steps=20, recadam_anneal_fun="sigmoid", recadam_anneal_k=0.1,
        recadam_anneal_t0=0, recadam_pretrain_cof=5000.0,
    )
    assert isinstance(optimizer, RecAdam)
    assert len(optimizer.param_groups) == 1
    assert len(optimizer.param_groups[0]["params"]) == len(optimizer.param_groups[0]["pretrain_params"])
    assert optimizer.param_groups[0]["anneal_t0"] == 10


def test_mammoth_lwf_distillation_prefers_teacher_aligned_student():
    from projects.active.terpene_screening.third_party.mammoth_lwf import distillation

    teacher = torch.tensor([[5.0, 1.0, -2.0], [0.0, 4.0, 1.0]])
    aligned = teacher.clone()
    reversed_student = teacher.flip(dims=[1])
    assert distillation(teacher, aligned, 2.0) < distillation(teacher, reversed_student, 2.0)


def test_mammoth_bidirectional_distillation_penalizes_column_reordering():
    from projects.active.terpene_screening.third_party.mammoth_lwf import bidirectional_distillation

    teacher = torch.tensor([[6.0, 1.0, 0.0], [5.0, 4.0, -1.0], [0.0, 2.0, 7.0]])
    aligned = teacher.clone()
    column_swapped = teacher[:, [1, 0, 2]]
    assert bidirectional_distillation(teacher, aligned, 1.0) < bidirectional_distillation(teacher, column_swapped, 1.0)
