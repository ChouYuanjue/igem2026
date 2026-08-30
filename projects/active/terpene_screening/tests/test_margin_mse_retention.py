from __future__ import annotations

import torch

from projects.active.terpene_screening.third_party.margin_mse_loss import MarginMSELoss
from projects.active.terpene_screening.train_general_evidence_retriever import (
    _bidirectional_margin_mse,
    _build_bidirectional_margin_pairs,
)


def test_margin_mse_zero_when_student_preserves_teacher_gaps():
    teacher = torch.tensor([[4.0, 2.0, 1.0], [0.5, 3.0, 1.0]])
    pairs = _build_bidirectional_margin_pairs(teacher, [(0, 0), (1, 1)], topk=1)
    loss = _bidirectional_margin_mse(teacher.clone(), teacher, pairs, MarginMSELoss())
    assert torch.isclose(loss, torch.tensor(0.0))


def test_margin_mse_detects_boundary_swaps_in_either_direction():
    teacher = torch.tensor([[4.0, 2.0, 1.0], [0.5, 3.0, 1.0]])
    pairs = _build_bidirectional_margin_pairs(teacher, [(0, 0), (1, 1)], topk=1)
    student = teacher.clone()
    student[0, 0] = 1.5
    loss = _bidirectional_margin_mse(student, teacher, pairs, MarginMSELoss())
    assert loss > 0
    assert len(pairs['row_query']) > 0
    assert len(pairs['col_query']) > 0


def test_margin_pair_mining_masks_other_known_positives():
    teacher = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    pairs = _build_bidirectional_margin_pairs(teacher, [(0, 0), (0, 1)], topk=1)
    assert set(pairs['row_negative'].tolist()) == {2}
