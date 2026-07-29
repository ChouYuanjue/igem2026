from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.gate_matrix import (
    product_carbon_skeleton_signature,
)
from projects.active.terpene_screening.evaluate_dual_kernel_collaborative_retrieval import (
    normalized_adjacency,
    topk_affinity,
)
from projects.active.terpene_screening.evaluate_locked_dual_kernel_route import rescue
from projects.active.terpene_screening.evaluate_two_stage_motif_residual import (
    build_base_hard_triples,
    rank_top_shortlist,
)
from projects.active.terpene_screening.train_dual_tower_cold import (
    batch_hard_tps_metric_loss,
    build_tps_metric_masks,
    build_tps_structured_negative_triples,
)


def test_carbon_skeleton_signature_ignores_bond_order_and_oxidation() -> None:
    saturated = product_carbon_skeleton_signature("CCOP(=O)(O)O>>C1CCCCC1")
    unsaturated = product_carbon_skeleton_signature("CCOP(=O)(O)O>>C1=CCCCC1")
    alcohol = product_carbon_skeleton_signature("CCOP(=O)(O)O>>OC1CCCCC1")
    alternative = product_carbon_skeleton_signature("CCOP(=O)(O)O>>CC1CCCC1")
    assert saturated != "unknown"
    assert saturated == unsaturated == alcohol
    assert alternative != saturated


def test_structured_negatives_are_same_precursor_different_skeleton() -> None:
    proteins = ["p1", "p2", "p3", "p4"]
    reactions = ["r1", "r2", "r3"]
    pairs = pd.DataFrame(
        [
            ("p1", "r1"),
            ("p2", "r1"),
            ("p3", "r2"),
            ("p4", "r3"),
        ],
        columns=["Entry", "rhea_id"],
    )
    precursor = {"r1": "C15", "r2": "C15", "r3": "C20"}
    skeleton = {"r1": "s1", "r2": "s2", "r3": "s3"}
    groups = {"p1": "g1", "p2": "g1", "p3": "g3", "p4": "g4"}
    features = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.9, 0.1], [0.0, 1.0]],
        dtype=np.float32,
    )
    reaction_rows, positive_rows, negative_rows = build_tps_structured_negative_triples(
        pairs,
        proteins,
        reactions,
        features,
        groups,
        precursor,
        skeleton,
        negatives_per_positive=2,
    )
    triples = list(zip(reaction_rows, positive_rows, negative_rows))
    assert triples
    positives_by_reaction = {
        reaction: set(group["Entry"])
        for reaction, group in pairs.groupby("rhea_id")
    }
    labels_by_protein = {
        protein: {
            (precursor[reaction], skeleton[reaction])
            for reaction in pairs.loc[pairs["Entry"].eq(protein), "rhea_id"]
        }
        for protein in proteins
    }
    for reaction_row, positive_row, negative_row in triples:
        reaction = reactions[int(reaction_row)]
        positive = proteins[int(positive_row)]
        negative = proteins[int(negative_row)]
        assert negative not in positives_by_reaction[reaction]
        assert groups[negative] != groups[positive]
        negative_skeletons = {
            local_skeleton
            for local_precursor, local_skeleton in labels_by_protein[negative]
            if local_precursor == precursor[reaction]
        }
        assert negative_skeletons
        assert skeleton[reaction] not in negative_skeletons


def test_metric_masks_support_multi_product_proteins() -> None:
    pairs = pd.DataFrame(
        [
            ("p1", "r1"),
            ("p1", "r2"),
            ("p2", "r1"),
            ("p3", "r3"),
            ("p4", "r4"),
        ],
        columns=["Entry", "rhea_id"],
    )
    precursor = {"r1": "C15", "r2": "C15", "r3": "C15", "r4": "C20"}
    skeleton = {"r1": "s1", "r2": "s2", "r3": "s3", "r4": "s4"}
    reaction_positive, reaction_negative, protein_positive, protein_negative = (
        build_tps_metric_masks(
            pairs,
            ["p1", "p2", "p3", "p4"],
            ["r1", "r2", "r3", "r4"],
            precursor,
            skeleton,
        )
    )
    assert reaction_negative[0, 1]
    assert reaction_negative[0, 2]
    assert not reaction_negative[0, 3]
    assert protein_positive[0, 1]
    assert not protein_negative[0, 1]
    assert protein_negative[0, 2]
    assert not protein_negative[0, 3]
    assert not np.diag(reaction_positive).any()
    assert not np.diag(protein_positive).any()


def test_batch_hard_metric_loss_is_finite_and_differentiable() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        dtype=torch.float32,
        requires_grad=True,
    )
    positive = torch.tensor(
        [
            [False, True, False, False],
            [True, False, False, False],
            [False, False, False, True],
            [False, False, True, False],
        ]
    )
    negative = torch.tensor(
        [
            [False, False, True, True],
            [False, False, True, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    loss = batch_hard_tps_metric_loss(embeddings, positive, negative, margin=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_batch_hard_metric_loss_ignores_anchors_without_both_sets() -> None:
    embeddings = torch.eye(3, requires_grad=True)
    empty = torch.zeros((3, 3), dtype=torch.bool)
    loss = batch_hard_tps_metric_loss(embeddings, empty, empty, margin=0.1)
    assert float(loss) == 0.0


def test_base_hard_triples_use_query_specific_safe_false_positives() -> None:
    proteins = ["p1", "p2", "p3", "p4"]
    reactions = ["r1", "r2", "r3"]
    pairs = pd.DataFrame(
        [("p1", "r1"), ("p2", "r1"), ("p3", "r2"), ("p4", "r3")],
        columns=["Entry", "rhea_id"],
    )
    base_scores = np.asarray(
        [
            [0.5, 0.4, 0.9, 0.8],
            [0.2, 0.1, 0.7, 0.3],
            [0.1, 0.2, 0.3, 0.8],
        ],
        dtype=np.float32,
    )
    triples = build_base_hard_triples(
        train_pairs=pairs,
        base_scores=base_scores,
        protein_ids=proteins,
        reaction_ids=reactions,
        protein_to_row={value: index for index, value in enumerate(proteins)},
        reaction_to_row={value: index for index, value in enumerate(reactions)},
        protein_groups={"p1": "g1", "p2": "g1", "p3": "g3", "p4": "g4"},
        precursor_by_reaction={"r1": "C15", "r2": "C15", "r3": "C20"},
        skeleton_by_reaction={"r1": "s1", "r2": "s2", "r3": "s3"},
        shortlist_depth=4,
        negatives_per_positive=2,
    )
    observed = set(zip(*(values.tolist() for values in triples)))
    assert observed == {
        (0, 0, 2),
        (0, 1, 2),
        (1, 2, 0),
        (1, 2, 1),
    }
    assert all(negative_row != 3 for _, _, negative_row in observed)


def test_dual_kernel_affinity_is_train_only_and_row_normalized() -> None:
    similarity = np.asarray(
        [
            [1.0, 0.8, 0.7, 0.6],
            [0.8, 1.0, 0.5, 0.4],
            [0.7, 0.5, 1.0, 0.9],
            [0.6, 0.4, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    affinity = topk_affinity(similarity, np.asarray([0, 2]), k=1, temperature=0.08)
    dense = affinity.toarray()
    assert np.allclose(dense.sum(axis=1), 1.0)
    assert not dense[:, [1, 3]].any()
    assert np.count_nonzero(dense, axis=1).tolist() == [1, 1, 1, 1]


def test_dual_kernel_degree_normalization_is_finite() -> None:
    pairs = pd.DataFrame(
        [("p1", "r1"), ("p2", "r1"), ("p2", "r2")],
        columns=["Entry", "rhea_id"],
    )
    adjacency = normalized_adjacency(
        pairs,
        {"r1": 0, "r2": 1, "r3": 2},
        {"p1": 0, "p2": 1, "p3": 2},
        (3, 3),
        degree_power=1.0,
    ).toarray()
    assert np.isfinite(adjacency).all()
    assert adjacency[2].sum() == 0.0
    assert adjacency[:, 2].sum() == 0.0
    assert adjacency[0, 0] > adjacency[0, 1]


def test_locked_kernel_rescue_is_unique_and_budget_limited() -> None:
    result = rescue(["a", "b", "c", "d"], ["b", "x", "a", "y"], budget=4, slots=2)
    assert result == ["a", "b", "x", "y"]
    assert len(result) == len(set(result)) == 4


def test_shortlist_reranking_respects_mask_and_depth() -> None:
    candidates = ["a", "b", "c", "d"]
    base = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    residual = np.asarray([0.0, 0.0, 1.0, 100.0], dtype=np.float32)
    ranking = rank_top_shortlist(
        base,
        residual,
        candidates,
        masked_ids={"b"},
        shortlist_depth=2,
        scale=0.3,
    )
    assert ranking == ["c", "a"]
    assert "b" not in ranking
    assert "d" not in ranking
