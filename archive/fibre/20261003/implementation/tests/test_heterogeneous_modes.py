from __future__ import annotations

import numpy as np
import torch

from projects.active.fibre.kernel.heterogeneous_modes import (
    HeterogeneousConditionalFibre,
    HeterogeneousFibreConfig,
    availability_aware_weights,
)
from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    MultiExpertConfig,
)
from projects.active.fibre.runtime.heterogeneous_fibre import (
    FibreHCMRuntime,
    load_fibre_hcm_checkpoint,
)


def _config() -> HeterogeneousFibreConfig:
    return HeterogeneousFibreConfig(
        base=MultiExpertConfig(
            protein_input_dim=5,
            reaction_input_dim=7,
            hidden_dim=11,
            global_dim=4,
            n_experts=4,
            expert_dim=3,
            dropout=0.0,
            gate_temperature=1.0,
            expert_mix_init=0.5,
        ),
        enzgfm_input_dim=6,
        reaction_center_input_dim=3,
        specialist_hidden_dim=8,
        specialist_dim=4,
    )


def test_availability_aware_weights_zero_missing_channels_and_renormalise() -> None:
    logits = torch.tensor([[0.0, 2.0, 3.0], [1.0, -1.0, 4.0]])
    available = torch.tensor([[True, False], [False, False]])
    weights = availability_aware_weights(logits, available)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert float(weights[0, 2]) == 0.0
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0, 0.0]))


def test_missing_all_specialists_is_exact_base_fallback() -> None:
    torch.manual_seed(11)
    model = HeterogeneousConditionalFibre(_config()).eval()
    n = 8
    protein = torch.randn(n, 5)
    reaction = torch.randn(n, 7)
    enzgfm = torch.randn(n, 6)
    center = torch.randn(n, 3)
    unavailable = torch.zeros(n, dtype=torch.bool)

    r2e, e2r, diag = model.score_pairs(
        protein_values=protein,
        reaction_values=reaction,
        enzgfm_values=enzgfm,
        reaction_center_values=center,
        enzgfm_available=unavailable,
        reaction_center_available=unavailable,
        clipzyme_available=unavailable,
        seed_context_available=unavailable,
    )
    base_r2e, base_e2r, _ = model.base.score_pairs(protein, reaction)
    torch.testing.assert_close(r2e, base_r2e)
    torch.testing.assert_close(e2r, base_e2r)
    torch.testing.assert_close(
        diag["r2e_channel_weights"][:, 0],
        torch.ones(n),
    )
    torch.testing.assert_close(
        diag["e2r_channel_weights"][:, 0],
        torch.ones(n),
    )


def test_available_fixed_expert_can_change_score_without_affecting_missing_pairs() -> None:
    torch.manual_seed(13)
    model = HeterogeneousConditionalFibre(_config()).eval()
    with torch.no_grad():
        model.r2e_router.bias.zero_()
        model.e2r_router.bias.zero_()
    n = 5
    protein = torch.randn(n, 5)
    reaction = torch.randn(n, 7)
    enzgfm = torch.randn(n, 6)
    center = torch.randn(n, 3)
    none = torch.zeros(n, dtype=torch.bool)
    clip_available = torch.tensor([True, False, True, False, True])
    clip_scores = torch.tensor([0.9, 0.9, -0.8, -0.8, 0.4])

    r2e, _, _ = model.score_pairs(
        protein_values=protein,
        reaction_values=reaction,
        enzgfm_values=enzgfm,
        reaction_center_values=center,
        enzgfm_available=none,
        reaction_center_available=none,
        clipzyme_scores=clip_scores,
        clipzyme_available=clip_available,
        seed_context_available=none,
    )
    base_r2e, _, _ = model.base.score_pairs(protein, reaction)
    torch.testing.assert_close(r2e[~clip_available], base_r2e[~clip_available])
    assert not torch.allclose(r2e[clip_available], base_r2e[clip_available])


def test_cached_cross_scoring_matches_aligned_pair_scoring_on_diagonal() -> None:
    torch.manual_seed(17)
    model = HeterogeneousConditionalFibre(_config()).eval()
    n = 6
    protein = torch.randn(n, 5)
    reaction = torch.randn(n, 7)
    enzgfm = torch.randn(n, 6)
    center = torch.randn(n, 3)
    clip = torch.linspace(-0.5, 0.7, n)
    clip_available = torch.tensor([True, False, True, True, False, True])
    pair_r2e, pair_e2r, _ = model.score_pairs(
        protein_values=protein,
        reaction_values=reaction,
        enzgfm_values=enzgfm,
        reaction_center_values=center,
        clipzyme_scores=clip,
        clipzyme_available=clip_available,
        seed_context_available=torch.zeros(n, dtype=torch.bool),
    )
    proteins = model.encode_protein_side(protein, enzgfm)
    reactions = model.encode_reaction_side(reaction, center)
    clip_matrix = torch.zeros(n, n)
    clip_mask = torch.zeros(n, n, dtype=torch.bool)
    idx = torch.arange(n)
    clip_matrix[idx, idx] = clip
    clip_mask[idx, idx] = clip_available
    cross_r2e, cross_e2r, _ = model.score_encoded_cross(
        proteins=proteins,
        reactions=reactions,
        clipzyme_scores=clip_matrix,
        clipzyme_available=clip_mask,
        seed_context_available=torch.zeros(n, n, dtype=torch.bool),
    )
    torch.testing.assert_close(pair_r2e, cross_r2e.diag())
    torch.testing.assert_close(pair_e2r, cross_e2r.diag())


def test_checkpoint_loader_roundtrip(tmp_path) -> None:
    torch.manual_seed(19)
    model = HeterogeneousConditionalFibre(_config()).eval()
    checkpoint = tmp_path / "fibre_hcm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "protein_input_dim": 5,
                "reaction_input_dim": 7,
                "enzgfm_input_dim": 6,
                "reaction_center_input_dim": 3,
                "hidden_dim": 11,
                "global_dim": 4,
                "n_experts": 4,
                "expert_dim": 3,
                "dropout": 0.0,
                "gate_temperature": 1.0,
                "expert_mix_init": 0.5,
                "specialist_hidden_dim": 8,
                "specialist_dim": 4,
            },
        },
        checkpoint,
    )
    restored = load_fibre_hcm_checkpoint(checkpoint)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_cached_cross_scoring_with_no_fixed_specialists_is_exact_base() -> None:
    torch.manual_seed(23)
    model = HeterogeneousConditionalFibre(_config()).eval()
    protein = torch.randn(4, 5)
    reaction = torch.randn(3, 7)
    enzgfm = torch.randn(4, 6)
    center = torch.randn(3, 3)
    proteins = model.encode_protein_side(protein, enzgfm)
    reactions = model.encode_reaction_side(reaction, center)
    off_p = torch.zeros(4, dtype=torch.bool)
    off_r = torch.zeros(3, dtype=torch.bool)
    r2e, e2r, _ = model.score_encoded_cross(
        proteins=proteins,
        reactions=reactions,
        protein_enzgfm_available=off_p,
        reaction_center_available=off_r,
    )
    global_scores = reactions["global"] @ proteins["global"].T
    expert_scores = torch.einsum(
        "rhd,phd->rph",
        reactions["experts"],
        proteins["experts"],
    )
    expected_r2e = (
        (1 - torch.sigmoid(model.base.r2e_mix_logit)) * global_scores
        + torch.sigmoid(model.base.r2e_mix_logit)
        * torch.einsum("rh,rph->rp", reactions["gates"], expert_scores)
    )
    expected_e2r = (
        (1 - torch.sigmoid(model.base.e2r_mix_logit)) * global_scores
        + torch.sigmoid(model.base.e2r_mix_logit)
        * torch.einsum("ph,rph->rp", proteins["gates"], expert_scores)
    )
    torch.testing.assert_close(r2e, expected_r2e)
    torch.testing.assert_close(e2r, expected_e2r)


def test_runtime_seed_cosine_uses_best_known_positive() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    score = FibreHCMRuntime._max_seed_cosine(
        matrix,
        np.asarray([0, 1, 2]),
        np.asarray([0, 1]),
    )
    np.testing.assert_allclose(score[:2], np.ones(2), atol=1e-6)
    np.testing.assert_allclose(score[2], 1 / np.sqrt(2), atol=1e-6)
