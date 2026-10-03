from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from reproducibility.bime_rank.support.evaluate_multi_expert_protocol_comparison import (
    DirectionalMultiExpertDualTower,
    MultiExpertConfig,
)


@dataclass(frozen=True)
class HeterogeneousFibreConfig:
    base: MultiExpertConfig
    enzgfm_input_dim: int
    reaction_center_input_dim: int
    specialist_hidden_dim: int = 256
    specialist_dim: int = 64
    specialist_names: tuple[str, ...] = (
        "enzgfm",
        "reaction_center",
        "clipzyme",
        "seed_context",
    )


class _SpecialistTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), p=2, dim=-1)


def availability_aware_weights(
    query_logits: torch.Tensor,
    specialist_available: torch.Tensor,
) -> torch.Tensor:
    """Pairwise-normalised mixture weights with exact missing-neutral fallback.

    query_logits has shape [N, 1 + S]; specialist_available has shape [N, S].
    The base BRIDGE channel (column 0) is always available. Missing specialist
    channels receive exactly zero mass and the remaining weights renormalise.
    """
    if query_logits.ndim != 2 or specialist_available.ndim != 2:
        raise ValueError("query_logits and specialist_available must be rank-2")
    if query_logits.shape[0] != specialist_available.shape[0]:
        raise ValueError("batch sizes must align")
    if query_logits.shape[1] != specialist_available.shape[1] + 1:
        raise ValueError("query_logits require one base column plus specialist columns")
    mask = torch.cat(
        [
            torch.ones(
                (len(specialist_available), 1),
                dtype=torch.bool,
                device=specialist_available.device,
            ),
            specialist_available.bool(),
        ],
        dim=1,
    )
    masked = query_logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(masked, dim=-1)


class HeterogeneousConditionalFibre(nn.Module):
    """BRIDGE conditional modes plus real heterogeneous evidence channels.

    The base channel is the promoted BRIDGE conditional-mode model. Two learned
    specialists use existing heterogeneous assets directly:
      * EnzGFM protein view against the reaction view;
      * atom-mapped reaction-center view against the ESM-C protein view.
    Two fixed pairwise channels accept already-computed CLIPZyme structure and
    known-positive seed-context scores. They are availability-aware and are
    exactly ignored when absent.

    The final router is query-conditioned in both directions. It consumes the
    base model's eight learned gate coordinates, so heterogeneous evidence is
    integrated into the same conditional interaction object rather than added
    as an external post-hoc route.
    """

    def __init__(self, config: HeterogeneousFibreConfig) -> None:
        super().__init__()
        if tuple(config.specialist_names) != (
            "enzgfm",
            "reaction_center",
            "clipzyme",
            "seed_context",
        ):
            raise ValueError("specialist_names must preserve the frozen channel order")
        self.config = config
        self.base = DirectionalMultiExpertDualTower(config.base)

        self.enzgfm_reaction = _SpecialistTower(
            config.base.reaction_input_dim,
            config.specialist_hidden_dim,
            config.specialist_dim,
        )
        self.enzgfm_protein = _SpecialistTower(
            config.enzgfm_input_dim,
            config.specialist_hidden_dim,
            config.specialist_dim,
        )
        self.center_reaction = _SpecialistTower(
            config.reaction_center_input_dim,
            config.specialist_hidden_dim,
            config.specialist_dim,
        )
        self.center_protein = _SpecialistTower(
            config.base.protein_input_dim,
            config.specialist_hidden_dim,
            config.specialist_dim,
        )

        n_channels = 1 + len(config.specialist_names)
        self.r2e_router = nn.Linear(config.base.n_experts, n_channels)
        self.e2r_router = nn.Linear(config.base.n_experts, n_channels)

        # Give the base BRIDGE channel half of the initial mass when all four
        # specialists are available. Missing specialists are still removed
        # exactly by the availability mask and the remaining mass renormalises.
        with torch.no_grad():
            self.r2e_router.weight.zero_()
            self.e2r_router.weight.zero_()
            self.r2e_router.bias.zero_()
            self.e2r_router.bias.zero_()
            self.r2e_router.bias[0] = torch.log(torch.tensor(4.0))
            self.e2r_router.bias[0] = torch.log(torch.tensor(4.0))

    def score_pairs(
        self,
        *,
        protein_values: torch.Tensor,
        reaction_values: torch.Tensor,
        enzgfm_values: torch.Tensor,
        reaction_center_values: torch.Tensor,
        enzgfm_available: torch.Tensor | None = None,
        reaction_center_available: torch.Tensor | None = None,
        clipzyme_scores: torch.Tensor | None = None,
        clipzyme_available: torch.Tensor | None = None,
        seed_context_scores: torch.Tensor | None = None,
        seed_context_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        n = len(protein_values)
        if not (
            len(reaction_values) == n
            and len(enzgfm_values) == n
            and len(reaction_center_values) == n
        ):
            raise ValueError("all aligned pair inputs must have equal batch length")

        base_r2e, base_e2r, base_diag = self.base.score_pairs(
            protein_values,
            reaction_values,
        )

        enz_score = (
            self.enzgfm_reaction(reaction_values)
            * self.enzgfm_protein(enzgfm_values)
        ).sum(dim=-1)
        center_score = (
            self.center_reaction(reaction_center_values)
            * self.center_protein(protein_values)
        ).sum(dim=-1)

        def _score_or_zero(value: torch.Tensor | None) -> torch.Tensor:
            if value is None:
                return torch.zeros(n, dtype=base_r2e.dtype, device=base_r2e.device)
            value = value.to(device=base_r2e.device, dtype=base_r2e.dtype)
            if value.shape != (n,):
                raise ValueError("fixed specialist scores must have shape [batch]")
            return value

        def _availability(
            value: torch.Tensor | None,
            *,
            default: bool,
        ) -> torch.Tensor:
            if value is None:
                return torch.full(
                    (n,),
                    default,
                    dtype=torch.bool,
                    device=base_r2e.device,
                )
            value = value.to(device=base_r2e.device, dtype=torch.bool)
            if value.shape != (n,):
                raise ValueError("specialist availability must have shape [batch]")
            return value

        clip_score = _score_or_zero(clipzyme_scores)
        seed_score = _score_or_zero(seed_context_scores)
        availability = torch.stack(
            [
                _availability(enzgfm_available, default=True),
                _availability(reaction_center_available, default=True),
                _availability(clipzyme_available, default=clipzyme_scores is not None),
                _availability(
                    seed_context_available,
                    default=seed_context_scores is not None,
                ),
            ],
            dim=1,
        )

        specialist_scores = torch.stack(
            [enz_score, center_score, clip_score, seed_score],
            dim=1,
        )
        r2e_logits = self.r2e_router(base_diag["reaction_gates"])
        e2r_logits = self.e2r_router(base_diag["protein_gates"])
        r2e_weights = availability_aware_weights(r2e_logits, availability)
        e2r_weights = availability_aware_weights(e2r_logits, availability)

        r2e_channels = torch.cat([base_r2e[:, None], specialist_scores], dim=1)
        e2r_channels = torch.cat([base_e2r[:, None], specialist_scores], dim=1)
        r2e = (r2e_weights * r2e_channels).sum(dim=1)
        e2r = (e2r_weights * e2r_channels).sum(dim=1)

        diagnostics = {
            **base_diag,
            "specialist_scores": specialist_scores,
            "specialist_available": availability,
            "r2e_channel_weights": r2e_weights,
            "e2r_channel_weights": e2r_weights,
        }
        return r2e, e2r, diagnostics

    def encode_protein_side(
        self,
        protein_values: torch.Tensor,
        enzgfm_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_embedding, experts, gates = self.base.encode_proteins(protein_values)
        return {
            "global": global_embedding,
            "experts": experts,
            "gates": gates,
            "enzgfm": self.enzgfm_protein(enzgfm_values),
            "center": self.center_protein(protein_values),
        }

    def encode_reaction_side(
        self,
        reaction_values: torch.Tensor,
        reaction_center_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        global_embedding, experts, gates = self.base.encode_reactions(reaction_values)
        return {
            "global": global_embedding,
            "experts": experts,
            "gates": gates,
            "enzgfm": self.enzgfm_reaction(reaction_values),
            "center": self.center_reaction(reaction_center_values),
        }

    def score_encoded_cross(
        self,
        *,
        proteins: dict[str, torch.Tensor],
        reactions: dict[str, torch.Tensor],
        protein_enzgfm_available: torch.Tensor | None = None,
        reaction_center_available: torch.Tensor | None = None,
        clipzyme_scores: torch.Tensor | None = None,
        clipzyme_available: torch.Tensor | None = None,
        seed_context_scores: torch.Tensor | None = None,
        seed_context_available: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Score a cross product from cached molecular encodings.

        This is the inference path for large candidate universes: candidate
        towers are encoded once, then many query rows can be scored with matrix
        products without re-running 3k-dimensional input MLPs per pair.
        """
        nr = int(reactions["global"].shape[0])
        np_ = int(proteins["global"].shape[0])
        global_scores = reactions["global"] @ proteins["global"].T
        expert_scores = torch.einsum(
            "rhd,phd->rph",
            reactions["experts"],
            proteins["experts"],
        )
        r2e_mix = torch.sigmoid(self.base.r2e_mix_logit)
        e2r_mix = torch.sigmoid(self.base.e2r_mix_logit)
        base_r2e = (1 - r2e_mix) * global_scores + r2e_mix * torch.einsum(
            "rh,rph->rp",
            reactions["gates"],
            expert_scores,
        )
        base_e2r = (1 - e2r_mix) * global_scores + e2r_mix * torch.einsum(
            "ph,rph->rp",
            proteins["gates"],
            expert_scores,
        )
        enz_scores = reactions["enzgfm"] @ proteins["enzgfm"].T
        center_scores = reactions["center"] @ proteins["center"].T

        dtype = base_r2e.dtype
        device = base_r2e.device
        clipzyme_supplied = clipzyme_scores is not None
        seed_context_supplied = seed_context_scores is not None
        if clipzyme_scores is None:
            clipzyme_scores = torch.zeros((nr, np_), dtype=dtype, device=device)
        else:
            clipzyme_scores = clipzyme_scores.to(device=device, dtype=dtype)
        if seed_context_scores is None:
            seed_context_scores = torch.zeros((nr, np_), dtype=dtype, device=device)
        else:
            seed_context_scores = seed_context_scores.to(device=device, dtype=dtype)
        for name, value in (
            ("clipzyme_scores", clipzyme_scores),
            ("seed_context_scores", seed_context_scores),
        ):
            if value.shape != (nr, np_):
                raise ValueError(f"{name} must have shape [reactions, proteins]")

        if protein_enzgfm_available is None:
            protein_enzgfm_available = torch.ones(np_, dtype=torch.bool, device=device)
        else:
            protein_enzgfm_available = protein_enzgfm_available.to(
                device=device, dtype=torch.bool
            )
        if reaction_center_available is None:
            reaction_center_available = torch.ones(nr, dtype=torch.bool, device=device)
        else:
            reaction_center_available = reaction_center_available.to(
                device=device, dtype=torch.bool
            )
        enz_available = protein_enzgfm_available[None, :].expand(nr, np_)
        center_available = reaction_center_available[:, None].expand(nr, np_)

        def _pair_availability(
            value: torch.Tensor | None,
            default: bool,
        ) -> torch.Tensor:
            if value is None:
                return torch.full(
                    (nr, np_),
                    default,
                    dtype=torch.bool,
                    device=device,
                )
            value = value.to(device=device, dtype=torch.bool)
            if value.shape != (nr, np_):
                raise ValueError("pair availability must match cross-product shape")
            return value

        availability = torch.stack(
            [
                enz_available,
                center_available,
                _pair_availability(
                    clipzyme_available,
                    default=clipzyme_supplied,
                ),
                _pair_availability(
                    seed_context_available,
                    default=seed_context_supplied,
                ),
            ],
            dim=-1,
        )
        specialist_scores = torch.stack(
            [enz_scores, center_scores, clipzyme_scores, seed_context_scores],
            dim=-1,
        )
        r2e_logits = self.r2e_router(reactions["gates"])[:, None, :].expand(
            nr, np_, -1
        )
        e2r_logits = self.e2r_router(proteins["gates"])[None, :, :].expand(
            nr, np_, -1
        )
        flat_available = availability.reshape(-1, availability.shape[-1])
        r2e_weights = availability_aware_weights(
            r2e_logits.reshape(-1, r2e_logits.shape[-1]),
            flat_available,
        ).reshape(nr, np_, -1)
        e2r_weights = availability_aware_weights(
            e2r_logits.reshape(-1, e2r_logits.shape[-1]),
            flat_available,
        ).reshape(nr, np_, -1)
        r2e_channels = torch.cat([base_r2e[..., None], specialist_scores], dim=-1)
        e2r_channels = torch.cat([base_e2r[..., None], specialist_scores], dim=-1)
        r2e = (r2e_weights * r2e_channels).sum(dim=-1)
        e2r = (e2r_weights * e2r_channels).sum(dim=-1)
        return r2e, e2r, {
            "base_r2e": base_r2e,
            "base_e2r": base_e2r,
            "specialist_scores": specialist_scores,
            "specialist_available": availability,
            "r2e_channel_weights": r2e_weights,
            "e2r_channel_weights": e2r_weights,
        }
