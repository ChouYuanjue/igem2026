from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .eram import EnzymaticModel, MultiLayerPerceptron, meanpooling


@dataclass(frozen=True)
class ExpertSpec:
    """One pluggable biochemical evidence source.

    The relational core does not interpret expert names. Each expert only owns an
    adapter from its native representation into the shared ERAM token space.
    """

    input_dim: int


@dataclass
class ExpertEvidence:
    values: torch.Tensor
    available: torch.Tensor | None = None


class ExpertAdapter(nn.Module):
    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = (
            nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, d_model))
            if input_dim > 1
            else nn.Linear(input_dim, d_model)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            values = values.unsqueeze(1)
        if values.ndim != 3:
            raise ValueError("expert values must have shape [B,D] or [B,L,D]")
        return self.proj(values)


class FibreRelationalModel(nn.Module):
    """ERAM relational core with a variable-length pluggable evidence set.

    The upstream ERAM transformer and relation geometry are kept intact. BRIDGE
    changes only the enzyme-side evidence contract: the mandatory broad protein
    representation is followed by zero or more independently registered evidence
    token streams. Available streams are concatenated before ERAM's original
    reaction-conditioned cross-attention. No router, expert selection, residual
    score, or fixed expert list is used.
    """

    def __init__(
        self,
        *,
        protein_input_dim: int = 1152,
        molecule_input_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 2048,
        hidden_dim: int = 1024,
        out_dim: int = 512,
        experts: Mapping[str, ExpertSpec] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.core = EnzymaticModel(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            require_attn=False,
        )
        # Keep the official ERAM MLP shape, changing only native input widths.
        self.core.enzyme_mlp1 = MultiLayerPerceptron(
            input_dim=protein_input_dim,
            hidden_dims=[1024, d_model],
        )
        self.core.molecule_mlp1 = MultiLayerPerceptron(
            input_dim=molecule_input_dim,
            hidden_dims=[1024, d_model],
        )
        self.expert_adapters = nn.ModuleDict()
        for name, spec in (experts or {}).items():
            self.register_expert(name, spec)

    def register_expert(self, name: str, spec: ExpertSpec) -> None:
        if not name or "." in name:
            raise ValueError("expert name must be a non-empty module-safe key")
        if name in self.expert_adapters:
            raise ValueError(f"expert already registered: {name}")
        self.expert_adapters[name] = ExpertAdapter(spec.input_dim, self.d_model)

    @staticmethod
    def _as_tokens(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 2:
            return values.unsqueeze(1)
        if values.ndim != 3:
            raise ValueError("base protein values must have shape [B,D] or [B,L,D]")
        return values

    @staticmethod
    def _mask(
        batch: int,
        length: int,
        *,
        device: torch.device,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return torch.zeros((batch, length), dtype=torch.bool, device=device)
        padding_mask = padding_mask.to(device=device, dtype=torch.bool)
        if padding_mask.shape != (batch, length):
            raise ValueError("padding mask shape does not match token sequence")
        return padding_mask

    def _pack_evidence(
        self,
        base_values: torch.Tensor,
        base_padding_mask: torch.Tensor | None,
        evidence: Mapping[str, ExpertEvidence] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_values = self._as_tokens(base_values)
        base_tokens = self.core.enzyme_mlp1(base_values)
        batch, base_len, _ = base_tokens.shape
        tokens = [base_tokens]
        masks = [
            self._mask(
                batch,
                base_len,
                device=base_tokens.device,
                padding_mask=base_padding_mask,
            )
        ]
        for name, item in (evidence or {}).items():
            if name not in self.expert_adapters:
                raise KeyError(f"unregistered expert: {name}")
            expert_tokens = self.expert_adapters[name](item.values)
            if expert_tokens.shape[0] != batch:
                raise ValueError(f"expert batch does not align: {name}")
            length = int(expert_tokens.shape[1])
            if item.available is None:
                unavailable = torch.zeros((batch, length), dtype=torch.bool, device=base_tokens.device)
            else:
                available = item.available.to(device=base_tokens.device, dtype=torch.bool)
                if available.ndim == 1:
                    if available.shape[0] != batch:
                        raise ValueError(f"expert availability does not align: {name}")
                    available = available[:, None].expand(batch, length)
                elif available.shape != (batch, length):
                    raise ValueError(f"expert availability does not align: {name}")
                unavailable = ~available
            tokens.append(expert_tokens)
            masks.append(unavailable)
        return torch.cat(tokens, dim=1), torch.cat(masks, dim=1)

    def forward(
        self,
        *,
        protein: torch.Tensor,
        reactant: torch.Tensor,
        product: torch.Tensor,
        protein_padding_mask: torch.Tensor | None = None,
        reactant_padding_mask: torch.Tensor | None = None,
        product_padding_mask: torch.Tensor | None = None,
        evidence: Mapping[str, ExpertEvidence] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enzyme_emb, evidence_mask = self._pack_evidence(
            protein,
            protein_padding_mask,
            evidence,
        )
        reactant_emb = self.core.molecule_mlp1(reactant)
        product_emb = self.core.molecule_mlp1(product)
        batch = int(reactant_emb.shape[0])
        reactant_padding_mask = self._mask(
            batch,
            int(reactant_emb.shape[1]),
            device=reactant_emb.device,
            padding_mask=reactant_padding_mask,
        )
        product_padding_mask = self._mask(
            batch,
            int(product_emb.shape[1]),
            device=product_emb.device,
            padding_mask=product_padding_mask,
        )

        for layer in self.core.layers:
            reactant_emb, enzyme_emb = layer(
                src=reactant_emb,
                tgt=enzyme_emb,
                src_key_padding_mask=reactant_padding_mask,
                tgt_key_padding_mask=evidence_mask,
                memory_key_padding_mask=reactant_padding_mask,
            )
            product_emb = layer(
                src=product_emb,
                tgt=None,
                src_key_padding_mask=product_padding_mask,
            )

        enzyme_emb = self.core.enzyme_mlp2(enzyme_emb)
        reactant_emb = self.core.molecule_mlp2(reactant_emb)
        product_emb = self.core.molecule_mlp2(product_emb)
        enzyme_emb = meanpooling(enzyme_emb, evidence_mask)
        reactant_emb = meanpooling(reactant_emb, reactant_padding_mask)
        product_emb = meanpooling(product_emb, product_padding_mask)
        return reactant_emb, enzyme_emb, product_emb

    def relation_distance(self, **kwargs: object) -> torch.Tensor:
        reactant, enzyme, product = self.forward(**kwargs)
        return torch.linalg.vector_norm(reactant + enzyme - product, dim=-1)

    def score(self, **kwargs: object) -> torch.Tensor:
        return -self.relation_distance(**kwargs)
