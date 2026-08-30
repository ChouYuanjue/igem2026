"""Generic Fisher-merging core extracted from MIT-licensed FusionBench.

Upstream: https://github.com/tanganke/fusion_bench
Commit: 54c9e8c9d9621620c720452cd8533332a32d3689
Source: fusion_bench/method/fisher_merging/fisher_merging.py

Only framework-independent helpers are retained here; Catalyst supplies its
own retrieval loss/data adapter instead of depending on FusionBench's full
Hydra/Lightning/Transformers stack.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor, nn


def get_param_squared_gradients(
    model: nn.Module, param_names_to_merge: List[str]
) -> Dict[str, Tensor]:
    """Return squared gradients for selected parameters (FusionBench core)."""
    return {
        param_name: param_value.grad.detach() ** 2
        for param_name, param_value in model.state_dict(keep_vars=True).items()
        if param_name in param_names_to_merge
    }


def get_models_fisher_norm(
    models_to_merge_param_dict: dict,
    models_to_merge_fisher_weights_list: list,
) -> Tensor:
    """Return one global Fisher norm per model (FusionBench core)."""
    models_fisher_norm_dict = {}
    for param_name in models_to_merge_param_dict:
        models_fisher = torch.stack(
            [
                model_to_merge_fisher_weights[param_name]
                for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
            ],
            dim=0,
        )
        dims = [dim_idx for dim_idx in range(1, models_fisher.dim())]
        models_fisher_norm_dict[param_name] = torch.linalg.vector_norm(models_fisher, dim=dims)
    models_fisher_norm = torch.stack(list(models_fisher_norm_dict.values()), dim=1)
    return torch.norm(models_fisher_norm, dim=1)


def merging_with_fisher_weights(
    models_to_merge_param_dict: Dict[str, List[Tensor]],
    models_to_merge_fisher_weights_list: list,
    fisher_scaling_coefficients: torch.Tensor,
    normalize_fisher_weight: bool = True,
    minimal_fisher_weight: float = 1e-6,
) -> Dict[str, Tensor]:
    """Merge model parameters with diagonal Fisher weights (FusionBench core)."""
    merged_params = {}
    if normalize_fisher_weight:
        models_fisher_norm = get_models_fisher_norm(
            models_to_merge_param_dict=models_to_merge_param_dict,
            models_to_merge_fisher_weights_list=models_to_merge_fisher_weights_list,
        )

    for param_name, param_value_list in models_to_merge_param_dict.items():
        param_values = torch.stack(param_value_list, dim=0)
        models_to_merge_fisher_weights = (
            torch.stack(
                [
                    model_to_merge_fisher_weights[param_name]
                    for model_to_merge_fisher_weights in models_to_merge_fisher_weights_list
                ],
                dim=0,
            )
            + minimal_fisher_weight
        )
        reshaped_scaling_coefficients = fisher_scaling_coefficients.reshape(
            -1, *[1 for _ in range(param_values.dim() - 1)]
        ).to(param_values.device)
        if normalize_fisher_weight:
            inverse_norm = 1.0 / (models_fisher_norm + minimal_fisher_weight)
            normalized_models_fisher_norm = (inverse_norm / inverse_norm.sum()).reshape(
                -1, *[1 for _ in range(param_values.dim() - 1)]
            )
            reshaped_scaling_coefficients = (
                reshaped_scaling_coefficients * normalized_models_fisher_norm
            )
        numerator = (
            reshaped_scaling_coefficients * models_to_merge_fisher_weights * param_values
        ).sum(dim=0)
        denominator = (
            reshaped_scaling_coefficients * models_to_merge_fisher_weights
        ).sum(dim=0)
        merged_params[param_name] = numerator / denominator
    return merged_params
