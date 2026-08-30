"""Generic RegMean core extracted from MIT-licensed FusionBench.

Upstream: https://github.com/tanganke/fusion_bench
Commit: 54c9e8c9d9621620c720452cd8533332a32d3689
Source: fusion_bench/method/regmean/regmean.py

Catalyst-specific activation collection and source weighting live outside this module.
"""
from __future__ import annotations

import torch


def reduce_non_diagonal_elements(regmean_weights: torch.Tensor, reduce_non_diagonal_ratio: float):
    diag_weights = torch.diag(
        torch.ones(regmean_weights.shape[0], device=regmean_weights.device)
        - reduce_non_diagonal_ratio
    )
    non_diag_weights = torch.zeros_like(diag_weights).fill_(reduce_non_diagonal_ratio)
    return regmean_weights * (diag_weights + non_diag_weights)


def merging_with_regmean_weights(
    models_to_merge_param_dict: dict,
    models_to_merge_regmean_weights_list: list,
    reduce_non_diagonal_ratio: float = 1.0,
    weight_transpose: bool = True,
):
    """Merge Linear weights with RegMean; average unsupported parameters."""
    merged_params = {}
    for param_name, param_value_list in models_to_merge_param_dict.items():
        merged_by_regmean = False
        if param_name.endswith('.weight'):
            module_name = param_name[:-len('.weight')]
            if module_name in models_to_merge_regmean_weights_list[0]:
                param_multiplied_results = []
                module_regmean_weights_list = []
                for model_idx, model_weights in enumerate(models_to_merge_regmean_weights_list):
                    gram = reduce_non_diagonal_elements(
                        model_weights[module_name], reduce_non_diagonal_ratio
                    )
                    module_regmean_weights_list.append(gram)
                    param = param_value_list[model_idx]
                    param_multiplied_results.append(
                        gram @ (param.transpose(0, 1) if weight_transpose else param)
                    )
                sum_gram = sum(module_regmean_weights_list)
                sum_product = sum(param_multiplied_results)
                if reduce_non_diagonal_ratio == 0.0:
                    # For a diagonal Gram matrix, the RegMean solve is exactly elementwise.
                    # This avoids a dense O(d^3) pseudoinverse for Catalyst's 2115-D input.
                    diagonal = torch.diagonal(sum_gram).clamp_min(1e-12)
                    merged = sum_product / diagonal[:, None]
                else:
                    # pinv is numerically safer than inverse when activation covariance is low rank.
                    merged = torch.linalg.pinv(sum_gram) @ sum_product
                merged_params[param_name] = merged.transpose(0, 1) if weight_transpose else merged
                merged_by_regmean = True
        if not merged_by_regmean:
            merged_params[param_name] = torch.stack(param_value_list, dim=0).mean(dim=0)
    return merged_params
