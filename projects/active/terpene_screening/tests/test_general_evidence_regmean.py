from __future__ import annotations
import torch
from projects.active.terpene_screening.third_party.fusionbench_regmean import merging_with_regmean_weights


def test_regmean_follows_activation_geometry_for_linear_weight():
    params={'layer.weight':[torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])]}
    source={'layer':torch.diag(torch.tensor([10.,1.]))}
    target={'layer':torch.diag(torch.tensor([1.,10.]))}
    merged=merging_with_regmean_weights(params,[source,target])['layer.weight']
    assert torch.allclose(merged, torch.tensor([[10/11,10/11]]), atol=1e-5)


def test_source_weighting_moves_regmean_toward_source():
    params={'layer.weight':[torch.tensor([[1.,0.]]),torch.tensor([[0.,1.]])]}
    source={'layer':torch.eye(2)*5}; target={'layer':torch.eye(2)}
    merged=merging_with_regmean_weights(params,[source,target])['layer.weight']
    assert merged[0,0] > merged[0,1]


def test_diagonal_regmean_matches_exact_closed_form():
    params={'layer.weight':[torch.tensor([[2.,0.]]),torch.tensor([[0.,4.]])]}
    source={'layer':torch.tensor([[9.,3.],[3.,1.]])}
    target={'layer':torch.tensor([[1.,2.],[2.,4.]])}
    merged=merging_with_regmean_weights(
        params,[source,target],reduce_non_diagonal_ratio=0.0
    )['layer.weight']
    # dimension 0: (9*2 + 1*0)/(9+1); dimension 1: (1*0 + 4*4)/(1+4)
    assert torch.allclose(merged,torch.tensor([[1.8,3.2]]),atol=1e-6)
