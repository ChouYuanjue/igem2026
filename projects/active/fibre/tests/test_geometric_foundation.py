import math

import numpy as np

from projects.active.fibre.geometry.foundation import (
    defect_stability_bound,
    local_fiber_distance_upper_bound,
    product_support_diagnostics,
    relation_sampling_distance_bounds,
)


def test_product_support_decomposition_reconstructs_joint_distance():
    d=product_support_diagnostics(0.25,1.0,2.75)
    assert d.joint_precedent_sq==4.0
    assert d.joint_precedent_distance==2.0


def test_local_set_valued_fiber_bound_matches_lipschitz_graph_case():
    # Gamma={(r,e):e=r}; G is 1-Lipschitz.  z=(0,1) has product
    # distance 1/sqrt(2) to Gamma and fiber distance 1.
    d_gamma=1.0/math.sqrt(2.0)
    bound=math.sqrt(2.0)*d_gamma
    # Choose an accepted joint precedent exactly at the nearest Gamma point so
    # J=d_gamma^2 and marginal novelty is zero only for this synthetic algebra.
    actual=local_fiber_distance_upper_bound(
        d_gamma*d_gamma,0.0,0.0,hausdorff_lipschitz_constant=1.0
    )
    assert math.isclose(actual,bound,rel_tol=0,abs_tol=1e-12)
    assert math.isclose(actual,1.0,rel_tol=0,abs_tol=1e-12)


def test_relation_sampling_bound_is_one_sided_when_omega_subset_gamma():
    assert relation_sampling_distance_bounds(
        0.3,one_sided_coverage_radius=0.2
    )==(0.3,0.5)


def test_defect_stability_bound_from_factor_squared_distance_errors():
    assert defect_stability_bound(0.01,0.02)==0.06


def test_small_defect_alone_does_not_imply_small_joint_distance():
    near=product_support_diagnostics(0.0,0.01,0.01)
    far=product_support_diagnostics(0.0,100.0,100.0)
    assert near.joint_precedent_distance < 0.2
    assert far.joint_precedent_distance > 14.0


def test_invalid_negative_support_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        product_support_diagnostics(0.0,-1.0,0.0)
