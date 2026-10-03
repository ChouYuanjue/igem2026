from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.active.fibre.geometry.foundation import (
    local_fiber_distance_upper_bound,
    product_support_diagnostics,
)

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"results/fibre_geometric_foundation_synthetic_v1"


def main() -> None:
    # A genuinely many-to-many toy relation:
    #   G(r) = {r,-r}.
    # Both branches are 1-Lipschitz in Hausdorff distance, so the local fiber
    # theorem applies with L=1 while no global single-valued enzyme map exists.
    anchor_r=np.linspace(-1.0,1.0,41)
    omega=np.vstack([
        np.column_stack([anchor_r,anchor_r]),
        np.column_stack([anchor_r,-anchor_r]),
    ])
    spacing=float(anchor_r[1]-anchor_r[0])
    # Along either diagonal branch the product arclength between adjacent
    # anchors is sqrt(2)*spacing, so every true branch point is within half that.
    h=float(np.sqrt(2.0)*spacing/2.0)

    grid=np.linspace(-0.9,0.9,73)
    max_decomposition_error=0.0
    max_sampling_upper_violation=0.0
    max_sampling_lower_violation=0.0
    max_fiber_bound_violation=0.0
    true_relation_points=0
    true_relation_joint_max=0.0

    for r in grid:
        for e in grid:
            dr2=np.square(r-omega[:,0])
            de2=np.square(e-omega[:,1])
            joint=float(np.min(dr2+de2))
            mr=float(np.min(dr2))
            me=float(np.min(de2))
            defect=max(joint-mr-me,0.0)
            diag=product_support_diagnostics(defect,mr,me)
            max_decomposition_error=max(
                max_decomposition_error,
                abs(diag.joint_precedent_sq-joint),
            )

            # Exact distance to the union of the two infinite diagonal branches.
            d_gamma=float(min(abs(e-r),abs(e+r))/np.sqrt(2.0))
            d_omega=float(np.sqrt(joint))
            max_sampling_lower_violation=max(
                max_sampling_lower_violation,
                max(d_gamma-d_omega,0.0),
            )
            max_sampling_upper_violation=max(
                max_sampling_upper_violation,
                max(d_omega-(d_gamma+h),0.0),
            )

            fiber_distance=float(min(abs(e-r),abs(e+r)))
            bound=local_fiber_distance_upper_bound(
                defect,mr,me,hausdorff_lipschitz_constant=1.0,
            )
            max_fiber_bound_violation=max(
                max_fiber_bound_violation,
                max(fiber_distance-bound,0.0),
            )

        for e in (r,-r):
            if -0.9 <= e <= 0.9:
                dr2=np.square(r-omega[:,0])
                de2=np.square(e-omega[:,1])
                joint=float(np.min(dr2+de2))
                true_relation_joint_max=max(true_relation_joint_max,joint)
                true_relation_points+=1

    summary={
        "schema":"fibre-geometric-foundation-synthetic-v1",
        "relation":"Gamma={(r,e): e=r or e=-r}",
        "many_to_many":True,
        "single_global_function_exists":False,
        "hausdorff_lipschitz_constant":1.0,
        "anchor_count":int(len(omega)),
        "one_sided_coverage_radius":h,
        "checks":{
            "max_J_equals_Delta_plus_marginals_error":max_decomposition_error,
            "max_dGamma_le_dOmega_violation":max_sampling_lower_violation,
            "max_dOmega_le_dGamma_plus_h_violation":max_sampling_upper_violation,
            "max_local_fiber_bound_violation":max_fiber_bound_violation,
            "true_relation_point_count":true_relation_points,
            "max_J_on_true_relation":true_relation_joint_max,
            "h_squared":h*h,
        },
        "passed":bool(
            max_decomposition_error < 1e-12
            and max_sampling_lower_violation < 1e-12
            and max_sampling_upper_violation < 1e-12
            and max_fiber_bound_violation < 1e-12
            and true_relation_joint_max <= h*h + 1e-12
        ),
        "interpretation":(
            "algebraic/sampling sanity check only; it verifies the stated FIBRE "
            "bounds on a set-valued two-branch relation and is not biological evidence"
        ),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
