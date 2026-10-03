# Atlas consistency of the product-manifold field

## Why this is a mainline issue

The mathematical claim is one scalar compatibility field `F(r,e)` on `M_R × M_E`. A query-centred finite chart is only a numerical coordinate patch. Therefore independently solved local patches may be called sections of one field only when their values agree, up to numerical error, on overlaps.

The current v8 implementation has a sound global prior `F0` but solves the nonlinear deformation on query-dependent product charts. R2E and E2R use factor-swapped charts. This is computationally useful, but by itself it does not prove that the resulting local deformations are restrictions of one global `F*`.

The immediate audit is therefore not another ranking benchmark. For the same pair `(r,e)` represented as a node in both a reaction-centred and an enzyme-centred chart, measure

`d_section(r,e) = F_R*(r,e) - F_E*(r,e)`.

The corresponding prior defect must be approximately zero because both directions read the same scalar `F0`. A nonzero post-flow defect is an atlas-discretisation defect, not evidence for a second task-specific model.

## What not to do

Do not repair a section defect with:

- separate R2E and E2R rankers;
- a direction router or hard gate;
- a tuned mixture of the two scores;
- an arbitrary consistency-loss coefficient;
- a direction-specific positive-measure strength;
- external-label tuning.

Those would contradict the original object rather than improve its discretisation.

## Parameter-free geometric resolution

A natural global discretisation is a partition-of-unity variational field. Let `{U_a}` be a geometric atlas of product charts and `{psi_a}` a partition of unity subordinate to the atlas,

`psi_a >= 0`, `supp(psi_a) subset U_a`, and `sum_a psi_a = 1`.

Represent one global deformation, not independent final scores,

`F = F0 + sum_a psi_a u_a`,

where `u_a` is represented in the local numerical basis of chart `U_a`. Substitute this representation into the **same** global Charbonnier–Bregman energy already defining the method and optimize the local degrees of freedom jointly.

This has several desirable properties:

1. there is exactly one scalar `F` by construction;
2. chart overlap consistency requires no extra penalty weight;
3. R2E and E2R remain fibres of the same assembled function;
4. local charts remain computational devices rather than regions where the model is allowed to act;
5. the empirical positive measure is still one measure on the product space;
6. biological witnesses remain observations of the same object.

This is a discretisation change, not a new scorer.

## Promotion rule

Do not replace canonical v8 merely because partition-of-unity assembly is mathematically cleaner. Promotion requires:

- a measurable reduction of R2E/E2R section-overlap defect;
- exact preservation of the frozen no-leakage contract;
- no material regression on canonical R2E clean-development metrics;
- no new direction-specific hyperparameters;
- no use of the spent external retention set for selection.

Until a joint assembly is implemented and validated, v8 remains the canonical retrieval result and atlas gluing remains an active numerical-completeness problem.

## Preferred implementation candidate: one query-independent tensor-product atlas

A particularly clean implementation avoids solving and then reconciling thousands of query-centred sections. Build one coarse factor atlas from **fold-train support geometry only**:

- `m_R = ceil(sqrt(N_R_train))` reaction landmarks;
- `m_E = ceil(sqrt(N_E_train))` protein landmarks;
- geometric partitions `P_R` and `P_E` from full factor points to those landmarks.

For fold 0 this gives `90 × 345 = 31,050` product degrees of freedom from `8,010` train reactions and `118,806` train-positive proteins. This is small enough for the existing nonlinear variational solver.

Let `A` be the sparse fold-train reaction×protein positive relation. Its representation on the coarse product atlas is not built by ad-hoc pooling. It is the tensor-product restriction

`mu_coarse = P_R^T A P_E`.

After the same symmetric degree normalization, product-geometric mollification, and Charbonnier–Bregman solve, let `Delta C` be the solved coarse field deformation. The global correction is then the single function

`Delta F(r,e) = p_R(r)^T Delta C p_E(e)`.

Consequently,

- R2E uses `p_R(r)^T Delta C P_E^T`;
- E2R uses `P_R Delta C p_E(e)`;
- both are contractions of the same `Delta C` and therefore cannot disagree at a pair because of retrieval direction.

Landmarks must be selected from the intrinsic fold-train factor geometry, independent of query ranking and held-out labels. Their count follows the same `sqrt(N)` discretisation scale already used by the local method; no development sweep over landmark count is allowed for this experiment.

This construction should be viewed as a Galerkin/Nyström discretisation of the existing field, not as a low-rank score fusion model.
