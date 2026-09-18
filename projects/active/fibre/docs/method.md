# FIBRE — Field Inference for Bidirectional Reaction–Enzyme Retrieval

FIBRE is the current method identity. Its mathematical core is **sparse correspondence geometry** on the reaction–enzyme product manifold; the name refers to the two retrieval directions as fibres of one shared compatibility field.

## One sentence

**With sparse positive enzyme–reaction labels but rich, incomplete molecular observations, we build intrinsic reaction and enzyme manifolds and score a pair by the excess product-geodesic cost of explaining both coordinates with one shared known biochemical precedent rather than with unrelated marginal neighbours.**

Formally, let

\[
M = M_R \times M_E,
\qquad
\Omega \subset M_R\times M_E
\]

be the reaction–enzyme product manifold and the sparse set of known positive biochemical correspondences.

The factor geometries are label-free. M_R is built from global reaction chemistry; M_E is a partially observed multiresolution molecular-state manifold. Each factor geodesic is normalized by its own intrinsic median local edge length before forming the canonical Cartesian product.

For a candidate pair (r,e),

\[
J_\Omega(r,e)
=
\min_{(r_i,e_i)\in\Omega}
\left[
d_R(r,r_i)^2+d_E(e,e_i)^2
\right],
\]

while the marginal familiarity terms are

\[
m_R(r)=\min_{r_i\in\Omega_R}d_R(r,r_i)^2,
\qquad
m_E(e)=\min_{e_i\in\Omega_E}d_E(e,e_i)^2.
\]

The **correspondence defect**

\[
\boxed{
\Delta_\Omega(r,e)
=
J_\Omega(r,e)-m_R(r)-m_E(e)
}
\]

is the extra geometric price paid when the reaction and enzyme must be explained by the **same** known biochemical precedent instead of independently resembling unrelated observed reactions and proteins.

The compatibility field is simply

\[
\boxed{F(r,e)=-\Delta_\Omega(r,e).}
\]

R2E and E2R are rows and columns of this one scalar field. There is no direction-specific model.

## Why this matches the data-scarcity problem

The scarce resource is not molecular information; it is experimentally established **pairing information**.

- Every protein has a global sequence coordinate.
- Where available, pocket-local sequence, whole-structure 3Di, pocket-3Di, pocket-OT and family-applicable catalytic-motif context refine the same protein manifold.
- Missing views are missing observations, never negative evidence.
- Reaction chemistry is observed independently of pair labels.
- Known biochemical pairs only define the sparse correspondence set Omega; no synthetic negatives are required.

Thus unlabeled molecules still contribute to geometry even when no enzyme–reaction pair is known for them.

## What “use all available information” means

It does **not** mean every data source must be forced into one score.

An observation enters the ranking geometry only when it has a mathematically coherent, empirically non-destructive role in the factor manifold. Otherwise it remains a directly inspectable local/mechanistic observable attached to the same molecular object.

Current status:

- Protein multiresolution observations form a stable ranking geometry and are canonical.
- Global reaction chemistry forms the canonical reaction atlas.
- Atom-mapped reaction-center Wasserstein and explicit transition-token geometry are valid catalytic-local observations, but forcing them into the reaction factor metric causes a statistically clear E2R regression on the current internal split. Therefore they remain tangent/mechanistic witnesses rather than a second scoring expert.

This is a stricter interpretation of “use all information”: **nothing scientifically useful is discarded, but no information source is allowed to deform the mathematical object merely because it exists.**

## Exact seed update

Adding one newly validated pair (r*,e*) requires no retraining.

\[
J' = \min\{J,\ d_R(\cdot,r^*)^2+d_E(\cdot,e^*)^2\},
\]

\[
m_R'=\min\{m_R,\ d_R(\cdot,r^*)^2\},
\qquad
m_E'=\min\{m_E,\ d_E(\cdot,e^*)^2\}.
\]

Then Delta' = J' - m_R' - m_E'.

The implementation in `projects/active/fibre/geometry/correspondence.py` is exactly equal, element by element, to rebuilding the field from scratch after adding the seed.

## Current internal-only evidence

Canonical development result:

results/terpene_product_correspondence_dev_v1/

Using global reaction geometry × multiresolution protein geometry:

- E2R: MRR ≈ 0.0723, median best-positive rank 34, Hit@20 ≈ 0.391.
- R2E: MRR ≈ 0.0747, median best-positive rank 115, Hit@20 ≈ 0.241.
- Every development cell is double-cold with zero train/test protein and reaction overlap.

Ablation:

- Replacing the multiresolution protein manifold with the global-only protein geometry sharply reduces R2E MRR (paired delta ≈ +0.0486 for multiresolution, 95% bootstrap CI [0.0143, 0.0906]).
- Forcing catalytic reaction-center geometry into the factor metric materially hurts E2R relative to the canonical protein-multiresolution field (paired MRR delta ≈ -0.0125, 95% CI [-0.0215, -0.00324]).

Seed audit:

results/terpene_correspondence_seed_update_audit_v1/

Across leave-one-seed updates on all nine development cells, excluding the seed pair itself:

- E2R: 553 improved / 1325 tied / 181 worsened comparisons; mean reciprocal-rank delta ≈ +0.0538.
- R2E: 371 improved / 884 tied / 55 worsened comparisons; mean reciprocal-rank delta ≈ +0.0302.
- Incremental-update parity against full reconstruction: max absolute difference = 0.

These are development diagnostics only and are not a reason to reopen the spent external-retention protocol.

## Relationship to the broad Rhea mainline

The broad Rhea v8 result remains frozen and should not be rewritten retroactively as this exact operator.

Both lines share the same scientific object:

1. label-free factor geometry;
2. sparse positive correspondence on M_R × M_E;
3. one scalar compatibility field;
4. R2E/E2R as fibres of that field;
5. missing information is absence of observation, not negative evidence.

The broad benchmark currently uses a smooth variational/transport realization; the compact MARTS application admits a global product-geodesic correspondence field. A future unification is acceptable only if it arises naturally from the same geometric principle, not by declaring two different algorithms identical.

The most direct finite-temperature softening was tested once at intrinsic unit temperature, without a temperature sweep, and regressed both directions (results/terpene_free_energy_correspondence_dev_v1/). It is therefore a rejected diagnostic, not a tunable branch. The zero-temperature correspondence defect remains canonical.

## Product and reproducibility workflow

The same object is served through a versioned reference atlas and progressive
query-side observation acquisition; research evaluation, domain reconstruction,
and production visibility of positive pairs are kept separate. See
PRODUCT_RESEARCH_OBSERVATION_WORKFLOW.md.
\n

## Strict inductive transfer audit

A stronger deployment-style audit removes each held-out protein/reaction fold
from the factor atlas itself, rebuilds the reference geometry from the remaining
entities, and attaches every excluded entity out of sample using only
query-to-reference molecular observations. The pair field remains exactly the
same correspondence defect.

Result: results/terpene_product_correspondence_strict_inductive_v1/summary.json.

- E2R: MRR 0.06236, median rank 38, Hit@10 0.17391 versus canonical
  transductive-side-information MRR 0.07230, median 34, Hit@10 0.18261.
- R2E: MRR 0.06060, median rank 126, Hit@10 0.13253 versus canonical
  MRR 0.07469, median 115, Hit@10 0.14458.
- All nine evaluation cells preserve zero train/test protein and reaction overlap.
- The full 5 protein-fold + 5 reaction-fold reference rebuild and nine-cell audit
  completes in about 30 seconds on the current server; ordinary online query
  extension does not rebuild these atlases.

The modest transfer loss is the empirical cost of removing held-out entities
from the unsupervised geometry, not a switch to a different model. This audit is
the direct reproducibility bridge for a new application domain.

