# FIBRE — Field Inference for Bidirectional Reaction–Enzyme Retrieval

FIBRE is the current method identity. Its mathematical core is **sparse correspondence geometry** on the reaction–enzyme product manifold; the name refers to the two retrieval directions as fibres of one shared compatibility field. The canonical global field is also the first coordinate of a stratified biological relation: catalytic-pocket and mechanistic resolutions may refine what the global field leaves unresolved, but cannot cross its coarse numerical levels.

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
- Atom-mapped reaction-center Wasserstein and explicit transition-token geometry are valid catalytic-local observations, but forcing them into the **global reaction factor metric** causes a statistically clear E2R regression. They therefore do not alter the global geodesic.
- The same reaction-center observations can still enter FIBRE at a finer resolution: inside one global numerical level, the zero-temperature correspondence operator is evaluated again against active-site/pocket protein coordinates. Pocket-local ESM-C, pocket 3Di and pocket OT form catalytic-local coordinates; family-aware catalytic motifs form a still finer mechanistic chart. These are nested coordinates of the same relation, not a second scoring expert.

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


A verified positive is therefore an exact new observation, but exactness does not
imply that every unrelated ranking changes monotonically. The current 130-seed
audit finds that a seed changes about 1.08% of product-field entries on average;
unrelated-query RR worsens in about 8.1% of E2R and 4.4% of R2E comparisons while
mean RR change remains positive. Simple anchor isolation does not explain these
side effects, so no density penalty is currently justified. FIBRE reports the
seed's intrinsic novelty and exact influence footprint instead of changing its
weight. See stability_uncertainty.md.

## Exact scalable sections

The full reaction-by-protein matrix is a reference implementation, not a runtime
requirement. For fixed reaction r,

J_Omega(r,e) = min over p in Omega_E of
[d_E(e,p)^2 + min over (q,p) in Omega of d_R(r,q)^2].

Thus one R2E section requires only query-to-positive-reaction-support distances
and streamed candidate-to-positive-protein-support distances. E2R is symmetric.
The support-distance section functions in geometry/correspondence.py implement
this exact factorization. Candidate/support batching changes peak memory only.

On the current MARTS atlas the section solver's joint min-plus cost is bitwise
identical to the dense reference. The final defect differs by at most 1.78e-15
from floating subtraction order, which motivates explicit numerical level-set
semantics rather than relying on candidate-ID ordering inside ties.

## Numerical level sets and geometric applicability

Candidates whose defects differ by at most 64 float64 machine eps at the scale
of one section are one numerical level. FIBRE reports optimistic, neutral
expected, and pessimistic best-positive ranks plus exact expected reciprocal
rank under a neutral within-level permutation. This does not randomize product
output; it records what the scalar field itself resolves.

On current double-cold development, 29.6% of E2R queries and 18.1% of R2E
queries have a nontrivial best-positive rank interval. Starase Navigator exposes
threshold-free geometric_uncertainty provenance: query distance to positive
marginal support, best-level size/fraction, next-level gap, and candidate-support
distances. These are not probabilities or OOD classes.

## Stratified biological resolution

The scalar defect is the canonical **global coordinate**, but the scientific output is richer than one total order. A machine-stable global defect level defines a coarse correspondence class. FIBRE may resolve that class at a finer biological scale by applying the same correspondence defect on local factor manifolds.

The current catalytic-pocket resolution uses reaction-center transition geometry on the reaction side and three independent active-site/pocket coordinates on the protein side: pocket-local ESM-C, pocket 3Di and pocket OT. No scalar modality weight is learned or hand-set. In the strongest current construction, these local coordinates define Pareto strata inside one coarse level: candidate a dominates b only when a is no worse in every available pocket correspondence coordinate and is strictly better in at least one. Disagreement leaves candidates incomparable; missing local coordinates leave the coarse level unresolved.

Family-aware catalytic-motif coordinates such as class-I aspartate, NSE/DTE, DXDD and QW form a still finer **mechanistic chart**. Their absence is never negative evidence and they are not converted into a bonus score.

A finer scientific relation is not automatically promoted to the deterministic total rank. Pocket-only scalar refinement is non-degrading on the current double-cold development cells but still yields three small E2R regressions under strict factor-atlas rebuilding. Pareto-consensus refinement is more conservative but also retains a very small strict-inductive negative mean delta. The third mechanistic layer is now implemented as family-applicability charts with independent motif-coordinate correspondence defects and Pareto refinement inside an observed catalytic parent. It is sparse but survives strict factor rebuilding as a relation: development refines 8/115 E2R and 17/83 R2E queries; strict rebuilding refines the same 8/115 E2R queries and 11/83 R2E queries, while resolving no held-out positive at this layer. Therefore the canonical total rank remains the global correspondence order. Catalytic and mechanistic strata are first-class FIBRE outputs but currently **non-order-bearing**. This makes their inclusion rank-preserving by construction rather than by choosing a small fusion weight.

See stratified_geometry.md and the development/strict-inductive artifacts under results/fibre_stratified_*, results/fibre_consensus_stratified_*, results/fibre_mechanistic_stratified_dev_v1/ and results/fibre_mechanistic_stratified_strict_inductive_v1/.

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
- Runtime few-shot updates now expose the exact complete-section influence footprint under `seed_update_stability`; registered query/seed pairs also expose canonical product-support novelty and single-seed global field influence. These quantities are descriptive provenance only: a verified positive is never downweighted, gated, or blended by its novelty/influence diagnostics.

These are development diagnostics only and are not a reason to reopen the spent external-retention protocol.

## Relationship to the broad Rhea mainline

The broad Rhea v8 result remains frozen and should not be rewritten retroactively as this exact operator.

Both lines share the same scientific object:

1. label-free factor geometry;
2. sparse positive correspondence on M_R × M_E;
3. one scalar compatibility field;
4. R2E/E2R as fibres of that field;
5. missing information is absence of observation, not negative evidence.

The broad benchmark uses a smooth variational/transport realization; the compact MARTS application admits an exact zero-temperature correspondence field. We implemented the exact zero-temperature section at broad scale using lossless protein-support grouping and evaluated it on the same 1,903 clean-development queries and 185,918 candidates. It is computationally practical but materially weaker than frozen v8 (pooled MRR 0.12911 versus 0.14442; paired delta -0.01530, 95% CI [-0.02768,-0.00274]). Consequently numerical unification by simply replacing v8 with the raw zero-temperature defect is rejected. The two realizations remain related through the same product-manifold object, while the broad nonlinear variational extension is retained as scientifically consequential rather than treated as implementation noise.

The most direct finite-temperature softening was tested once at intrinsic unit temperature, without a temperature sweep, and regressed both directions (results/terpene_free_energy_correspondence_dev_v1/). It is therefore a rejected diagnostic, not a tunable branch. The zero-temperature correspondence defect remains canonical.

## Product and reproducibility workflow

The same object is served through a versioned reference atlas and progressive
query-side observation acquisition; research evaluation, domain reconstruction,
and production visibility of positive pairs are kept separate. See
`product_workflow.md`.

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

