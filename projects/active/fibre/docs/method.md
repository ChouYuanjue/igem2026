# FIBRE — Field Inference for Bidirectional Reaction–Enzyme Retrieval

FIBRE is the current method identity. Its mathematical core is **sparse correspondence geometry** on the product of reaction and enzyme catalytic-state factor spaces; the name refers to the two retrieval directions as fibres of one shared correspondence object. We do not require one globally smooth manifold: the intended factors are intrinsic metric spaces that are locally represented by regular/low-dimensional strata, while the true biochemical relation is a many-to-many subset of their product. The validated scalar global field remains the deterministic production readout, while scientifically meaningful local correspondence coordinates participate in a fixed-domain Pareto partial relation with no scalar fusion weights and no lexicographic priority.

## One sentence

**With sparse positive enzyme–reaction observations but rich, incomplete molecular measurements, we build intrinsic reaction and enzyme factor geometries and score a pair by the excess product-geodesic cost of explaining both coordinates with one shared known biochemical precedent rather than with unrelated marginal neighbours.**

Formally, let

\[
M = M_R \times M_E,
\qquad
\Omega \subset M_R\times M_E
\]

be the reaction–enzyme product metric space and the sparse set of accepted positive biochemical correspondences.

The factor geometries are label-free. (M_R) is built from global reaction chemistry; (M_E) is a partially observed multiresolution catalytic-state metric space. On regular local regions either factor may admit a smooth/low-dimensional chart, but the method does not require one global manifold. Each factor geodesic is normalized by its own intrinsic median local edge length before forming the canonical Cartesian product.

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

The geometric meaning of the three terms is exact:

\[
J_\Omega=\Delta_\Omega+m_R+m_E
=d_{M_R\times M_E}((r,e),\Omega)^2.
\]

Thus the defect is joint-consistency **after subtracting marginal novelty**, while the sum reconstructs absolute distance to the nearest accepted biochemical precedent. This is why FIBRE now reports support/applicability beside the defect rather than treating a low defect as an activity probability.

The formal assumptions and local-generalization propositions are stated in [`geometric_foundation.md`](geometric_foundation.md). FIBRE treats the factors as locally regular intrinsic metric spaces and the catalytic relation as a many-to-many set-valued correspondence; it does **not** require one globally smooth enzyme manifold or a one-to-one reaction↔enzyme map. If the declared working reaction-to-enzyme fiber (G^\star) varies locally in Hausdorff distance with constant (L), then

\[
d_E(e,G^\star(r))
\le
\sqrt{1+L^2}\,
\sqrt{\Delta_\Omega(r,e)+m_R(r)+m_E(e)}.
\]

The symmetric statement holds for E2R. If accepted positives cover the local working relation within product radius (h), every true pair in that covered region has (J_\Omega\le h^2). Local prediction is therefore justified by **regularity of the true fiber + coverage of accepted local precedents + faithful factor geodesics**, not by a vague claim that nearby embeddings should have the same label.

## Why this matches the data-scarcity problem

The asymmetry is relative, not absolute. Sequence and some structural observations are much more densely available than **pair-specific, condition-resolved catalytic evidence**, but the catalytically decisive molecular state is itself incomplete. FIBRE therefore does not assume that sequence or a predicted apo structure is a complete enzyme state.

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
- The same reaction-center observations can still enter FIBRE as independent catalytic-local correspondence coordinates evaluated against active-site/pocket protein coordinates. Pocket-local ESM-C, pocket 3Di and pocket OT now join the global defect in the validated four-coordinate Pareto relation. A globally better candidate can therefore become biologically incomparable when it is worse on catalytic coordinates. Family-aware catalytic motifs remain mechanistic observations/coordinates but are not yet part of the common order-comparison family because their applicable domain is substantially sparser. None of these are independent score experts.

This is a stricter interpretation of “use all information”: **nothing scientifically useful is discarded, but no information source is allowed to deform the mathematical object merely because it exists.**

## Enzymology state: correspondence, catalytic state, assay constraints, applicability

The biological world is not assumed to be a static Boolean relation. A more faithful latent object is an activity response such as

\[
A(e,r;c),
\]

where \(c\) denotes assay/process context. The current sparse positive set \(\Omega\) is therefore interpreted as a **task-level projection of accepted observations**, not as a claim that a pair is universally active under every condition.

FIBRE now represents one candidate through four coordinated but deliberately non-fused components:

1. **molecular correspondence** — the global/pocket FIBRE defect coordinates and their partial relation;
2. **catalytic-state evidence** — pair-associated mechanism observations together with protein-level catalytic/cofactor/site annotations, with source scope preserved;
3. **assay constraints** — pair-specific, source-bound pH/temperature/cofactor observations when an experiment can be assigned unambiguously to that pair;
4. **absolute support/applicability** — intrinsic distance to accepted positive support, reported beside the correspondence defect rather than hidden inside a confidence score.

These components are not added with hand weights. In particular, protein-level UniProt annotations are not promoted to pair-specific assay facts, and absence of an assay at a requested condition remains unresolved.

An explicitly scoped inactive/below-detection/no-conversion assay may contradict a candidate **only under the matched reported context**. When the user explicitly requests that same context, such an exact pair-specific contradiction acts as a censoring constraint on candidate eligibility; it never changes the FIBRE defect and is never converted into a permanent negative edge. A matched positive assay does not automatically dominate a candidate whose condition evidence is missing, and conflicting positive/negative evidence remains unresolved rather than censored. This preserves the distinction between unknown, unsupported, contradicted and conflicting evidence without requiring a dense activity model that the current data cannot support.

Current source-bound assay coverage is still too sparse to make pH, temperature or cofactor state a general ranking coordinate. The pair-specific context index therefore has no scalar ranking effect. Its only ordering consequence is the narrow exact-context censor above. The current materialized corpus contains no such negative assay, so present production ranks are unchanged. This is an intentional coverage boundary rather than a missing implementation.

Full ligand-bound conformational ensembles, protonation states, catalytic waters and other active-state details are not fabricated when unavailable. They remain explicit unobserved state variables rather than reasons to penalize a candidate.

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


A newly accepted positive is therefore an **exact set update of the accepted
pair-level projection**. This mathematical exactness does not imply that all
experimental sources have equal biological evidential strength, nor that the
pair is active under every assay condition. Provenance and assay context remain
part of the observation state rather than being collapsed into the update rule.

Exactness also does not imply that every unrelated ranking changes monotonically. The current 130-seed
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
distances. Each returned candidate also carries its own canonical support
distance next to the correspondence defect. These are not probabilities or OOD
classes and do not alter the molecular ordering.

## Partial biological relation

The scalar defect is the validated **global coordinate**, but the scientific output is not assumed to be a total order. For a fixed query and candidate, FIBRE currently uses the declared coordinate family

\[
(\Delta_{\mathrm{global}},\Delta_{\mathrm{pocket\text{-}ESMC}},\Delta_{\mathrm{pocket\text{-}3Di}},\Delta_{\mathrm{pocket\text{-}OT}}).
\]

Each coordinate is produced by the same sparse correspondence idea on a biologically appropriate factor geometry. Candidate (a) dominates candidate (b) only if it is no worse in every declared coordinate and strictly better in at least one. Otherwise the two candidates remain incomparable on this family. No modality receives a learned or hand-set scalar weight, and the global coordinate has no lexicographic priority. A candidate missing any declared coordinate is left unordered rather than imputed or penalized.

This construction is materially different from the older stratified compatibility view, which allowed pocket information to refine only an unresolved global numerical level. That interface remains useful for historical experiments and backward-compatible runtime provenance, but it is no longer the scientific ontology of FIBRE.

The relation survives the strict held-out-factor audit. On informative queries, the fraction of otherwise global-better comparisons converted into genuine trade-offs is 68.7% E2R / 86.3% R2E in development and 68.0% / 85.7% after reference-only factor rebuilding and out-of-sample attachment. On queries informative in both audits, first-front membership agrees 94.1% E2R and 96.3% R2E. Strict median first-front size is only 4.86% and 0.68% of complete candidates, respectively, so the result is not a trivial explosion of nondominated hypotheses.

The operational deterministic list still uses the global defect. Earlier attempts to linearize pocket information can regress strict-inductive retrieval, so global ranking is retained as a stable product readout rather than as a claim of biological precedence. Family-aware catalytic motifs such as class-I aspartate, NSE/DTE, DXDD and QW remain mechanistic coordinates with explicit applicability/missingness, but their sparse common domain does not yet justify adding them to the current fixed comparison family.

There is also a weight-robust interpretation of this partial relation. If candidate \(a\) Pareto-dominates \(b\), every strictly positive linear weighting of the declared defect coordinates prefers \(a\). If the candidates trade off, their total ordering necessarily depends on an additional scalarization/utility choice. FIBRE therefore reports the comparison that is invariant to unknown positive weights and leaves unsupported trade-offs unresolved.

See `geometric_foundation.md`, `partial_relation.md`, `observation_model.md`, the current audits under `results/fibre_partial_relation_*`, and `stratified_geometry.md` for the formal assumptions, proofs, biological observation model and retained compatibility/history view.

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

