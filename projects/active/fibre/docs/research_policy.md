# Unified geometry mainline policy

## Baseline authority

The only legacy/current comparison authority is BiME-Rank:

- numeric claims: `reproducibility/bime_rank/canonical.json`
- production contract: `configs/production_routes/default.yaml`
- replay/test scope: `reproducibility/bime_rank/source_roles.json`

No historical MARTS, ad-hoc confirmation, salted split, pair-memory confirmation, pocket-OT confirmation, or geometry benchmark result may be used as a baseline, tuning target, or promotion criterion.

## New method identity

**One-sentence definition:** enzyme-reaction retrieval is one sparse correspondence object on the product `M_R × M_E` of intrinsic reaction and enzyme catalytic-state factor spaces; these factors may be stratified and are only assumed locally regular on supported regions, not globally smooth. The zero-temperature defect `F(r,e)=-Delta_Omega` is its validated global coordinate and current deterministic linearization, while additional biologically justified correspondence coordinates define a fixed-domain Pareto partial relation with no hand-set weights or lexicographic priority. Positive-temperature kernel/heat and broad variational realizations are admissible only when their relationship to the same object is explicit. See `geometric_foundation.md`, `operator_family.md`, `partial_relation.md`, and `observation_model.md`.

Every algorithmic change must be interpretable as changing exactly one property of this same object: `(1)` the intrinsic metric on the reaction factor `M_R`, `(2)` the intrinsic metric on the protein factor `M_E`, `(3)` the sparse positive correspondence on `M_R × M_E`, `(4)` the mathematically specified product-field operator, or `(5)` a biologically meaningful coordinate/partial relation defined by that same correspondence operator on a declared comparison domain. A new score expert, router, gate, fallback, task-specific reranker, lexicographic biological hierarchy, or independent scalar fusion layer is outside the mainline by definition.

The active research method is not another routing layer on BiME-Rank. BiME-Rank is only the comparison baseline under matched splits, query sets, candidate support and metrics. The method under development is a scalar compatibility field `F(r,e)` on the Cartesian product of a reaction graph and a protein-state graph. Its mathematical definition must not inherit BiME-Rank's hard routing thresholds, fixed fallback choices, LambdaRank assembly, or other implementation artifacts. A BiME score field may be used only as an optional warm start in an ablation, not as a required component of the method.

For smooth operator-family members, adaptivity may arise from nonlinear anisotropic conductance as a continuous function of the local compatibility gradient. That construction is not the definition of FIBRE itself and is not required by the canonical zero-temperature defect. No OOD threshold, structure gate, or expert router belongs to the mathematical definition.

Accepted enzyme-reaction pairs are the current pair-level projection of richer biological observations/anchors sampled from a latent many-to-many biochemical relation on the same product space. R2E and E2R are two fibers/sections of the same correspondence object. Local generalization claims require an explicit supported-region assumption: sufficiently faithful factor geodesics, finite accepted-relation coverage radius, and locally regular set-valued fibers. Missing structure, assay context, mechanism or annotation is missing observation, never negative evidence. Pair-specific inactive/below-detection assays may act only as matched-context contradictions; they are never converted into permanent global negative edges.

## Asset policy

Reusable method implementations and geometry builders are active assets even if an earlier evaluation of one particular residual/reranker failed. Old benchmark data and old metric outputs remain quarantined and must not be reused for evaluation.

Reusable active assets include:

- `geometry/correspondence.py`: canonical zero-temperature correspondence field plus exact scalable sections and the finite-temperature bridge.
- `geometry/product_field.py`: reusable smooth Cartesian-product/Charbonnier operators for broad-family research; not the canonical focused solver.
- 3Di / pocket-3Di / pocket-OT builders: candidate ways to refine the protein-state graph geometry; they are not independent ranking experts.
- reaction-center geometry definitions: candidate ways to refine the reaction graph.
- BiME runtime scores/rankings: baseline outputs and optional warm-start assets only. They are not part of the geometric method definition and may be replaced by a more natural geometric initialization/readout when that preserves or improves matched-protocol performance.

Historical pair-memory experiments are not benchmarks. Their useful mathematical content is only the observation that one positive joint measure on enzyme×reaction space induces both R2E and E2R conditionals; this should be reimplemented against the BiME authority rather than replaying old splits.

## Stability and local-observation policy

Accepted positives receive an exact **set update** in the pair-level FIBRE
projection. This does not assert that all source records have equal biological
evidential status or that activity is condition-independent. A density/isolation
correction is not part of FIBRE unless intrinsic-geometry evidence shows that
support density systematically predicts harmful updates. The current 130-seed
audit does not show that relationship; exact update plus influence provenance is
canonical.

Reaction-center and transition observations may not deform the canonical global reaction factor: fixed-topology tangent refinement still causes a statistically clear E2R regression. They may, however, define additional catalytic-local correspondence coordinates of the same FIBRE object. Such a coordinate is part of an order-comparison relation only when it uses the same correspondence operator, has an explicit fixed comparison domain, leaves missing observations unordered rather than penalized, survives strict held-out-factor reconstruction, and is not silently converted into an independent weighted score. It is not required to stay inside a global numerical level; the current validated relation deliberately allows global/local disagreement to become biological incomparability.

Numerical level sets are scientific equivalence classes at machine-stable
precision. Candidate-ID ordering may be used only for deterministic display,
never as evidence that FIBRE resolved two candidates.

## Geometric generalization discipline

The product-space story is not allowed to stop at a visual manifold analogy. Any theoretical generalization claim must identify which quantities are assumptions and which are estimated from data. The current formalization uses: `(1)` one-sided accepted-relation coverage radius `h`; `(2)` reaction/protein squared-geodesic approximation errors `eps_R, eps_E`; and `(3)` local Hausdorff-Lipschitz fiber variation `L`. Under these assumptions, `J_Omega = Delta_Omega + m_R + m_E` is squared distance to accepted joint support, true-relation distance is controlled by `h`, and local fiber error is bounded by `sqrt(1+L^2) * sqrt(J_Omega)`. See `geometric_foundation.md` and the synthetic two-branch audit. None of `h`, `eps`, or `L` may be presented as empirically known until a matched real-data estimator/audit exists.

## Validation discipline

Method selection uses the current BiME internal clean-development protocol. The canonical strict temporal/external BiME evaluations are retention-only and must not be used for iterative tuning. All future geometry evaluations must preserve the BiME query/candidate support and metric definitions unless a change is explicitly versioned as a new benchmark rather than compared numerically to BiME.

Biological ordering authority requires a biological stress test appropriate to the claim, not merely aggregate MRR. Positive-only promiscuity can be evaluated from current accepted pairs; mechanism annotations may diagnose those results but are not a ranking gate. Assay-context shift requires repeated pair-specific assays, explicit-negative functional cliffs require source-bound inactive/below-detection observations, and cofactor compatibility requires comparable pair/reaction-side requirements. If those data are absent, the test is reported as blocked; unknown database edges must not be synthesized into biological negatives.

Geometric claims must also be scoped. FIBRE may assume locally regular/stratified factor geometry and a locally Hausdorff-Lipschitz many-to-many catalytic fiber for theorem statements, but it must not claim that all enzyme or reaction states form one globally smooth manifold. Product-space locality diagnostics may test consequences of the assumption; they do not estimate a universal biological Lipschitz constant or define deployment thresholds unless a separate held-out calibration justifies that use.

## Frozen broad reference versus current canonical operator

The broad Rhea v8 variational/product-flow result remains frozen as a matched-protocol reference and must not be retroactively renamed as the zero-temperature FIBRE defect. The current focused canonical operator is the correspondence defect in `geometry/correspondence.py`. A future broad FIBRE promotion must satisfy the derivation and parity gates in `operator_family.md`; old one-step/two-step Cartesian-walk or direction-specific variants remain diagnostic failures.
