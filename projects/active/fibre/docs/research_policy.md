# Unified geometry mainline policy

## Baseline authority

The only legacy/current comparison authority is BiME-Rank:

- numeric claims: `reproducibility/bime_rank/canonical.json`
- production contract: `configs/production_routes/default.yaml`
- replay/test scope: `reproducibility/bime_rank/source_roles.json`

No historical MARTS, ad-hoc confirmation, salted split, pair-memory confirmation, pocket-OT confirmation, or geometry benchmark result may be used as a baseline, tuning target, or promotion criterion.

## New method identity

**One-sentence definition:** enzyme-reaction retrieval is one stratified correspondence relation on the product manifold `M_R × M_E`: the canonical global coordinate is the scalar zero-temperature defect `F(r,e)=-Delta_Omega`, while catalytic-pocket and mechanistic coordinates may refine its numerical level sets only as finer resolutions of the same relation. Positive-temperature kernel/heat and broad variational realizations are admissible only when their relationship to the same field is explicit. See `operator_family.md` and `stratified_geometry.md`.

Every algorithmic change must be interpretable as changing exactly one property of this same object: `(1)` the intrinsic metric on the reaction factor `M_R`, `(2)` the intrinsic metric on the protein factor `M_E`, `(3)` the sparse positive correspondence on `M_R × M_E`, `(4)` the mathematically specified product-field operator, or `(5)` a nested finer-resolution equivalence/partial-order relation defined by the same operator inside a coarser numerical level. A new score expert, router, gate, fallback, task-specific reranker, or independent scalar fusion layer is outside the mainline by definition.

The active research method is not another routing layer on BiME-Rank. BiME-Rank is only the comparison baseline under matched splits, query sets, candidate support and metrics. The method under development is a scalar compatibility field `F(r,e)` on the Cartesian product of a reaction graph and a protein-state graph. Its mathematical definition must not inherit BiME-Rank's hard routing thresholds, fixed fallback choices, LambdaRank assembly, or other implementation artifacts. A BiME score field may be used only as an optional warm start in an ablation, not as a required component of the method.

For smooth operator-family members, adaptivity may arise from nonlinear anisotropic conductance as a continuous function of the local compatibility gradient. That construction is not the definition of FIBRE itself and is not required by the canonical zero-temperature defect. No OOD threshold, structure gate, or expert router belongs to the mathematical definition.

Known enzyme-reaction pairs are observations/anchors on the same product space. R2E and E2R are two slices/conditionals of the same field. Missing structure is missing geometric observation, never negative evidence.

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

Verified positives are exact observations. A density/isolation correction is not
part of FIBRE unless intrinsic-geometry evidence shows that support density
systematically predicts harmful updates. The current 130-seed audit does not
show that relationship; exact update plus influence provenance is canonical.

Reaction-center and transition observations may not deform the canonical global reaction factor: fixed-topology tangent refinement still causes a statistically clear E2R regression. They may, however, define a finer catalytic-local correspondence resolution inside one coarse numerical level. Such a resolution is part of FIBRE only when it uses the same correspondence operator, preserves missingness neutrality and coarse-level boundaries, and is not silently converted into an independent weighted score.

Numerical level sets are scientific equivalence classes at machine-stable
precision. Candidate-ID ordering may be used only for deterministic display,
never as evidence that FIBRE resolved two candidates.

## Validation discipline

Method selection uses the current BiME internal clean-development protocol. The canonical strict temporal/external BiME evaluations are retention-only and must not be used for iterative tuning. All future geometry evaluations must preserve the BiME query/candidate support and metric definitions unless a change is explicitly versioned as a new benchmark rather than compared numerically to BiME.

## Frozen broad reference versus current canonical operator

The broad Rhea v8 variational/product-flow result remains frozen as a matched-protocol reference and must not be retroactively renamed as the zero-temperature FIBRE defect. The current focused canonical operator is the correspondence defect in `geometry/correspondence.py`. A future broad FIBRE promotion must satisfy the derivation and parity gates in `operator_family.md`; old one-step/two-step Cartesian-walk or direction-specific variants remain diagnostic failures.
