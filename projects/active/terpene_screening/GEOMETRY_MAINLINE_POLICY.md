# Unified geometry mainline policy

## Baseline authority

The only legacy/current comparison authority is BiME-Rank:

- numeric claims: `reproducibility/bime_rank/canonical.json`
- production contract: `configs/production_routes/terpene_v1.yaml`
- replay/test scope: `reproducibility/bime_rank/source_roles.json`

No historical MARTS, ad-hoc confirmation, salted split, pair-memory confirmation, pocket-OT confirmation, or geometry benchmark result may be used as a baseline, tuning target, or promotion criterion.

## New method identity

**One-sentence definition:** enzyme-reaction retrieval is one scalar compatibility field `F(r,e)` on the product manifold `M_R × M_E`, where known positive pairs form an empirical measure and prediction is the intrinsic-geometry-constrained variational extension of that field.

Every algorithmic change must be interpretable as changing exactly one property of this same object: `(1)` the intrinsic metric on the reaction factor `M_R`, `(2)` the intrinsic metric on the protein factor `M_E`, `(3)` the empirical positive measure on `M_R × M_E`, or `(4)` the variational energy/operator used to extend `F`. A new score expert, router, gate, fallback, task-specific reranker, or independent fusion layer is outside the mainline by definition.

The active research method is not another routing layer on BiME-Rank. BiME-Rank is only the comparison baseline under matched splits, query sets, candidate support and metrics. The method under development is a scalar compatibility field `F(r,e)` on the Cartesian product of a reaction graph and a protein-state graph. Its mathematical definition must not inherit BiME-Rank's hard routing thresholds, fixed fallback choices, LambdaRank assembly, or other implementation artifacts. A BiME score field may be used only as an optional warm start in an ablation, not as a required component of the method.

Adaptive means nonlinear anisotropic geometric flow: edge conductance is a continuous function of the current local compatibility gradient. Smooth regions propagate evidence; compatibility boundaries suppress transport. There is no OOD threshold, structure gate, or expert router in the mathematical definition.

Known enzyme-reaction pairs are observations/anchors on the same product space. R2E and E2R are two slices/conditionals of the same field. Missing structure is missing geometric observation, never negative evidence.

## Asset policy

Reusable method implementations and geometry builders are active assets even if an earlier evaluation of one particular residual/reranker failed. Old benchmark data and old metric outputs remain quarantined and must not be reused for evaluation.

Reusable active assets include:

- `geometric_product_field.py`: robust Cartesian-product scalar field and nonlinear Charbonnier flow.
- 3Di / pocket-3Di / pocket-OT builders: candidate ways to refine the protein-state graph geometry; they are not independent ranking experts.
- reaction-center geometry definitions: candidate ways to refine the reaction graph.
- BiME runtime scores/rankings: baseline outputs and optional warm-start assets only. They are not part of the geometric method definition and may be replaced by a more natural geometric initialization/readout when that preserves or improves matched-protocol performance.

Historical pair-memory experiments are not benchmarks. Their useful mathematical content is only the observation that one positive joint measure on enzyme×reaction space induces both R2E and E2R conditionals; this should be reimplemented against the BiME authority rather than replaying old splits.

## Validation discipline

Method selection uses the current BiME internal clean-development protocol. The canonical strict temporal/external BiME evaluations are retention-only and must not be used for iterative tuning. All future geometry evaluations must preserve the BiME query/candidate support and metric definitions unless a change is explicitly versioned as a new benchmark rather than compared numerically to BiME.

## Frozen canonical implementation

The current canonical internal mainline is `evaluate_geometric_product_flow_clean_dev_v8.py`, summarized in `PRODUCT_MANIFOLD_MAINLINE.md`. Its empirical-measure mollifier is the factor-exchange-invariant `1/2 (K_R K_E + K_E K_R)`. The earlier v5 axis ordering is retained only as a statistically equivalent diagnostic reference, not as a separate conceptual model. The one-step and two-step joint Cartesian-walk variants are diagnostic failures and must not be promoted by selective metrics.
