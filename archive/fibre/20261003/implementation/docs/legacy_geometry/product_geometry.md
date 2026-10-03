# Product-Manifold Mainline

> **Invariant narrative.** Pair labels are scarce; molecular observations are not. We use all auditable molecular information to construct partially observed reaction/enzyme geometry, then let the sparse known positive correspondence induce one compatibility field on their product. R2E and E2R are fibres of that field. Missing observations are never negative evidence, and no biological view is forced into the ranking metric if doing so breaks the geometry.

For the current MARTS application realization, the canonical object is the correspondence defect in `method.md` and `operator_family.md`. The broad Rhea v8 realization remains frozen separately under the same product-manifold scientific narrative.

## One sentence

**Because experimentally known enzyme–reaction pairs are scarce, we use every available observation to define one partially observed reaction–enzyme geometry `M_R × M_E`; known positive pairs form a sparse empirical measure on that same space, and one compatibility field `F(r,e)` is obtained by intrinsic-geometry-constrained variational extension.**

Nothing else is a separate model component.

## Data-scarcity principle: use information, not extra scorers

The central statistical asymmetry is that **pair labels are scarce but side observations are much richer**. Sequence is broadly available; structure is available for only part of the library; reaction chemistry admits several deterministic descriptions; domain architecture, catalytic motifs, pocket geometry, homology, and curated reference enzymes are informative only for subsets of proteins or reaction families. The method should not discard these observations merely because they are heterogeneous or incomplete.

The governing rule is therefore: **every scientifically meaningful observation must have a defined role in the same geometric object.**

- observations that define pairwise neighbourhood geometry contribute to `g_R` or `g_E`;
- experimentally established enzyme–reaction pairs contribute mass to the single positive measure `mu`;
- sparse family-conditional annotations that cannot yet define an unbiased global metric remain local observables attached to the same manifold and are reported as mechanistic evidence rather than converted into hard gates;
- missing observations contribute neither attraction nor repulsion. Missing information is absence of a coordinate, not negative evidence.

This is the precise sense in which the method “uses all available information”: it maximizes the observed geometry around each point without inventing labels or independent experts. As coverage improves, an observation may move from an interpretive observable into the intrinsic metric, but it must remain a coordinate of the same `M_R` or `M_E`, never a separate ranker.

## Narrative hierarchy — do not invert it

The method starts from the geometric object, not from a retrieval stack and not from a list of biological heuristics. The intended story is, in order:

1. a reaction is a point on one chemical manifold `M_R`;
2. an enzyme molecular state is a point on one protein manifold `M_E`;
3. a candidate enzyme–reaction hypothesis is therefore a point on the product `M_R × M_E`;
4. known biochemical pairs are an empirical positive measure on this same product space;
5. one nonlinear anisotropic variational problem extends a prior compatibility field over the product geometry;
6. R2E and E2R are fibres of that one inferred field;
7. sequence homology, Pfam, catalytic motifs, pockets, and known reference enzymes are observations that make a point or neighbourhood biologically inspectable. They do not become a second model layer merely because they are easy to explain.

Biological evidence therefore **closes the interpretation loop of the geometric object**; it does not replace the geometric story. Any future presentation that reads as “embedding model + graph correction + biological reranking + explanation module” has drifted away from the method definition.

## Mathematical object

Let `M_R` be the reaction manifold with intrinsic metric `g_R`, `M_E` the protein-state manifold with intrinsic metric `g_E`, and `M = M_R × M_E` their Cartesian product with direct-sum metric `g = g_R ⊕ g_E`. Let `F0 : M -> R` be the smooth prior compatibility field and let `mu = sum_(r,e in Omega) w_re delta_(r,e)` be the empirical positive measure from known fold-train pairs only.

Reaction and protein representations do not vote as experts. They only determine the two factor metrics.

## Factor metrics

The reaction factor uses local coordinates for global chemistry, whole-reaction physicochemical geometry, and mapped reaction-center geometry. Each coordinate precision is determined without labels by local effective-support contraction,

`lambda_v = exp(KL(p_v || U)) - 1 = N / exp(H(p_v)) - 1`.

A locally flat coordinate therefore contributes zero precision, while a coordinate that genuinely resolves the local reaction neighbourhood contributes according to its intrinsic resolution gain. There are no learned inter-view weights.

The protein factor keeps the self-tuning sequence/observed-structure atlas topology and conformally corrects conductance on the same edges using exact one-step diffusion distance. Missing structure remains missing geometric observation and never becomes negative evidence. On the terpene application submanifold this principle is extended to a partially observed multiresolution molecular-state geometry (`MULTIRESOLUTION_MOLECULAR_STATE_GEOMETRY.md`): every jointly observed sequence/structure/pocket/local-context coordinate may refine the same `g_E` after intrinsic scale normalization, while missing coordinates are omitted exactly.

## Empirical measure

The positive relation is restricted to the numerical chart by a mass-conserving geometric partition of unity, symmetrically degree-normalized, and globally normalized to one joint probability measure.

A finite-chart atomic measure is mollified by the same product geometry before entering the field equation. Let `K_R` and `K_E` be mass-conserving, `F0`-dependent Charbonnier Markov kernels along the two factor directions. The canonical mollifier is

`mu_tilde = 1/2 (K_R K_E + K_E K_R) mu`.

It is positive, mass-conserving, and invariant to exchanging the factor order. It retains two-factor paths to diagonal product neighbourhoods without privileging reaction-first or protein-first propagation.

Direct atomic forcing, one-step joint Cartesian diffusion, and two-step joint Cartesian diffusion were clean-dev diagnostics and are not part of the mainline.

## Variational extension

Let `Phi_g(F)` be the Charbonnier energy induced by the product geometry. The inferred compatibility field is

`F* = argmin_F [ 1/2 ||F-F0||_M^2 + D_{Phi_g}(F,F0) - sigma <mu_tilde, F-F0> ]`,

where `D_{Phi_g}` is the Bregman divergence and `sigma` is the characteristic field scale derived from the field and geometry.

With no positive observations, `F0` is exactly the minimizer. Positive observations deform the field; missing observations do not create negative evidence. The Euler–Lagrange operator is nonlinear because Charbonnier conductance is recomputed from the current field: smooth regions propagate evidence while sharp compatibility boundaries suppress transport.

## One field, two retrieval directions

R2E and E2R are fibres of the same field, not separate models: R2E ranks `F*(r, ·)` and E2R ranks `F*(·, e)`.

Local charts are only a numerical device. Final R2E ranking is restored to the full `185,918`-protein support under the frozen evaluation contract.

## Frozen clean-dev result

Canonical implementation: `evaluate_geometric_product_flow_clean_dev_v8.py`.

On the frozen 1,903-query clean-dev protocol: MRR `0.1444156685`, MAP `0.1264155984`, macro ROC-AUC `0.9847072374`, NDCG@10 `0.1526840533`, Hit@10 `0.3431424067`, Hit@20 `0.4466631634`, Hit@50 `0.5827640568`, median best-positive rank `28`.

All `1,903 / 1,903` solves converge at relative tolerance `1e-5`; no query reaches the 128-step safety cap. Every fold has zero train/dev reaction overlap and zero exact pair overlap.

Relative to the asymmetric-order v5 mollifier, pooled differences are negligible: MRR `-0.000118`, MAP `-0.000007`, NDCG@10 `-0.000149`, while Hit@20, Hit@50, and median rank are identical; paired bootstrap intervals include zero. The factor-exchange-invariant definition is therefore preferred as the canonical mainline.

The one-time strict external retention remains spent and is not used to choose or tune this definition.

## Why the protein coordinates do not need learned weights

For unit-normalized ESM-C and CLIP coordinates with cosine similarities `s_ESM` and `s_CLIP`, the canonical observed-pair combination

`1 - (s_ESM + s_CLIP)/2`

is exactly one half of the sum of their squared chordal distances. Hence, up to a global constant absorbed by the self-tuning local scale, the current protein atlas is the direct-sum coordinate metric `g_ESM ⊕ g_CLIP`, not a tuned two-expert score fusion. If CLIP is missing, only the observed ESM coordinate remains. A clean-dev diagnostic that replaced this direct sum by additional perplexity-dependent coordinate precision reduced top-rank performance, so no extra adaptive weighting is retained.

## Biological support witnesses

The mathematical field is accompanied by a ranking-invariant biological witness map described in `BIOLOGICAL_INTERPRETABILITY_CONTRACT.md`. For a predicted `(r,e)`, the witness identifies nearby observed positive support `(r_i,e_i)∈Ω` under the same reaction/protein geometry, then reports human-readable chemistry, sequence, structure, Pfam/domain, motif, and pocket observations when available. These observations explain the field; they are not extra experts and missing observations never penalize a candidate.

## Resolution-product protein metric diagnostic (v10)

A fully symmetric-looking diagnostic replaced the canonical protein direct-sum ESM-C/observed-CLIP metric by the same perplexity-contraction resolution-product construction used on the reaction factor. This is mathematically attractive but is **not promoted**. Across all 1,903 clean-dev queries, relative to v8 it changes MRR by `+0.00228` and MAP by `+0.00148`, but Hit@10 by `-0.00841`, Hit@50 by `-0.00736`, macro ROC-AUC by about `-4.6e-6`, and median best-positive rank from `28` to `30`. Paired bootstrap intervals for the MRR/MAP gains include zero, while the Hit@10/Hit@50 losses are one-sidedly unfavorable.

This is an important design rule: **formal symmetry is not sufficient if it damages the frozen retrieval behavior**. The current v8 protein direct-sum metric remains canonical because it already has a clean geometric interpretation and does not incur that regression.
