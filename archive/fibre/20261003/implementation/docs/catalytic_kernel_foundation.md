# 第一版交互图册数学基础（历史保留）

> 当前 FIBRE 已晋级到第二版“条件催化模式”解释。本文件保留第一版图册、分区粘合和重叠一致性理论，用于开发史与复现追溯；现行数学定义见
> `projects/active/fibre/docs/theory/FIBRE_CONDITIONAL_MODES_THEORY_ZH.md`，
> 现行证据见
> `reproducibility/bime_rank/records/FIBRE_CONDITIONAL_MODES_SCORECARD_V2.json`。
> 第一版中的 0.02 跨专家一致性惩罚已经在第二版删除。

## 0. Biological target and task-level object

Catalytic activity is condition-dependent. Conceptually let

\[
A:\mathcal R\times\mathcal E\times\mathcal C\to\mathbb R
\]

represent a context-resolved activity response. Present registries contain sparse, selectively observed projections of this object and often do not resolve \(c\in\mathcal C\). FIBRE therefore estimates a declared task-level working compatibility \(K^\star(r,e)\). All results below concern estimators of that working object; they are not claims that physical enzyme activity is context-free.

## 1. Multi-view observation geometry

For reaction \(r\) and enzyme \(e\), let

\[
X_R(r)=\{x^v_R(r):v\in V_R(r)\},\qquad
X_E(e)=\{x^u_E(e):u\in V_E(e)\},
\]

where the available-view sets \(V_R(r)\) and \(V_E(e)\) may vary by object.

A missing view is absence of a coordinate chart, not a zero-valued biological observation.

FIBRE uses an interaction-domain cover

\[
\mathcal R\times\mathcal E=\bigcup_{\alpha\in A}U_\alpha.
\]

A chart selects molecular views relevant to one biochemical regime and maps them to dual coordinate spaces

\[
\phi_\alpha:U^R_\alpha\to H^R_\alpha,\qquad
\psi_\alpha:U^E_\alpha\to H^E_\alpha.
\]

Its local interaction is

\[
K_\alpha(r,e)=\phi_\alpha(r)^\top G_\alpha\psi_\alpha(e).
\]

## 2. Coordinate invariance

The value of a local interaction does not depend on a particular latent basis.

For invertible coordinate changes

\[
\tilde\phi=A\phi,\qquad
\tilde\psi=B\psi,
\]

choose

\[
\tilde G=A^{-T}GB^{-1}.
\]

Then

\[
\tilde\phi^\top\tilde G\tilde\psi
=
\phi^\top G\psi.
\]

Therefore ESM-C, EnzGFM, structure and reaction-centre charts do not need identical dimensions or aligned axes. Their coordinates are meaningful only together with their local interaction form.

This invariance is tested in projects/active/fibre/tests/test_interaction_atlas.py.

## 3. Partition-of-unity gluing

Let \(A(r,e)\) denote charts available for a pair. FIBRE learns non-negative applicability functions

\[
\rho_\alpha(r,e)\ge0,\qquad
\sum_{\alpha\in A(r,e)}\rho_\alpha(r,e)=1.
\]

The pair-aware atlas estimator is

\[
\widehat K(r,e)=
\sum_{\alpha\in A(r,e)}
\rho_\alpha(r,e)K_\alpha(r,e).
\]

A universal chart \(U_0=\mathcal R\times\mathcal E\) based on broad raw molecular inputs guarantees that \(A(r,e)\neq\varnothing\) for every valid pair. Consequently the prediction remains defined when every optional chart is absent.

Because the weights are non-negative and sum to one, the glued score obeys the convex-hull bound

\[
\min_{\alpha\in A(r,e)}K_\alpha(r,e)
\le \widehat K(r,e) \le
\max_{\alpha\in A(r,e)}K_\alpha(r,e).
\]

If only one chart is available, its normalized weight is exactly one and the glued score is exactly that chart score. These properties are tested in `test_interaction_atlas.py`.

## 4. Local low-rank structure

Within one chart, assume the catalytic interaction is square-integrable. The associated Hilbert–Schmidt operator has singular expansion

\[
K_\alpha(r,e)
=
\sum_{k\ge1}
\sigma_{\alpha k}
u_{\alpha k}(r)v_{\alpha k}(e).
\]

The rank-\(d_\alpha\) truncation obeys

\[
\|K_\alpha-K_{\alpha,d_\alpha}\|_{HS}^2
=
\sum_{k>d_\alpha}\sigma_{\alpha k}^2.
\]

Thus a dual tower is a finite-rank approximation inside one chart. A multi-expert model uses several such local low-rank approximations rather than forcing one global low-rank model to explain every catalytic regime.

## 5. Consistency on chart overlaps

If two charts are simultaneously applicable, they are two coordinate descriptions of the same task-level catalytic compatibility. Therefore

\[
K_\alpha(r,e)\approx K_\beta(r,e)
\quad
\text{on }U_\alpha\cap U_\beta.
\]

A trainable compatibility penalty is

\[
\mathcal L_{\mathrm{glue}}
=
\mathbb E
\sum_{\alpha<\beta}
\rho_\alpha\rho_\beta
\left(K_\alpha-K_\beta\right)^2.
\]

This constrains agreement only where both experts claim biochemical applicability. It does not require heterogeneous molecular representations to be isometric. Every term is non-negative. On an item for which every overlapping chart pair has positive partition product, the itemwise penalty is zero exactly when all such active chart scores agree.

## 6. Relation to the existing gated multi-expert model

The existing directional multi-expert prototype already contains the essential pieces:

\[
K_{\mathrm{global}}=\langle z_R,z_E\rangle,
\]

local expert scores

\[
K_h=\langle z^h_R,z^h_E\rangle,
\]

and softmax gates over experts. Existing balance and diversity penalties discourage gate collapse and duplicate expert coordinates.

Under the atlas interpretation:

- global_embedding is the universal chart;
- expert_embeddings are local chart coordinates;
- softmax gates approximate partition weights;
- expert diversity is chart diversity;
- overlap/gluing consistency is the geometric compatibility regularizer.

The reproduction multi-expert trainer exposes this as an opt-in glue-weight term with default 0, so historical runs remain unchanged. The current prototype uses query-side gates and distinct learned expert masses for R2E and E2R. Its deployed readouts are therefore

\[
S_d(r,e)=\sum_\alpha \rho_\alpha^{(d)}(r,e)K_\alpha(r,e),
\qquad d\in\{\mathrm{R2E},\mathrm{E2R}\}.
\]

This does not imply pointwise equality between the two directional estimators. It states instead that both directions reuse the same local pair scores while applying different retrieval partitions. A future unified implementation may use pair-aware \(\rho_\alpha(r,e)\), but promotion requires empirical validation rather than formal symmetry alone.

### Directional discrepancy bound

Let \(p\) and \(q\) be two valid partitions over the same active chart-score vector \(k\). Since \(\sum_\alpha(p_\alpha-q_\alpha)=0\), subtract any constant \(c\):

\[
S_p-S_q=\sum_\alpha(p_\alpha-q_\alpha)(k_\alpha-c).
\]

Choosing \(c=(k_{\max}+k_{\min})/2\) gives

\[
|S_p-S_q|
\le \sum_\alpha|p_\alpha-q_\alpha|
\frac{k_{\max}-k_{\min}}{2}
=\frac12\|p-q\|_1(k_{\max}-k_{\min}).
\]

Hence directional disagreement vanishes if the partitions agree, if all active charts agree, or both. `partition_readout_discrepancy_bound` implements this deterministic diagnostic and the bound is tested directly.

## 7. TPS as a biochemical chart

TPS specialization is

\[
U_{\mathrm{TPS}},
\quad
K_{\mathrm{TPS}},
\quad
\rho_{\mathrm{TPS}}.
\]

The support of \(U_{\mathrm{TPS}}\) is determined by TPS-relevant molecular state, not by dataset membership. A TPS-like candidate in a large external protein universe can therefore receive TPS chart mass; a non-TPS candidate in the same universe need not.

This separates where we search from which local interaction coordinates are informative.

## 8. Double-unseen prediction

For an unseen reaction \(r^\*\) and unseen enzyme \(e^\*\), each available chart computes

\[
\phi_\alpha(r^\*),\qquad
\psi_\alpha(e^\*),
\]

then \(K_\alpha(r^\*,e^\*)\), and an applicable partition glues them into a retrieval estimate of \(K^\star(r^\*,e^\*)\).

Thus double-unseen capability follows from molecular coordinate functions plus complete chart coverage, not from interpolation among entity IDs.

## 9. Sparse observations as bounded finite-rank interaction updates

Let

\[
\Omega=\{(r_i,e_i,w_i)\}_{i=1}^{n}
\]

be accepted experimental observations. In chart \(\alpha\), define

\[
\Delta_\alpha(\Omega)
=
\sum_i
w_i\rho_\alpha(r_i,e_i)
\phi_\alpha(r_i)\psi_\alpha(e_i)^\top.
\]

Each summand is rank at most one, so subadditivity of matrix rank gives

\[
\operatorname{rank}(\Delta_\alpha)\le n.
\]

After rescaling to a declared Frobenius budget \(\varepsilon_\alpha\),

\[
\|\Delta_\alpha\|_F\le\varepsilon_\alpha.
\]

For unit-norm chart coordinates, Cauchy--Schwarz and \(\|M\|_2\le\|M\|_F\) imply the pointwise perturbation bound

\[
|\phi_\alpha(r)^\top\Delta_\alpha\psi_\alpha(e)|
\le\varepsilon_\alpha.
\]

`FiniteRankInteractionUpdate` implements this rectangular chart-local operator, including the norm cap; its weights may directly include \(w_i\rho_\alpha\). The earlier `PositivePairConditioner` is the square shared-latent special case already used by the project. The mathematical primitive is implemented and tested; automatic application-layer projection of each external observation across all applicable charts remains separate orchestration and is not assumed here.

## 10. What the geometry explains

The interaction-atlas construction explains in one object:

- heterogeneous representation dimensions;
- multiple experts;
- missing modalities;
- family specialization;
- universal prediction coverage;
- double-unseen scoring;
- sparse pair supervision;
- train-free experimental updates;
- interpretable expert/evidence decomposition.

It does not require a global enzyme manifold, a global reaction manifold or a product-manifold distance.
