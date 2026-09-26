# Mathematical foundation: the FIBRE interaction atlas

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

The global interaction is

\[
K(r,e)=
\sum_{\alpha\in A(r,e)}
\rho_\alpha(r,e)K_\alpha(r,e).
\]

A universal chart \(U_0=\mathcal R\times\mathcal E\) based on broad raw molecular inputs guarantees that \(A(r,e)\neq\varnothing\) for every valid pair. Consequently the global prediction remains defined when every optional chart is absent.

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

If two charts are simultaneously applicable, they are two coordinate descriptions of the same physical interaction. Therefore

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

This constrains agreement only where both experts claim biochemical applicability. It does not require heterogeneous molecular representations to be isometric.

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
- overlap/gluing consistency is the natural next regularizer.

The current prototype uses query-side gates for R2E/E2R. A future unified implementation may use pair-aware \(\rho_\alpha(r,e)\), but the ontology does not depend on that engineering choice.

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

then \(K_\alpha(r^\*,e^\*)\), and the partition glues them into \(K(r^\*,e^\*)\).

Thus double-unseen capability follows from molecular coordinate functions plus complete chart coverage, not from interpolation among entity IDs.

## 9. Sparse observations as an empirical interaction measure

Let

\[
\Omega=\{(r_i,e_i,w_i)\}
\]

be accepted experimental observations. In chart \(\alpha\),

\[
C_\alpha(\Omega)
=
\sum_i
w_i\rho_\alpha(r_i,e_i)
\phi_\alpha(r_i)\psi_\alpha(e_i)^\top.
\]

This finite-rank operator is a train-free sufficient statistic of newly observed interaction evidence. It can update a chart-local estimator or calibration state without retraining molecular encoders.

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
