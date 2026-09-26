# FIBRE geometric foundation

This note states the mathematical assumptions under which the current FIBRE construction is a reasonable local generalization mechanism. The assumptions are deliberately weaker than saying that all enzymes or all reactions lie on one globally smooth manifold.

## 1. Factor spaces

Let \((M_R,d_R)\) denote reaction state space and \((M_E,d_E)\) enzyme catalytic-state space. In the implementation these are finite graph approximations built from molecular observations, but the mathematical object is an intrinsic metric space.

The working assumption is:

- each factor is locally geodesic on the region being queried;
- away from singular transitions it is well approximated by a low-dimensional smooth stratum or another set of positive local reach;
- different enzyme families, catalytic mechanisms or reaction regimes may occupy different strata;
- the current molecular views are partial observation maps of the latent catalytic state, not a claim that sequence or an apo structure completely determines catalysis.

This is compatible with classical manifold-learning theory: under regularity and sampling assumptions, shortest-path distances on sufficiently dense neighborhood graphs can approximate intrinsic geodesic distances on an embedded manifold. Reach/condition-number assumptions are a standard way to rule out arbitrarily high curvature and near self-intersection. We use these results only as motivation for the factor-geometry approximation; the present data do not establish a global smooth-manifold theorem.

The canonical product metric is

\[
d_M\big((r,e),(r',e')\big)^2
=
d_R(r,r')^2+d_E(e,e')^2.
\]

This is exactly the metric implemented by the zero-temperature correspondence operator after factor-specific intrinsic normalization.

The additive product metric is **not** a claim that catalytic free energy or biochemical compatibility decomposes additively into a reaction term plus an enzyme term. It is the neutral product geometry obtained when the two factor coordinates are measured before a pairing is chosen. The biological coupling is carried by the relation \(\Gamma\) and its observed support \(\Omega\): only selected reaction/enzyme combinations occupy that relation. An off-diagonal cross-factor metric tensor would be a stronger pair-supervised modeling choice; the present data do not justify introducing one, and doing so would sacrifice the exact defect decomposition below.

For smooth strata, the same construction is the ordinary Riemannian product metric. For the finite implementation, only the corresponding metric identity is required; global differentiability is not.

## 2. The biochemical relation is a subset, not a function

The biologically complete object is context-conditioned:

\[
\Gamma\subset M_R\times M_E\times M_C,
\]

where \(M_C\) is assay/process context. For a fixed context \(c\),

\[
\Gamma_c=\{(r,e):(r,e,c)\in\Gamma\}\subset M_R\times M_E.
\]

Today's accepted registry is mostly not condition-resolved, so it should not be silently identified with one particular \(\Gamma_c\). Let \(\Gamma^\star\subset M_R\times M_E\) denote the **working catalytic relation** for the declared task. It may be a context-specific section \(\Gamma_c\) when context is known, or a stated projection such as “activity observed under at least one admissible assay context” when context is unresolved.

No global function \(e=g(r)\) or \(r=h(e)\) is assumed. Such a function would be biologically inappropriate because many enzymes can catalyze one reaction, one enzyme can catalyze several reactions, and different mechanism/family branches can realize similar transformations.

Define the set-valued reaction-to-enzyme fiber

\[
G^\star(r)=\{e:(r,e)\in\Gamma^\star\}
\]

and symmetrically the enzyme-to-reaction fiber

\[
H^\star(e)=\{r:(r,e)\in\Gamma^\star\}.
\]

The local regularity assumption needed for retrieval is only that, on one supported biological region and for one declared working relation, these closed fibers do not change arbitrarily fast. For R2E, assume locally

\[
d_H(G^\star(r),G^\star(r'))\le L_{R\to E}\,d_R(r,r'),
\]

where \(d_H\) is Hausdorff distance. E2R has the symmetric assumption with its own constant \(L_{E\to R}\).

This is a local bounded-variation assumption on a many-to-many correspondence, not a claim of a global one-to-one map. It also makes clear that changing assay context may change the relation itself; context variation is not hidden inside one universal pair label.

## 3. Accepted observations and the FIBRE decomposition

Let

\[
\Omega\subset\Gamma^\star
\]

be the finite set of accepted positive correspondences for the declared working relation. In the present context-unresolved deployment, \(\Gamma^\star\) is the task-level projected relation represented by the accepted registry, not a claim that every accepted pair belongs to every context-specific section. Define

\[
J_\Omega(r,e)
=
\min_{(r_i,e_i)\in\Omega}
\left[d_R(r,r_i)^2+d_E(e,e_i)^2\right].
\]

Because of the product metric,

\[
J_\Omega(r,e)=d_M((r,e),\Omega)^2.
\]

Define marginal support distances

\[
m_R(r)=d_R(r,\Omega_R)^2,
\qquad
m_E(e)=d_E(e,\Omega_E)^2.
\]

The FIBRE defect is

\[
\Delta_\Omega(r,e)
=
J_\Omega(r,e)-m_R(r)-m_E(e).
\]

### Proposition 1: non-negativity and exact decomposition

For every \((r,e)\),

\[
\Delta_\Omega(r,e)\ge0
\]

and

\[
\boxed{
J_\Omega
=
\Delta_\Omega+m_R+m_E.
}
\]

Proof. For each accepted pair \((r_i,e_i)\),

\[
d_R(r,r_i)^2\ge m_R(r),\qquad
d_E(e,e_i)^2\ge m_E(e).
\]

Therefore every joint precedent cost is at least \(m_R+m_E\), so its minimum is also at least that quantity. Rearranging gives the identity.

This proposition gives a precise interpretation:

- \(\Delta_\Omega\) measures the extra cost of requiring the two marginally familiar objects to share one biochemical precedent;
- \(m_R,m_E\) measure marginal novelty;
- \(J_\Omega\) is the absolute product-space distance to the nearest accepted joint precedent.

Therefore a small defect by itself is not a statement that the candidate is near experimental support. Candidate-level support/applicability is mathematically necessary, not auxiliary confidence decoration.

For finite \(\Omega\), the equality condition is also informative:

\[
\Delta_\Omega(r,e)=0
\]

if and only if at least one accepted joint pair simultaneously realizes the reaction-side marginal minimum and the enzyme-side marginal minimum. Thus the defect is exactly a **precedent-coherence gap**: it measures whether the two individually nearest supported neighborhoods can be explained by one observed biochemical correspondence rather than by unrelated precedents.

## 4. Why local observations can generalize

Two different statements are required: coverage of true pairs, and soundness of a proposed pair.

### Proposition 2: coverage from a dense accepted relation

Assume \(\Omega\subset\Gamma^\star\) and that on a covered compact region every true pair in the declared working relation is within product distance \(h\) of an accepted positive:

\[
\sup_{\gamma\in\Gamma^\star}d_M(\gamma,\Omega)\le h.
\]

Then for any \(z\) in that region,

\[
d_M(z,\Gamma^\star)
\le
d_M(z,\Omega)
\le
d_M(z,\Gamma^\star)+h.
\]

Hence

\[
d_M(z,\Gamma^\star)^2
\le
J_\Omega(z)
\le
\big(d_M(z,\Gamma^\star)+h\big)^2.
\]

In particular, if \(z\in\Gamma^\star\),

\[
J_\Omega(z)\le h^2,\qquad
\Delta_\Omega(z)\le h^2.
\]

Proof. The lower bound follows from \(\Omega\subset\Gamma^\star\). For the upper bound, choose a nearest true point \(\gamma\in\Gamma^\star\), then an accepted point within \(h\) of \(\gamma\), and apply the triangle inequality.

This is the finite-sample reason a sufficiently dense set of local precedents can cover unobserved true pairs.

### Proposition 3: local fiber soundness for a many-to-many relation

Assume the local R2E fiber map \(G^\star\) is Hausdorff-Lipschitz with constant \(L\). Then for any candidate \(z=(r,e)\) in the local chart,

\[
d_E(e,G^\star(r))
\le
\sqrt{1+L^2}\;d_M(z,\Gamma^\star).
\]

Because \(\Omega\subset\Gamma^\star\),

\[
d_M(z,\Gamma^\star)\le d_M(z,\Omega)=\sqrt{J_\Omega(z)}.
\]

Therefore

\[
\boxed{
d_E(e,G^\star(r))
\le
\sqrt{1+L^2}
\sqrt{\Delta_\Omega(r,e)+m_R(r)+m_E(e)}.
}
\]

The symmetric statement holds for E2R.

Proof. Let \((r',e')\in\Gamma^\star\) be a nearest relation point. Since \(e'\in G^\star(r')\),

\[
d_E(e,G^\star(r))
\le
d_E(e,e')+d_H(G^\star(r'),G^\star(r)).
\]

Using the local Hausdorff-Lipschitz condition,

\[
d_E(e,G^\star(r))
\le b+La,
\]

where \(a=d_R(r,r')\) and \(b=d_E(e,e')\). Cauchy-Schwarz gives

\[
b+La\le\sqrt{1+L^2}\sqrt{a^2+b^2}.
\]

The second inequality follows because accepted positives are contained in the true relation.

This is the core local-generalization statement. It does not say that low \(\Delta_\Omega\) alone proves activity. It says that under a locally regular catalytic correspondence, a candidate with both low correspondence defect and small marginal support distances must lie near the true reaction-conditioned enzyme fiber.

This also explains why extrapolative candidates require explicit support reporting: if \(m_R\) or \(m_E\) is large, the theorem gives no strong fiber guarantee even when \(\Delta_\Omega\) is small.

## 5. Stability to approximate factor geometry

The implementation does not know exact latent geodesics. It uses molecular graph approximations.

Suppose every squared reaction distance used by a query is approximated within \(\varepsilon_R\), and every squared enzyme distance within \(\varepsilon_E\). Minima are 1-Lipschitz under uniform perturbation, so

\[
|\widehat J-J|\le\varepsilon_R+\varepsilon_E,
\]

\[
|\widehat m_R-m_R|\le\varepsilon_R,
\qquad
|\widehat m_E-m_E|\le\varepsilon_E.
\]

Therefore

\[
\boxed{
|\widehat\Delta-\Delta|
\le
2(\varepsilon_R+\varepsilon_E).
}
\]

Thus graph-geodesic approximation quality transfers directly into a defect-error bound. FIBRE does not need a separate learned score to obtain this stability statement.

This is where standard graph-geodesic consistency results are relevant: if molecular observations densely sample a regular stratum and the neighborhood graph respects its local geometry, the graph metric can approach the intrinsic geodesic metric. FIBRE then inherits that approximation through the bound above.

### Proposition 4: exact out-of-sample extension on a frozen reference graph

Let \(G\) be the frozen reference length graph and let an external query \(q\) be attached to reference vertices \(a\in A_q\) with non-negative edge lengths \(\ell(q,a)\). Without adding any other query edges, the shortest-path distance from \(q\) to a reference vertex \(j\) is exactly

\[
d_{G\cup q}(q,j)
=
\min_{a\in A_q}
\left[
\ell(q,a)+d_G(a,j)
\right].
\]

Proof. Every path from \(q\) to the reference graph must first enter \(G\) through one attachment vertex \(a\); after that first edge, its shortest possible continuation is \(d_G(a,j)\). Minimizing over all possible first attachment vertices gives the expression above.

This is precisely the multi-source Dijkstra operation implemented by query_geodesic_to_reference. Therefore online FIBRE does not approximate or refit reference-reference geometry at query time. Its additional modeling error comes from the local attachment map itself, not from replacing the frozen graph shortest-path problem with a different operator.

## 6. Why the scientific output is a partial relation

Different molecular coordinates can approximate different physically meaningful aspects of the latent catalytic state. Let

\[
D(q,x)
=
(\Delta_1,\ldots,\Delta_k)
\]

be the currently validated defect family.

If candidate \(a\) Pareto-dominates candidate \(b\), then for every strictly positive linear weighting \(w_i>0\),

\[
\sum_i w_i\Delta_i(a)
<
\sum_i w_i\Delta_i(b).
\]

Therefore Pareto dominance is exactly the part of the comparison that is invariant to all positive choices of unknown scalarization weights. When two candidates trade off across coordinates, choosing a total order requires extra biological utility assumptions that are not currently identified by the data.

This gives a mathematical reason for the fixed-domain partial relation: it reports only weight-robust statements and leaves unsupported trade-offs unresolved.

## 7. Context is an extra biological coordinate of the relation, not a score bonus

The most faithful latent object is

\[
\Gamma\subset M_R\times M_E\times M_C,
\]

where \(M_C\) is assay/process context.

The current product relation \(\Gamma_c\) is a section at one context \(c\), while today's sparse positive registry is mostly a projection in which \(c\) is unobserved.

Because pair-specific context coverage is currently extremely sparse, FIBRE does not estimate a dense metric on \(M_C\). Instead it retains context as a sparse constraint:

- matched active observation: supported;
- matched inactive/below-detection observation: contradicted at that context;
- conflicting experiments: conflicting;
- missing or mismatched condition: unresolved.

When enough repeated condition-resolved assays become available, the mathematically natural extension is a context-conditioned correspondence on \(M_R\times M_E\times M_C\), not a pH/cofactor bonus added to the current score.

## 8. What is assumed and what is actually validated

The following are model assumptions, not established biological facts:

- local low-dimensional/regular factor strata;
- sufficiently accurate graph approximation to their intrinsic metrics;
- local Hausdorff-Lipschitz variation of the true catalytic fibers;
- adequate accepted-positive coverage radius in a queried region.

The repository currently validates only consequences that can be tested with available data:

- exact algebraic decomposition of the zero-temperature operator;
- exact dense/section parity;
- strict-inductive OOS reconstruction;
- local-coordinate partial-relation stability;
- exact positive-set updates;
- positive-only promiscuity holdout;
- explicit support-distance reporting.

A future stronger biological theorem would require estimating or bounding the local sampling radius \(h\), graph-geodesic distortion \(\varepsilon_R,\varepsilon_E\), and local fiber variation \(L\) on real biochemical neighborhoods. Those are now well-defined research quantities rather than unspecified manifold assumptions.

### Current positive-only assumption audit

The present repository does not estimate a true global \(h\) or \(L\), because the unobserved catalytic relation is unknown. It does, however, test the directional implication of locality on 1,555 leave-one-positive-out activities from 422 multi-activity enzymes. For each held-out true pair, we measure its product-space distance to the remaining accepted relation and its reaction-space distance to the nearest other verified activity of the same enzyme.

- median leave-one-out product distance to the remaining accepted relation: 0.823; 90th percentile: 1.021;
- product distance versus warm expected rank: Spearman \(\rho=0.664\);
- nearest same-enzyme reaction distance versus improvement from positive-context updating: \(\rho=-0.387\);
- the nearest same-enzyme-distance quartile improves 86.5% of held-out positives, whereas the farthest quartile improves 48.9%.

These are assumption diagnostics, not a proof of Hausdorff-Lipschitz biology and not deployment thresholds. They show that the observed positive-only continuation behavior is substantially stronger in intrinsically local neighborhoods, which is the qualitative behavior required by the local-fiber model. The machine-readable audit is projects/active/fibre/evaluation/local_fiber_geometry_audit.py.

## External mathematical basis

The factor-geometry assumptions are aligned with established manifold-learning and geometric-inference results rather than invented specifically for FIBRE:

- Bernstein, de Silva, Langford & Tenenbaum, Graph Approximations to Geodesics on Embedded Manifolds (2000/2001): consistency of neighborhood-graph distances with manifold geodesics under sampling/geometric conditions.
- Niyogi, Smale & Weinberger, Finding the Homology of Submanifolds with High Confidence from Random Samples, Discrete & Computational Geometry 39 (2008), 419–441: sampling guarantees expressed through a geometric condition number/reach controlling curvature and near self-intersection.
- Federer, Curvature Measures (1959): positive reach and unique nearest-point projection in a tubular neighborhood, providing a standard regularity language weaker than requiring global linearity.
- Set-valued analysis provides the standard language for locally Lipschitz/Aubin-continuous multifunctions; FIBRE uses the stronger symmetric Hausdorff-Lipschitz form only as an explicit local modeling assumption.

These references motivate the factor-space regularity assumptions only. The correspondence-defect decomposition, set-valued fiber bound and fixed-domain Pareto interpretation above are stated directly for the FIBRE object and do not depend on importing an external algorithm.
