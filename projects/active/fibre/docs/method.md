# FIBRE — Factorized Interaction Basis for Reaction–Enzyme

## 1. The data problem determines the model

For enzyme discovery, rich information about each object is much easier to obtain than reliable pair labels.

A reaction can often be described from substrates, products, atom mapping, changed bonds, reaction centres and physicochemical descriptors. A protein can be described from sequence, protein-language-model state, family context and, when available, structure, pocket, motifs and cofactors. In contrast, experimentally supported enzyme–reaction correspondences are sparse.

FIBRE therefore does not try to learn one monolithic pair classifier from a dense interaction table. It uses rich per-object information to construct local coordinate descriptions of catalysis, and uses sparse verified pairs to connect the reaction and enzyme sides.

## 2. The geometric object is an interaction atlas

Let \(\mathcal R\) be valid chemical transformations and \(\mathcal E\) valid enzyme molecular inputs. Physical catalytic activity is condition-dependent; the more complete object is conceptually

\[
A(r,e;c),
\]

where \(c\) denotes assay, cellular and environmental context. Current pair registries are not dense measurements of this object. They are sparse, selectively observed records whose context is often incomplete. FIBRE therefore targets a declared task-level **working catalytic compatibility**

\[
K^\star:\mathcal R\times\mathcal E\rightarrow\mathbb R,
\]

which should not be interpreted as a universal context-free law of enzyme activity. A reaction is a directed transformation \(r:x\rightarrow y\), and an enzyme provides a molecular environment capable of stabilizing and organizing particular transformations.

FIBRE covers the interaction domain by biochemical and information regimes

\[
\{U_\alpha\}_{\alpha\in A},\qquad
\mathcal R\times\mathcal E=\bigcup_{\alpha} U_\alpha.
\]

Each \(U_\alpha\) is a local chart in which a particular collection of molecular observations is informative. Examples include a broad sequence/whole-reaction chart, reaction-centre charts, structure-aware charts, pocket/mechanism charts and a terpene-synthase family chart.

Inside chart \(\alpha\),

\[
\phi_\alpha(r)\in H^R_\alpha,\qquad
\psi_\alpha(e)\in H^E_\alpha,
\]

are reaction-demand and enzyme-capability coordinates, and the local catalytic interaction is

\[
K_\alpha(r,e)=
\phi_\alpha(r)^\top G_\alpha\psi_\alpha(e).
\]

The two coordinate spaces may have different meanings and dimensions. They are paired by \(G_\alpha\); they are not required to be one common embedding space.

## 3. Local charts estimate one working interaction; retrieval readouts may be directional

Charts overlap. In an ideal pair-aware atlas, local interaction estimates are glued by non-negative weights

\[
\rho_\alpha(r,e)\ge0,\qquad
\sum_{\alpha\in A(r,e)}\rho_\alpha(r,e)=1,
\]

where \(A(r,e)\) is the set of charts both available and applicable to that pair. The pairwise atlas estimator is

\[
\widehat K(r,e)=
\sum_{\alpha\in A(r,e)}
\rho_\alpha(r,e)K_\alpha(r,e).
\]

The frozen gated multi-expert realization is deliberately more specific. It shares the same local chart scores \(K_\alpha(r,e)\), but R2E and E2R use different query-side partitions:

\[
S_d(r,e)=\sum_\alpha \rho_\alpha^{(d)}(r,e)K_\alpha(r,e),
\qquad d\in\{\mathrm{R2E},\mathrm{E2R}\}.
\]

R2E currently uses reaction-side gates and its learned expert mass; E2R uses protein-side gates and its own learned expert mass. Therefore FIBRE does **not** claim that the frozen implementation satisfies \(S_{\mathrm{R2E}}(r,e)=S_{\mathrm{E2R}}(r,e)\) pointwise. The direction-independent object is the task-level compatibility being estimated, not an unverified equality between two deployed retrieval readouts.

For each direction, the previous implementation's global/expert convex mixture is exactly a partition of unity. If \(m_d\in(0,1)\) is the learned expert mass and \(g_k^{(d)}\) are softmax expert gates, then

\[
\rho_0^{(d)}=1-m_d,\qquad
\rho_k^{(d)}=m_d g_k^{(d)},
\qquad
\rho_0^{(d)}+\sum_k\rho_k^{(d)}=1.
\]

The global channel is therefore chart \(0\), not a privileged score to which experts add residuals. This is an algebraic re-expression of the existing inference function, so adopting the atlas ontology does not by itself perturb any score. The atlas-native training recipe additionally applies overlap consistency to the universal chart and all active local charts.

The two directional readouts also admit a useful deterministic bound. If \(p\) and \(q\) are their partitions over the same active chart scores and \(K_{\min},K_{\max}\) are the minimum and maximum active local scores, then

\[
|S_p-S_q|
\le \frac12\|p-q\|_1
(K_{\max}-K_{\min}).
\]

Thus direction disagreement can arise only from partition disagreement, chart-score disagreement, or both. Stronger overlap agreement directly tightens the effect of using different directional partitions. The bound is implemented by `partition_readout_discrepancy_bound`.

A broad raw-input chart is defined for every valid reaction/protein input. Therefore the cover is complete even when optional structure, pocket, family or mechanism views are absent. Missing optional charts receive zero mass and the remaining partition is renormalized. FIBRE never needs to reject a valid input.

## 4. Multiple representations are intrinsic coordinates

Different measurements observe different biochemical aspects of the same object.

Reaction-side views include whole-reaction DRFP, substrate/product molecular states, signed molecular change, atom-mapped bond changes, reaction-centre neighbourhoods and descriptors. Protein-side views include ESM-C, EnzGFM, sequence family state, structure, pocket and catalytic-context features.

These views are not concatenated merely because they are available. A chart chooses the representations meaningful for its biochemical regime and learns its own coordinates and interaction form. Consequently:

- representation dimensions may differ;
- a structure chart can be absent without becoming a negative;
- a reaction-centre chart may dominate where local chemistry matters;
- a family chart can use family-specific information without shrinking the candidate universe;
- global and local charts can coexist and agree on their overlaps.

The current multiview and gated multi-expert lineages are practical approximations of this atlas construction.

## 5. Expert specialization is biochemical, not dataset membership

Terpene-synthase specialization is represented by a TPS chart

\[
U_{\mathrm{TPS}}\subset\mathcal R\times\mathcal E.
\]

Its applicability is determined by molecular and family evidence: TPS-like protein state, compatible precursor/reaction chemistry and other TPS-specific observations. It is not defined by whether an identifier came from MARTS or another dataset.

Therefore TPS specialization can operate inside a large general candidate universe. Where TPS information is strongly applicable, the TPS chart receives substantial partition mass; where it is not applicable, its mass tends to zero. The broad chart remains available everywhere.

The same principle applies to structure, pocket, motif and mechanistic experts.

## 6. Factorization inside each chart

Each local interaction \(K_\alpha\) can have low effective mechanistic rank,

\[
K_\alpha(r,e)
\approx
\sum_{k=1}^{d_\alpha}
\sigma_{\alpha k}
u_{\alpha k}(r)v_{\alpha k}(e).
\]

This is the mathematical reason for dual-tower and local-expert factorizations. A finite set of interaction modes approximates a much larger combinatorial enzyme–reaction space.

An unseen reaction and unseen enzyme remain scoreable because chart coordinates are functions of molecular input rather than table IDs.

## 7. Sparse pair data are the bridge

The scarce enzyme–reaction pairs do not create the molecular coordinates from scratch. They teach:

1. the local interaction forms \(G_\alpha\);
2. which charts are informative in which biochemical regimes;
3. consistency between overlapping charts;
4. ranking and calibration of the resulting retrieval readouts.

Unknown pairs are not automatically biological negatives.

On chart overlaps, a natural gluing loss is

\[
\mathcal L_{\mathrm{glue}}
=
\sum_{\alpha<\beta}
\rho_\alpha\rho_\beta
\left(K_\alpha-K_\beta\right)^2.
\]

It encourages two charts that both claim applicability to agree on the same task-level catalytic compatibility without forcing their latent coordinates to be metrically identical.

## 8. Train-free incorporation of new experiments

For accepted external observations \((r_i,e_i)\), an applicable chart can form the finite-rank empirical interaction update

\[
\Delta_\alpha(\Omega)
=
\sum_i
\underbrace{w_i\rho_\alpha(r_i,e_i)}_{\eta_{i\alpha}}
\phi_\alpha(r_i)\psi_\alpha(e_i)^\top.
\]

If there are \(n\) accepted observations, then

\[
\operatorname{rank}(\Delta_\alpha)\le n.
\]

FIBRE applies a declared Frobenius budget \(\|\Delta_\alpha\|_F\le\varepsilon_\alpha\). For unit-norm chart coordinates, the induced score change obeys

\[
|\phi_\alpha(r)^\top\Delta_\alpha\psi_\alpha(e)|
\le \|\Delta_\alpha\|_2
\le \|\Delta_\alpha\|_F
\le \varepsilon_\alpha.
\]

The kernel now implements this rectangular chart-local primitive as `FiniteRankInteractionUpdate`, so reaction and enzyme chart dimensions need not match. The older `PositivePairConditioner` remains the already-used square shared-latent realization. Passing \(w_i\rho_\alpha\) as the update weights realizes the chart-local statistic itself; automatic application-layer projection of every external observation into every applicable chart remains a separate orchestration step and is not claimed as completed here.

Because the update acts on the local interaction form rather than the molecular encoders, verified observations can refine a chart without retraining those encoders. Experimental source, endpoint and assay context remain attached to the observation and are exposed as evidence.

## 9. Existing components under the atlas

The existing implementation assets map naturally to the same object:

- broad ESM-C plus reaction dual towers: universal chart;
- DRFP, multiview and reaction-centre representations: alternative reaction coordinates;
- EnzGFM and CLIPZyme: protein/family/structure chart coordinates;
- gated multi-expert towers: learned local charts with softmax partition weights;
- seed-context experts: observation-conditioned charts when verified pair context exists;
- TPS pair-supervised models: TPS-family charts;
- evidence and assay pipelines: provenance attached to chart applicability and final interpretation.

Rejected experts remain useful evidence about proposed charts that failed to transport under frozen evaluation.

## 10. Evidence and confidence

A result should expose which charts contributed, their partition weights, their local scores, which molecular views were present, and which experimental/database observations support the interpretation.

A probability-like confidence is reported only when calibrated for the matching evaluation population. Otherwise FIBRE reports interaction support, chart agreement/disagreement, view coverage and provenance separately.

## 11. Reproduction and application

The FIBRE Reproduction Bundle freezes paired data, split, candidate support, model assets and evaluator. Only this profile supports benchmark-performance claims.

The Starase Application Bundle may use all accepted molecular information, validated charts, full-data TPS specialization and provenance-bound wet-lab/database evidence. Application observations never flow backward into a frozen benchmark.

## 12. Historical and experimental boundary

The old reaction-manifold × enzyme-manifold product-correspondence geometry is historical and is not the current FIBRE ontology. It remains an important derivational stage: it correctly exposed sparse positive correspondence, missing-as-unknown semantics and the desire for a shared relation, but its global factor-geometry assumptions proved too restrictive for heterogeneous, partially observed molecular views. The interaction atlas keeps the useful relational ideas while relaxing the requirement that all views define one global molecular metric.

The previously tested global bilinear correction \(I+B\) is also not part of the current multi-expert construction: its fixed three-fold result was mixed and it was not promoted. Current FIBRE organizes specialization through local interaction charts and partition-of-unity gluing instead of additive residual fusion.
