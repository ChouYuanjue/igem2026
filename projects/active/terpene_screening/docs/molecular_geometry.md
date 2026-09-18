# Multiresolution molecular-state geometry under scarce pair labels

## Central statistical fact

**The supervision is sparse; the molecular information is not.**

Experimentally established enzyme–reaction pairs are much scarcer than unlabeled molecular observations. A candidate protein may have a full sequence, predicted or experimental global structure, catalytic-motif neighbourhoods, a predicted active pocket, local pocket sequence, local pocket structure, or several of these at once. Requiring every protein to possess the same observation set would throw away exactly the information that should compensate for sparse pair labels.

The protein factor is therefore a **partially observed multiresolution manifold**, not a fixed feature vector and not a panel of ranking experts.

## One protein state, several measurement maps

Let `e in M_E` be one underlying molecular state. Each available assay or representation is a measurement map

`phi_m : U_m subset M_E -> X_m`,

where `U_m` is the subset for which that measurement exists. Current useful maps include:

- global sequence state from ESM-C;
- whole-structure topology from 3Di/Foldseek;
- pocket-local sequence state;
- pocket-local 3Di structure;
- a P2Rank-weighted distribution of pocket structures compared by exact balanced OT;
- catalytic-motif residue-context embeddings when a biologically applicable motif is observed.

Pfam/domain labels and reaction-conditioned known reference enzymes remain annotations of the same point unless and until they define a continuous, coverage-safe geometric coordinate. They are never hard admissibility gates.

## Missing-neutral pullback geometry

For each view `m`, convert the observed pairwise dissimilarity into a dimensionless local energy using a self-tuning scale,

`E_m(i,j) = d_m(i,j)^2 / (sigma_m(i) sigma_m(j))`.

For a pair of proteins, let `O_ij` be only the views observed at both endpoints. Define

`E_E(i,j) = mean_{m in O_ij} E_m(i,j)`.

This is implemented by `partial_observation_pullback_affinity` in `multiscale_geometry.py`.

The arithmetic mean is not a tuned modality mixture. It is the direct-sum pullback energy after each coordinate system is locally nondimensionalized, normalized by the number of coordinates actually observed for that pair. It has useful invariances:

1. for a fixed sampled observation atlas, an unavailable view contributes no pair energy;
2. merely having more modalities does not create an availability bonus;
3. duplicating an identical view leaves the energy unchanged;
4. changing the physical/numerical scale of one view by a positive constant is absorbed by its local scale;
5. no learned or manually selected inter-view coefficient exists.

Pairs with no jointly observed coordinate in a view receive no contribution from that view. Absence of information is not dissimilarity. When a previously unobserved molecular point is later measured, the self-tuning scale of nearby *observed* points may change because the sampled manifold itself has become denser. This is a geometric re-estimation of local scale, not negative evidence from the former missing value.

## Making structural scores geometrically honest

Raw Foldseek similarity and the pocket-OT-derived similarity are not asserted to satisfy metric axioms. They first define nonnegative observation graphs. We then use exact one-step diffusion distance

`D_1(i,j)^2 = sum_k (P_ik - P_jk)^2 / pi_k`,

which is a Euclidean/pseudometric distance between Markov neighbourhoods. The structural observation therefore enters the molecular-state geometry through the topology it induces, rather than by relabelling a heuristic similarity score as a distance.

## Current label-free terpene atlas

The current frozen application-domain protein geometry is `data/terpene_multiresolution_protein_geometry_v4`. It is built on the fixed 1,421-protein terpene universe **without reading enzyme–reaction labels**.

The base coordinate is now complete: global ESM-C covers **1,421 / 1,421** proteins. Higher-resolution measurements refine that same molecular-state geometry wherever they are actually observed:

- global ESM-C chordal geometry: 1,421 / 1,421;
- current-P2Rank-backed pocket-local ESM-C: 1,287 / 1,421;
- family-aware type-I aspartate-rich contextual coordinate: 827 / 1,421;
- family-aware NSE/DTE contextual coordinate: 647 / 1,421;
- family-aware DXDD contextual coordinate: 142 / 1,421;
- family-aware QW contextual coordinate: 27 / 1,421;
- whole-3Di diffusion geometry: 1,287 / 1,421;
- pocket-3Di diffusion geometry: 1,068 / 1,421;
- pocket-OT diffusion geometry: 1,287 / 1,421.

Because some motif coordinates are family-specific and mutually non-applicable, no claim is made that every protein should possess all nine defined views. The current number of observed coordinates per protein is:

- 59 proteins with 8 views;
- 502 with 7;
- 309 with 6;
- 319 with 5;
- 98 with 4;
- 13 with 3;
- 6 with 2;
- 115 with the global sequence coordinate only.

There are **no isolated protein points**. The local graph uses the intrinsic `ceil(sqrt(1421)) = 38` neighbourhood scale and contains 33,908 undirected edges before/after the conductance-preserving diffusion-conformal correction; median degree is 43. No pair label is used to choose an edge, a view, or an inter-view weight.

### Provenance is part of information utilization

“Use all available information” means **all current, auditable molecular observations**, not every historical file that happens to exist. Two clean-up operations are therefore part of the method rather than bookkeeping:

- the global coordinate combines 1,414 historically aligned ESM-C vectors with seven newly computed sequence vectors, giving 1,421 / 1,421 coverage. Eight duplicate accession aliases were retained only after their global-vector cosine agreement was at least 0.99999988;
- the pocket-local coordinate is tied to the **current** P2Rank top-1 pocket semantics. It contains 711 historical vectors whose old snippet is byte-identical to the snippet regenerated from the current pocket, 567 newly filled vectors, and nine vectors recomputed after current P2Rank residues changed. Thirteen stale historical alias/semantic observations are explicitly excluded.

Thus data abundance is exploited without turning stale provenance into artificial biological evidence.

### Catalytic local coordinates without another tuning scale

The motif coordinates do not reuse the old concatenated `global + motif` feature vector and do not introduce a manually chosen flanking window. ESM-C residue states are contextualized by the full sequence; for each biologically applicable motif occurrence we average only the contextual embeddings on the matched motif span, normalize each occurrence, then equally average multiple occurrences. Family applicability comes from the canonical 1,421-row `protein_architecture_annotations.csv`. A family/motif that is not observed is a missing coordinate.

## Relation to scarce labels

The protein geometry above uses unlabeled molecular measurements. The reaction factor is constructed analogously from global reaction difference, whole-reaction physicochemical structure, and mapped reaction-centre observations. Sparse known positive enzyme–reaction pairs then form the empirical measure `mu` on `M_R x M_E`.

Thus the method uses abundant side information to define **where biochemical evidence is allowed to propagate**, while the scarce positive pairs determine **which compatibility observations are known**. No synthetic negatives are needed, and missing measurements are never converted into negative labels.

This is the intended meaning of “use all available information.”
