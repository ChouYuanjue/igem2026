# Biological Interpretability Contract for the Product-Manifold Method

## Principle

The retrieval method remains one scalar compatibility field `F(r,e)` on the product manifold `M_R × M_E`. Biological knowledge is not implemented as a collection of post-hoc score experts. Instead, biologically meaningful measurements are divided into two roles:

1. **Dense geometric coordinates**, which may enter the intrinsic metrics `g_R` and `g_E` because they cover the numerical support sufficiently well.
2. **Partial biological observables**, which annotate and audit the same manifold but do not alter ranking unless they later satisfy coverage and non-regression requirements.

This distinction is deliberate. Missing Pfam, motif, structure, or pocket evidence is absence of observation, never evidence against catalysis.

## Biological meaning of the two factors

### Reaction factor `M_R`

The reaction geometry is not a single fingerprint score. Its local coordinates describe complementary biochemical scales:

- global reaction transformation by DRFP;
- whole-reaction physicochemical structure by RDKitPlus;
- atom-mapped local reaction-centre change.

The local metric precision of each coordinate is determined by how strongly that coordinate resolves the current chemical neighbourhood. Flat coordinates vanish continuously; there is no reaction-class gate.

### Protein factor `M_E`

The dense protein geometry currently uses:

- ESM-C sequence representation as a complete sequence-level coordinate;
- CLIPZyme structure-dependent representation where structure is observed.

For unit-normalized coordinates, averaging the two observed cosine similarities is equivalent, up to a global scale absorbed by self-tuning bandwidth, to the direct sum of their squared chordal metrics. Thus the protein geometry can be read as a sequence/structure atlas rather than an expert ensemble.

The following biologically meaningful observations are retained as partial coordinates rather than universal ranking rules at current coverage:

- exact sequence alignment / homology;
- Pfam architecture and domain family;
- terpene-synthase catalytic motifs such as DDxxD, NSE/DTE, DxDD, and QW;
- P2Rank / pocket / 3Di observations.

## Biological witness

For every predicted pair `(r,e)`, the system may return a **biological witness**: an observed fold-train positive pair `(r_i,e_i) ∈ Ω` with low local product-support energy

`E_support((r,e),(r_i,e_i)) = E_R(r,r_i) + E_E(e,e_i)`.

`E_R` uses the same multiscale local reaction geometry as the retrieval method. `E_E` uses the same self-tuning sequence/observed-structure protein geometry. The witness is therefore an explanation of the existing geometric support, not a second scoring model.

After the witness is selected geometrically, human-readable biological observables may be attached: sequence-alignment identity, Pfam/domain annotation, motif state, structure similarity, and pocket evidence when present. These annotations never feed back into the rank.

## First label-free witness example

The first explanation probe was deliberately run on the model's own Top-1 prediction rather than a hand-selected dev positive.

- query reaction: `RHEA:10020`
- canonical v8 Top-1 protein: `P0DPE4`
- nearest observed support witness: `RHEA:34111 – Q9KRL3`
- query/source reaction similarities: DRFP `0.6964`, RDKitPlus `0.8248`, mapped reaction centre `0.8000`
- predicted/source protein similarities: ESM-C cosine `0.9973`, CLIP structural cosine `0.9916`
- BLOSUM62 global alignment sequence identity: `0.8406`

This gives an experimentally intelligible statement: the prediction is supported by a known enzyme-reaction pair that is simultaneously close in reaction chemistry and enzyme sequence/structure.

Artifact: `results/product_manifold_biological_witness_v1/fold0/RHEA_10020__P0DPE4.json`.

## Promotion rule for biological observables

A partial biological observable may enter an intrinsic metric only if all of the following hold:

- it has enough coverage to define geometry on the relevant numerical charts rather than a tiny selected submanifold;
- missingness can remain neutral without a special fallback branch;
- its inclusion has a clean mathematical interpretation as a coordinate/metric observation rather than a score expert;
- the frozen internal protocol shows no material regression.

A biologically meaningful observable that fails these conditions is still useful as a witness or diagnostic. It is not forced into ranking merely because it is interpretable.

This rule explains the current decisions:

- exact Pfam combination is biologically useful and had a small historical frozen gain, but the broader hierarchical single-domain rule regressed and is not used as a universal metric;
- homology context improved internal development but failed strict external retention badly, so homology is a witness, not a ranking expert;
- motif-only / corrected active-site reranking did not give stable non-regressing retrieval improvement, so motifs remain mechanistic observables;
- pocket geometry is currently too sparse in the canonical 432-node charts to define a universal protein metric, so it remains a partial structural observable.

## Two evidence layers

Biological evidence is split by scope rather than forced into one universal feature vector.

### General Rhea evidence

For arbitrary enzyme–reaction retrieval, `product_manifold_biological_witness.py` reports support geometry and nearest known positive precedents using exactly the same reaction/protein coordinates as the product field. It may attach sequence/structure evidence when observed. This layer is universal and ranking-neutral.

The reaction/protein locality coordinates describe the **support regime**, not calibrated confidence. On the first 64 unselected fold-0 canonical predictions their Spearman correlations with reciprocal rank are only `0.047` and `0.013`; no confidence or uncertainty interpretation is permitted from these quantities alone. See `BIOLOGICAL_SUPPORT_AUDIT.md`.

### Registered terpene application evidence

For registered terpene reactions, `terpene_mechanism_sheet.py` adds reaction-conditioned biological observations that are meaningful only in this application domain: substrate/product names, terpene type, TPS class, architecture observed among known positive reference enzymes, direct candidate-to-reference sequence homology, and family-aware motif contexts.

This layer does **not** inherit the historical `allowed_candidate_architectures` gate. A candidate architecture may be observed or unobserved among references, but that fact is reported rather than converted into a hard admissibility decision.

Family-aware motif scanning is evidence-only and intentionally permits known class-I aspartate-rich variants such as `DDXX(D/E)` and `DDXXX(D/E)`. It is separate from frozen historical motif-descriptor assets. Motif absence, variant motifs, or missing annotation never changes ranking.

The two layers can coexist because both are observations attached to the same predicted pair. Neither is a second ranking model.
