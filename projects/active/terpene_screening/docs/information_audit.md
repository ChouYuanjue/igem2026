# Information-utilization audit

The goal under scarce enzyme–reaction labels is not to create more experts; it is to ensure that every available observation has an explicit place in the same product-manifold model.

| Information | Role in one object | Coverage | Status |
|---|---|---|---|
| DRFP global reaction difference | reaction metric coordinate | all 11,081 canonical reactions | active |
| RDKitPlus whole-reaction physicochemical structure | reaction metric coordinate | all canonical reactions with deterministic features | active |
| atom-mapped reaction-center geometry | reaction metric coordinate | mapped canonical reaction support | active |
| ESM-C protein representation | protein metric coordinate | 185,918 / 185,918 proteins | active |
| CLIPZyme structure-dependent representation | protein metric coordinate when observed | partial, high chart coverage on current benchmark | active; missing-neutral |
| known enzyme–reaction pairs | joint empirical positive measure | fold-train positives only | active |
| sequence homology to known enzymes | biological observable / witness | full sequence library where reference sequence exists | active interpretation; not an independent scorer |
| Pfam/domain architecture | family-conditional biological observable | sparse/general + strong TPS application assets | active interpretation; no global gate |
| catalytic motifs | family-conditional mechanistic observable | only applicable enzyme families | active interpretation; no global gate |
| predicted pocket / pocket-3Di / pocket-OT | candidate protein metric refinement | ~1,282 proteins in current pocket asset; near-zero occupancy in typical 432-node benchmark charts | deferred from global metric because coverage too sparse; still used when inspecting applicable proteins |
| curated reaction→reference-enzyme architecture | application-specific biological observable | 208 / 240 registered terpene reactions with reference architecture evidence | active application evidence; no rerank |

## Promotion rule

An unused biological observation should be incorporated into `g_R` or `g_E` when it supplies a comparable pairwise geometry on enough of the relevant support to do so without turning missingness into a gate. If it is genuinely family-conditional or too sparse, it remains an observable on the same manifold until coverage justifies metric promotion. Known positive pairs always remain one empirical measure on the product space.
