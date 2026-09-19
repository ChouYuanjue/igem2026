# Information-utilization audit

The goal under scarce enzyme–reaction labels is not to create more experts; it is to ensure that every available observation has an explicit place in the same product-manifold model.

| Information | Role in one object | Coverage | Status |
|---|---|---|---|
| DRFP global reaction difference | reaction metric coordinate | all 11,081 canonical reactions | active |
| RDKitPlus whole-reaction physicochemical structure | reaction metric coordinate | all canonical reactions with deterministic features | active |
| atom-mapped reaction-center transition geometry | catalytic-local reaction coordinate inside coarse FIBRE levels | 453 / 453 application reactions | active finer resolution; does not deform global reaction geodesic |
| ESM-C protein representation | protein metric coordinate | 185,918 / 185,918 proteins | active |
| whole-structure 3Di | global protein molecular-state coordinate when observed | 1,287 / 1,421 application proteins | active; missing-neutral |
| known enzyme–reaction pairs | joint empirical positive measure | fold-train positives only | active |
| sequence homology to known enzymes | biological observable / witness | full sequence library where reference sequence exists | active interpretation; not an independent scorer |
| Pfam/domain architecture | family-conditional biological observable | sparse/general + strong TPS application assets | active interpretation; no global gate |
| catalytic motifs | family-conditional mechanistic FIBRE coordinates | type-I 827; NSE/DTE 647; DXDD 142; QW 27 / 1,421 | active third resolution; no scalar bonus or global gate |
| pocket-local ESM-C / pocket-3Di / pocket-OT | catalytic-pocket FIBRE coordinates | 1,287 / 1,068 / 1,287 of 1,421 application proteins | active second resolution; Pareto catalytic strata are currently non-order-bearing after strict-inductive promotion audit |
| curated reaction→reference-enzyme architecture | application-specific biological observable | 208 / 240 registered terpene reactions with reference architecture evidence | active application evidence; no rerank |

## Promotion rule

An available biological observation does not have to be forced into the global metrics `g_R` or `g_E`. It may instead define a finer correspondence resolution inside a coarse numerical level. Promotion to the global metric or to an order-bearing local linearization requires missingness neutrality, a coherent geometric role, matched double-cold evidence and strict-inductive non-degradation. Family-conditional motif coordinates may remain mechanistic strata indefinitely without being reduced to a universal score. Known positive pairs always remain one correspondence set on the same product space.
