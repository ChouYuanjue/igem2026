# BRIDGE Engineering Decision Tree

This tree is organized by technical causality rather than strict commit order. Every serious branch is kept, including rejected routes. Each leaf records the motivation, result, and what the attempt contributed to the next design decision.

Legend: **[KEEP]** retained in the current system; **[LOCAL]** retained only in a restricted role; **[REJECT]** tested and dropped; **[TURN]** rejected as a method but directly changed the next design.

```text
ROOT — TPS candidate screening under a limited wet-lab budget
│
├── 1. Closed candidate pool: can we rank a small TPS pool well?
│   │
│   ├── Reaction-similarity transfer
│   │   Motivation: similar chemistry can transfer known enzyme families.
│   │   Result: useful first candidate generator. -> [LOCAL] reaction-neighborhood evidence.
│   │
│   ├── Gate matrix / candidate gate
│   │   Motivation: remove obviously irrelevant proteins before expensive ranking.
│   │   Result: worked in closed tasks, but excluded positives could never be recovered. -> [TURN]
│   │
│   ├── CAGE-centered structural ranking
│   │   Motivation: use structure/pocket evidence to distinguish candidates inside the pool.
│   │   Result: strong early signal; later limited by scoreability and structural coverage. -> [LOCAL]
│   │
│   ├── RF / HGB meta-ranking
│   │   Motivation: combine reaction similarity, CAGE, and handcrafted evidence.
│   │   Result: useful on the original task, too tied to fixed features/pool. -> [REJECT as core]
│   │
│   └── Rescue slots
│       Motivation: reserve a few positions for strong evidence missed by the main ranker.
│       Result: useful principle. -> [TURN] precursor of bounded correction.
│
├── 2. Candidate expansion: can we recover positives outside the original pool?
│   │
│   ├── Few-shot / seed homology expansion
│   │   Motivation: known positives provide unusually strong local biological context.
│   │   Result: reliable when seeds exist. -> [KEEP] explicit context mode.
│   │
│   ├── MARTS external candidates
│   │   Motivation: move beyond the internal TPS dataset.
│   │   Result: expanded reaction/protein coverage and forced stricter evaluation. -> [KEEP]
│   │
│   ├── UniProt free merge
│   │   Motivation: maximize discovery coverage quickly.
│   │   Result: new unlabeled candidates displaced useful Top-K positions. -> [REJECT]
│   │
│   ├── Canonical prefix + tail quota
│   │   Motivation: expand coverage without destroying trusted head ranking.
│   │   Result: controlled rescue was stable. -> [LOCAL]
│   │
│   ├── Pfam single-domain gate
│   │   Motivation: exploit TPS family architecture.
│   │   Result: shared domains admitted unrelated proteins. -> [REJECT]
│   │
│   ├── Exact Pfam architecture / reaction-specific contract
│   │   Motivation: make family rules more specific.
│   │   Result: useful for rescue and audit, too narrow for universal ranking. -> [LOCAL]
│   │
│   ├── Taxonomy scope
│   │   Motivation: some users explicitly need eukaryotic/prokaryotic search boundaries.
│   │   Result: best handled as candidate-universe constraint. -> [KEEP]
│   │
│   └── Conformal / second-round candidate sets
│       Motivation: let candidate-set size follow uncertainty and experimental budget.
│       Result: useful coverage diagnostic, not the main ranking mechanism. -> [LOCAL]
│
├── 3. Open retrieval: candidate generation itself must become an ordered search
│   │
│   ├── ESM-C protein representation
│   │   Motivation: encode arbitrary proteins without fixed family IDs.
│   │   Result: robust open protein representation. -> [KEEP]
│   │
│   ├── DRFP reaction representation + reaction categories
│   │   Motivation: represent chemistry rather than database IDs.
│   │   Result: broad reaction-side representation. -> [KEEP]
│   │
│   ├── Multi-positive dual tower
│   │   Motivation: reactions and enzymes are many-to-many.
│   │   Result: became the Broad Retrieval foundation. -> [KEEP]
│   │
│   ├── PU / cluster mask
│   │   Motivation: unlabelled and near-homologous pairs may be false negatives.
│   │   Result: protects broad training from obvious false-negative pressure. -> [KEEP]
│   │
│   ├── Ordinary hard negatives
│   │   Motivation: random negatives were too easy.
│   │   Result: useful as training support. -> [KEEP]
│   │
│   ├── Dedicated hard-negative model / curriculum
│   │   Motivation: specialize aggressively for head confusion.
│   │   Result: no stable replacement for the main model. -> [REJECT]
│   │
│   └── Top-K surrogate loss
│       Motivation: optimize the actual wet-lab budget boundary directly.
│       Result: sensitive to K and false negatives; unstable across budgets. -> [REJECT]
│
├── 4. TPS-specific biochemical specialization: can mechanism features beat broad representations?
│   │
│   ├── Catalytic motif context
│   │   Motivation: DDxxD, NSE/DTE, DxDD and related motifs are mechanistically meaningful.
│   │   Result: local discrimination improved, global retrieval remained weak. -> [REJECT as core]
│   │
│   ├── Motif RRF / motif residual reranker
│   │   Motivation: use motifs only as second-stage evidence.
│   │   Result: dev gains did not survive freezing; best frozen weight returned to zero. -> [REJECT]
│   │
│   ├── P2Rank / pocket-local representation
│   │   Motivation: catalytic pockets should be closer to substrate selectivity than full sequence.
│   │   Result: some dev gain, failed frozen confirmation. -> [REJECT]
│   │
│   ├── Same-precursor / different-skeleton hard negatives
│   │   Motivation: create chemically realistic TPS confounders.
│   │   Result: reasonable training signal, no stable final gain. -> [REJECT]
│   │
│   ├── Mechanism auxiliary tasks / skeleton metric
│   │   Motivation: force the shared representation to retain precursor/topology/oxidation information.
│   │   Result: labels were too coarse or too sparse. -> [REJECT]
│   │
│   ├── Coarse skeleton classes
│   │   Motivation: supervise product skeleton identity explicitly.
│   │   Result: occasional dev wins, unstable freezing. -> [REJECT]
│   │
│   ├── Morgan reaction clusters
│   │   Motivation: use finer chemical clustering than coarse skeleton labels.
│   │   Result: unstable benefit. -> [REJECT]
│   │
│   ├── Carbon-connectivity graph labels / residual
│   │   Motivation: model the core carbon rearrangement made by TPS enzymes.
│   │   Result: residual versions again converged toward zero frozen correction. -> [REJECT]
│   │
│   └── TPS active-site cross-attention
│       Motivation: learn direct interaction between reaction features and active-site tokens.
│       Result: hard-pool/residual/HPO route later exposed evaluation-alignment problems. -> [REJECT]
│
├── 5. External pretrained-model transfer: can larger models replace or strengthen Broad?
│   │
│   ├── Horizyn global MLNCE transfer
│   │   Motivation: inherit a larger pretrained enzyme-reaction space.
│   │   Result: did not stably beat Broad. -> [REJECT]
│   │
│   ├── Frozen Horizyn reaction encoder
│   │   Motivation: keep a pretrained reaction geometry fixed and adapt the other side.
│   │   Result: insufficient. -> [REJECT]
│   │
│   ├── ESM-C -> ProtT5 bridge
│   │   Motivation: align protein spaces expected by different pretrained systems.
│   │   Result: did not solve the ranking problem. -> [REJECT]
│   │
│   ├── DRFP + Horizyn direct concatenation
│   │   Motivation: combine local chemistry and pretrained reaction features directly.
│   │   Result: unstable scale/interaction. -> [REJECT]
│   │
│   └── Reaction distillation residual / exact residual
│       Motivation: preserve Broad and import only incremental pretrained information.
│       Result: useful in restricted budgets and established the residual principle. -> [LOCAL/TURN]
│
├── 6. Graph, geometry, and non-parametric alternatives
│   │
│   ├── Geometry alignment
│   │   Motivation: align reaction-space and protein-space internal geometry.
│   │   Result: promiscuity, convergent function, and sparse labels break one-to-one geometry. -> [REJECT]
│   │
│   ├── Graph diffusion / multi-hop propagation
│   │   Motivation: propagate known function through the enzyme-reaction graph.
│   │   Result: hub amplification and poor double-cold entry. -> [REJECT]
│   │
│   ├── Candidate-hub normalization
│   │   Motivation: suppress proteins that score highly for almost every query.
│   │   Result: several normalizers failed to give stable gains. -> [REJECT]
│   │
│   └── Dual kernel: protein neighborhood + reaction neighborhood
│       Motivation: require support on both biological and chemical neighborhoods.
│       Result: current-only version failed freezing; restricted MARTS E2R Top-20 worked. -> [LOCAL]
│
├── 7. Fusion, reliability, and bidirectional diagnostics
│   │
│   ├── Raw score addition
│   │   Motivation: simplest ensemble.
│   │   Result: incompatible score scales. -> [REJECT]
│   │
│   ├── Percentile / tied-rank fusion
│   │   Motivation: normalize scales before fusion.
│   │   Result: useful only in limited settings. -> [LOCAL]
│   │
│   ├── Reciprocal Rank Fusion
│   │   Motivation: combine relative rank instead of raw score.
│   │   Result: stable in selected E2R budgets. -> [LOCAL]
│   │
│   ├── Fixed three-source fusion
│   │   Motivation: exploit complementary unique hits across Pfam/kernel/experts.
│   │   Result: dev improved, frozen performance fell. -> [REJECT/TURN]
│   │
│   ├── Three-seed ensemble
│   │   Motivation: reduce optimization variance.
│   │   Result: retained as a robustness measure. -> [KEEP]
│   │
│   ├── Reliability calibration / abstention
│   │   Motivation: expose risk instead of pretending raw scores are probabilities.
│   │   Result: retained as interpretation/risk layer. -> [KEEP]
│   │
│   ├── Conformal retrieval sets
│   │   Motivation: adapt returned set size to empirical coverage.
│   │   Result: useful diagnostic, not ordering authority. -> [LOCAL]
│   │
│   └── R2E <-> E2R cycle consistency
│       Motivation: high ranking in one direction should support the reverse direction.
│       Result: no stable new hits after confirmation. -> [REJECT as ranker, LOCAL diagnostic]
│
├── 8. Broad generalization and anti-forgetting
│   │
│   ├── Directional continuation
│   │   Motivation: absorb broader enzyme-reaction data while preserving TPS capability.
│   │   Result: established the generalization problem. -> [TURN]
│   │
│   ├── Historical replay / embedding anchor
│   │   Motivation: preserve old-domain representation during new-domain training.
│   │   Result: useful retention controls, not final architecture. -> [LOCAL]
│   │
│   ├── LwF
│   │   Motivation: preserve old model outputs.
│   │   Result: explored, not final. -> [REJECT as architecture]
│   │
│   ├── Score distillation
│   │   Motivation: preserve old ranking behavior directly.
│   │   Result: explored, not final. -> [REJECT as architecture]
│   │
│   ├── Margin-MSE
│   │   Motivation: preserve pairwise ranking margins.
│   │   Result: no decisive production advantage. -> [REJECT]
│   │
│   ├── RecAdam
│   │   Motivation: regularize early training toward the source parameters.
│   │   Result: tested as low-forgetting continuation. -> [LOCAL]
│   │
│   ├── Checkpoint blending / WiSE-FT
│   │   Motivation: move back toward the source model when the broad model forgets old capability.
│   │   Result: found Pareto points, but did not solve conditional expertise. -> [LOCAL]
│   │
│   ├── Fisher merging / RegMean / TIES / AdaMerging
│   │   Motivation: merge domain-specific capabilities in parameter space.
│   │   Result: less stable than ranking-level organization. -> [REJECT]
│   │
│   └── Post-hoc domain routing
│       Motivation: different models may be useful for different query regions.
│       Result: first strong evidence that expert value is conditional. -> [TURN]
│
├── 9. Novelty and reaction-center branches
│   │
│   ├── Low-similarity novelty expert / replay
│   │   Motivation: specialize for reaction-novel queries.
│   │   Result: routing/replay lacked stable general benefit. -> [REJECT]
│   │
│   ├── Functional-prototype residual
│   │   Motivation: correct Broad from learned functional prototypes.
│   │   Result: failed formal screening. -> [REJECT]
│   │
│   ├── Reaction-center V1
│   │   Motivation: represent only atoms/bonds that actually change.
│   │   Result: failed hard-slice gate. -> [REJECT]
│   │
│   ├── Reaction-center residual / identity-preserving residual
│   │   Motivation: make mechanism evidence incremental and safely absent.
│   │   Result: more stable. -> [TURN]
│   │
│   └── Bounded reaction-center V3
│       Motivation: cap how much a local mechanistic expert may move Broad.
│       Result: confirmed. -> [KEEP] direct ancestor of bounded expert correction.
│
├── 10. EnzGFM and local reranking
│   │
│   ├── Native EnzGFM baseline
│   │   Motivation: introduce a strong pretrained enzyme-reaction model.
│   │   Result: strong baseline/expert. -> [KEEP]
│   │
│   ├── EnzGFM + RDKit / RDKit+
│   │   Motivation: add explicit reaction chemistry to foundation-model evidence.
│   │   Result: entered the multi-expert candidate set. -> [KEEP]
│   │
│   ├── EnzGFM + reaction center
│   │   Motivation: combine foundation representations and mechanistic local change.
│   │   Result: useful expert source. -> [LOCAL]
│   │
│   ├── Top-2000 pair reranker / residual / bounded residual
│   │   Motivation: correct only the head instead of rescoring the universe.
│   │   Result: strengthened the limited-prefix principle. -> [TURN]
│   │
│   └── Difficulty-aware / reaction-center gate
│       Motivation: run a second stage only when the query needs it and evidence exists.
│       Result: specific variants rejected, query-level gating survived. -> [TURN]
│
├── 11. BiME-Rank: organize experts systematically
│   │
│   ├── R2E expert candidate union
│   │   Motivation: let independent retrievers contribute candidates before fusion.
│   │   Result: retained. -> [KEEP]
│   │
│   ├── R2E LambdaRank stack
│   │   Motivation: learn how raw score, rank, agreement, and novelty should combine.
│   │   Result: confirmed and frozen. -> [KEEP]
│   │
│   ├── R2E similarity router
│   │   Motivation: different novelty regions prefer different routes.
│   │   Result: deterministic fallback retained. -> [KEEP]
│   │
│   ├── E2R four-expert unrestricted LambdaRank
│   │   Motivation: copy R2E multi-expert success to E2R.
│   │   Result: damaged strong EnzGFM head ranking. -> [REJECT/TURN]
│   │
│   ├── Baseline-anchored rescue / Anchored LambdaMART V3
│   │   Motivation: protect the strong base and rerank only a limited prefix.
│   │   Result: retained. -> [KEEP] direct BRIDGE ancestor.
│   │
│   ├── CLIPZyme structural expert
│   │   Motivation: add 3D structural evidence.
│   │   Result: admitted with availability-aware fallback. -> [KEEP]
│   │
│   ├── Seed-context expert
│   │   Motivation: use known positives when the task provides them.
│   │   Result: conditionally admitted. -> [KEEP]
│   │
│   ├── Homology-context expert
│   │   Motivation: expose homology as an explicit common expert.
│   │   Result: failed external retention. -> [REJECT]
│   │
│   ├── Reciprocal-consistency expert
│   │   Motivation: promote bidirectional agreement into a formal expert.
│   │   Result: failed external retention. -> [REJECT]
│   │
│   ├── Generic CAGE Top-20 expert
│   │   Motivation: reintroduce mature structural scoring as a universal expert.
│   │   Result: OOF metrics fell. -> [REJECT/TURN]
│   │
│   └── Cost-aware hierarchical execution
│       Motivation: expensive experts cannot score the full 185k-protein universe.
│       Result: cheap experts search broadly, expensive experts run on shortlist. -> [KEEP]
│
├── 12. FIBRE detour: can one unified relational geometry replace expert assembly?
│   │
│   ├── Biological relation stratification / mechanistic strata / partial relation
│   │   Motivation: represent richer biology than a binary positive pair.
│   │   Result: launched the unified-model line. -> [HISTORICAL]
│   │
│   ├── Tensor-product field
│   │   Motivation: model reaction x enzyme interaction explicitly.
│   │   Result: expressive but difficult to justify as ranking core. -> [HISTORICAL]
│   │
│   ├── Interaction Atlas
│   │   Motivation: interpret experts as local charts of a shared catalytic space.
│   │   Result: coherent theory, weak necessity for production ranking. -> [HISTORICAL]
│   │
│   ├── Context-restricted domains + atlas gluing
│   │   Motivation: make local charts agree where their domains overlap.
│   │   Result: consistency assumption was too strong; gluing loss removed. -> [REJECT]
│   │
│   ├── Catalytic kernel / conditional catalytic kinetics
│   │   Motivation: tie abstract geometry to real catalytic interpretation.
│   │   Result: better interpretation, still not a stronger ranking core. -> [HISTORICAL]
│   │
│   ├── Conditional modes
│   │   Motivation: activate different latent catalytic modes by query.
│   │   Result: became the main FIBRE selection tree. -> [HISTORICAL]
│   │   │
│   │   ├── Fully symmetric joint potential -> lost direction information. [REJECT]
│   │   ├── Geometric-mean two-sided gates -> unstable. [REJECT]
│   │   ├── Symmetric linear mixture -> unstable. [REJECT]
│   │   ├── Gibbs / log-partition aggregation -> no stable gain. [REJECT]
│   │   ├── KL barycenter -> no stable gain. [REJECT]
│   │   ├── Molecular joint-potential compensation -> insufficient. [REJECT]
│   │   ├── Expert-variance fallback -> R2E degraded. [REJECT]
│   │   ├── Normalized conditional mixture -> R2E degraded strongly. [REJECT]
│   │   ├── Dimension-scaled consistency -> no advantage over deleting consistency. [REJECT]
│   │   ├── Linear conditional expectation -> most stable FIBRE mode. [LOCAL historical]
│   │   ├── Log-Mean-Exp -> dev acceptable, frozen failed. [REJECT]
│   │   ├── Second-order variance correction -> failed development. [REJECT]
│   │   └── Reaction-center posterior -> structurally cleaner, frozen R2E failed. [REJECT]
│   │
│   ├── Scientific-evidence layer
│   │   Motivation: anchor structure/mechanism/context on top of a stable core.
│   │   Result: missing=zero and anchored additive evidence survived. -> [TURN]
│   │
│   ├── Heterogeneous conditional modes
│   │   Motivation: put EnzGFM, reaction center, CLIPZyme, and seed context in one graph.
│   │   Result: unified execution worked, unified ranking core still not superior. -> [TURN]
│   │
│   ├── End-to-end query-adaptive mixture
│   │   Motivation: let each query learn expert weights.
│   │   Result: damaged R2E. -> [REJECT]
│   │
│   ├── E2R-only adaptive gate
│   │   Motivation: isolate adaptation to the direction that seemed safer.
│   │   Result: shared training still hurt R2E. -> [REJECT]
│   │
│   ├── Frozen-core post-hoc gate
│   │   Motivation: freeze the strong core and learn only a light query gate.
│   │   Result: preserved R2E but gains were not strong enough. -> [TURN]
│   │
│   ├── ERAM relational core + UniMol
│   │   Motivation: replace hand-designed FIBRE geometry with a broad relational learner.
│   │   Result: trained and evaluated, still failed to replace Broad. -> [REJECT as core]
│   │
│   └── Pluggable adapters / frozen context plugin
│       Motivation: add new evidence without retraining the universal core.
│       Result: plugin and fallback principles survived. -> [TURN]
│
├── 13. Return to Broad: redefine experts as bounded evidence
│   │
│   ├── Rebind all expert evidence to Broad Core
│   │   Motivation: Broad remained the most reliable global order.
│   │   Result: universal ordering authority returned to Broad. -> [KEEP]
│   │
│   ├── Expert-as-pair-evidence
│   │   Motivation: experts should contribute local evidence instead of new global geometry.
│   │   Result: retained. -> [KEEP]
│   │
│   ├── Missing = neutral
│   │   Motivation: unavailable structure/mechanism is absence of evidence, not negative evidence.
│   │   Result: retained as a hard contract. -> [KEEP]
│   │
│   ├── Direction-specific score permission
│   │   Motivation: the same expert may be useful in R2E but unsafe in E2R, or vice versa.
│   │   Result: ranking permission became directional. -> [KEEP]
│   │
│   ├── Expert-type hierarchy
│   │   Motivation: organize foundation, structural, mechanistic, contextual, and family evidence.
│   │   Result: retained. -> [KEEP]
│   │
│   ├── Dynamic expert router V4
│   │   Motivation: make expert value query-dependent rather than globally fixed.
│   │   Result: established query-conditioned expert weighting. -> [TURN]
│   │
│   └── Dynamic router V6 / applicability permission
│       Motivation: distinguish experts that may rerank, experts that may only report evidence,
│                   and experts that must remain silent.
│       Result: became the final applicability contract. -> [KEEP]
│
├── 14. CAGE and TPS return as specialists
│   │
│   ├── Broad -> generic CAGE reranking
│   │   Motivation: test whether CAGE failed only because the old candidate pool was narrow.
│   │   Result: Broad found many positives CAGE could not natively score. -> [REJECT/TURN]
│   │
│   ├── P450 CAGE specialist
│   │   Motivation: test whether CAGE still has value inside a coherent family.
│   │   Result: local gain. -> [KEEP]
│   │
│   ├── Phosphatase CAGE specialist
│   │   Motivation: same family-specific hypothesis.
│   │   Result: local gain. -> [KEEP]
│   │
│   ├── Terpene CAGE specialist
│   │   Motivation: recover local structural value in the original domain.
│   │   Result: local gain. -> [KEEP]
│   │
│   └── TPS specialist correction
│       Motivation: reuse the earliest TPS-specific knowledge without narrowing the global model.
│       Result: sparse activation only inside TPS applicability. -> [KEEP]
│
└── 15. BRIDGE
    │
    ├── Broad Retrieval owns the default global order. [KEEP]
    ├── Expert availability is explicit. [KEEP]
    ├── Expert applicability is query- and direction-specific. [KEEP]
    ├── Missing expert evidence is neutral. [KEEP]
    ├── Experts receive bounded correction authority only. [KEEP]
    ├── Family specialists are allowed to be locally strong without becoming universal. [KEEP]
    └── TPS and CAGE close the loop as specialists inside a broad system. [KEEP]
```

## Final engineering interpretation

The development history converged through four repeated failures:

1. **Closed candidate pools limited recall.** This forced open retrieval.
2. **Unordered expansion damaged Top-K quality.** This forced Broad Retrieval to become a real ranker.
3. **Fixed or unrestricted expert fusion damaged strong base rankings.** This forced anchored and bounded corrections.
4. **Experts were useful only for some queries, directions, or evidence states.** This forced query-level applicability and produced BRIDGE.

FIBRE remains important because it tested the strongest alternative hypothesis: replacing the modular ranking stack with a unified relational geometry. Its failure to outperform Broad as the universal ordering core, together with the survival of missing-neutral evidence, plugins, and query-conditioned routing, directly shaped the final BRIDGE design.
