# BRIDGE Engineering Storyline

This document separates **design evolution** from **current BRIDGE composition**.
The historical route is organized as decisions. Concrete experiments remain attached to the problem they tested;
evaluation records are treated as evidence; BRIDGE expert families are described again in the final architecture because they are current components rather than peer historical generations.

Live Engineering page: `https://nju-igem.runnelzhang.com/engineering/`

## 01 · Structure ranking worked only after the right candidates were already present.

**Problem.** We began by trying to improve EnzymeCAGE for TPS-like retrieval. Pocket choice was worth checking, but the larger failure appeared when structural pair scores had to search an entire protein library.

**What we learned.** Reaction-similarity transfer made the bounded task much better, yet every downstream ranker still inherited the candidate gate. On the full relation set, only about 43.98% of known enzyme records entered that gate.

**Decision.** Candidate coverage became the scientific bottleneck. The next system had to admit new reactions and proteins from molecular inputs rather than from membership in a precomputed relation table.

### Inside the bounded candidate system

We tried to improve recall, structural ranking and rescue without changing the closed-world assumption.

- **High-recall candidate union** `[TURN]` — Efficient, but excluded positives became unrecoverable. _Forced open candidate-space thinking._
- **CAGE pool-internal structural ranking** `[LOCAL]` — Important early signal; limited by support coverage and raw-score behavior. _Returned later as family-specific structural expertise._
- **Conditional CAGE rescue** `[LOCAL]` — Useful as a bounded rescue rather than a universal score. _Early availability-aware structure pattern._
- **Direct raw CAGE probability fusion** `[REJECT]` — Sigmoid saturation and ties manufactured unstable fine order. _Rejected; structure had to be handled more cautiously._
- **Tree-based score fusion** `[REJECT]` — Useful on the old fixed-pool task, brittle outside it. _Did not become the broad core._
- **Main ranking + rescue slots** `[TURN]` — Practical and interpretable. _Early ancestor of anchored / bounded correction._

## 02 · Open-world retrieval replaced candidate gating with a full-space order.

**Problem.** Once unseen enzymes or reactions had to enter the system, candidate generation could no longer be a hidden preprocessing step. It had to become a scored retrieval problem of its own.

**What we learned.** We explored candidate registries, protein/reaction representations, TPS-specific mechanism cues, uncertainty and graph propagation in parallel. The stable common denominator was a bidirectional molecular representation that could rank the full candidate universe even when optional evidence was absent.

**Decision.** Broad Retrieval became the default global order. Specialized evidence could still exist, but it could no longer define who was allowed into the search space.

### Candidate universe

Ways to admit new entities without silently losing positives.

- **Candidate space & open-world registry** `[KEEP]` — Established explicit candidate-universe semantics. _Feeds Broad and remains a BRIDGE boundary._
- **Persistent + temporary open-world registry** `[KEEP]` — Worked for persistent and request-local entities. _Retained as open-world infrastructure._
- **MARTS external reaction/enzyme expansion** `[KEEP]` — Expanded external coverage and forced stricter evaluation. _Retained in training/evaluation lineage._
- **Seeded few-shot expansion** `[KEEP]` — Very strong under homolog-visible conditions. _Retained as an explicit contextual capability._
- **Same/near-homolog seed expansion** `[KEEP]` — High Top-K retrieval when close homologs are allowed. _Kept as a distinct task, not mixed with zero-shot._
- **Cross-cluster seed expansion** `[LOCAL]` — Much harder but scientifically distinct. _Reported separately from homolog-visible few-shot._
- **UniProt TPS expansion** `[TURN]` — Exposed direct competition between coverage and Top-K quality. _Led to controlled expansion rather than free merge._
- **Free merge of 5,672 extra proteins** `[REJECT]` — Many unlabeled proteins displaced known positives in scarce Top-K. _Rejected._
- **Canonical prefix + controlled tail quota** `[LOCAL]` — Stable risk-controlled expansion. _Retained as bounded rescue._
- **Pfam architecture constraints** `[TURN]` — Fine-grained architecture helped; coarse domain logic overgeneralized. _Kept only as scoped evidence._
- **Pfam soft reranking** `[REJECT]` — Development gains were not stable. _Rejected._
- **Exact Pfam architecture** `[LOCAL]` — Small stable local gain. _Retained for rescue/audit._
- **Hierarchical single-domain + full architecture** `[REJECT]` — Shared domains mixed unrelated function on frozen evaluation. _Rejected._
- **R2E taxonomy scope** `[KEEP]` — Pre-score candidate restriction preserved model semantics. _Retained as a candidate-universe constraint._
- **Semantic candidate-universe routing** `[LOCAL]` — Useful application routing. _Kept as orchestration, not model evidence._

### Broad representations and training

Learn a continuous reaction–enzyme order directly from molecular inputs.

- **Broad representation & retrieval learning** `[KEEP]` — Produced the first robust open retrieval core. _Primary parent of Broad Retrieval._
- **ESM-C protein representation** `[KEEP]` — Provided open protein-side features. _Retained._
- **DRFP + reaction categories** `[KEEP]` — Supported unseen reaction identities. _Retained._
- **Multi-positive bidirectional dual tower** `[KEEP]` — Became the durable broad retrieval backbone. _Direct ancestor of Broad Retrieval._
- **Cluster-aware false-negative protection** `[KEEP]` — Reduced destructive negative pressure. _Retained._
- **Hard-negative training** `[LOCAL]` — Useful in moderation; aggressive specialisation became unstable. _Partially retained._
- **Dedicated hard-negative model** `[REJECT]` — No stable frozen superiority. _Rejected._
- **Hard-negative curriculum** `[REJECT]` — Failed to become a stable production recipe. _Rejected._
- **Budget-focused ranking loss** `[REJECT]` — Sensitive to K and false negatives. _Rejected from the main objective._
- **MARTS domain adaptation** `[KEEP]` — Improved domain coverage. _Retained in production ancestry._
- **Direction-specific tower adaptation** `[LOCAL]` — Useful because the two directions are not symmetric. _Foreshadowed direction-specific expert permissions._
- **Three-seed ensemble** `[KEEP]` — Improved robustness diagnostics. _Retained._
- **Horizyn transfer family** `[TURN]` — Direct transfer mostly failed; residual use was safer. _Established “strong base + incremental external evidence”._
- **Direct Horizyn contrastive transfer** `[REJECT]` — Negative cleanroom result. _Rejected._
- **Frozen Horizyn reaction encoder** `[REJECT]` — Insufficient ranking quality. _Rejected._
- **ESM-C → ProtT5 bridge** `[REJECT]` — Did not solve retrieval performance. _Rejected._
- **DRFP + Horizyn concatenation** `[REJECT]` — Unstable feature-scale interaction. _Rejected._
- **Bounded Horizyn residual transfer** `[LOCAL]` — Useful in restricted budgets and reproducible as a distiller. _Residual principle survived._

### TPS mechanism as a stress case

Biologically plausible local features were useful diagnostics, but most did not survive broad frozen evaluation.

- **TPS mechanistic specialization** `[TURN]` — Many biologically appealing features improved local slices but failed broad frozen confirmation. _Eventually returned as a gated TPS specialist._
- **Catalytic motif context** `[REJECT]` — Local discrimination improved; global retrieval remained weak. _Rejected as a core ranker._
- **Rank-fused motif evidence** `[REJECT]` — Development gain did not freeze. _Rejected._
- **Motif residual reranker** `[REJECT]` — Frozen optimum returned to zero correction. _Rejected._
- **Pocket-local representation** `[REJECT]` — Some development gain; frozen confirmation failed. _Rejected._
- **Mechanism-matched hard negatives** `[REJECT]` — Semantically good but no stable frozen gain. _Rejected._
- **Mechanism-aware auxiliary supervision** `[REJECT]` — Labels were too coarse/sparse. _Rejected._
- **Explicit scaffold supervision** `[TURN]` — Occasional development wins did not survive freezing. _Informed later mechanism-evidence caution._
- **Coarse scaffold classes** `[REJECT]` — Unstable. _Rejected._
- **Morgan reaction clusters** `[REJECT]` — Unstable. _Rejected._
- **Carbon-connectivity graph** `[REJECT]` — No stable frozen improvement. _Rejected._
- **Carbon-graph residual** `[REJECT]` — Frozen scale returned to zero. _Rejected._
- **TPS foundation R2E model** `[TURN]` — Provided a domain-specialist ancestry. _Fed the later TPS expert, not the universal core._
- **Active-site cross-attention** `[REJECT]` — Development winner was invalidated by evaluation alignment; corrected route did not become mainline. _Rejected as universal route._
- **Support-aware EnzymARC gate** `[LOCAL]` — Useful as a support-aware external gate, too expensive / limited for universal scoring. _Retained as comparative/support evidence._

### Reliability and applicability

Estimate when a prediction or evidence source should be trusted rather than forcing every signal into rank.

- **Evidence, applicability & uncertainty** `[KEEP]` — Created diagnostics that stayed orthogonal to ranking. _Several concepts later helped expert gating._
- **Grouped reliability calibration** `[KEEP]` — Produced protocol-bound reliability tiers. _Retained as a risk layer._
- **Selective abstention** `[KEEP]` — Useful application behavior. _Retained outside core ranking._
- **Candidate Evidence Passport** `[LOCAL]` — Made decisions auditable. _Retained as provenance/interpretation._
- **Open-world applicability proxy** `[TURN]` — Separated applicability from activity probability. _Conceptually anticipated expert applicability._
- **Conformal Retrieval Sets** `[LOCAL]` — All global calibration combinations passed finite-sample tolerance; sets were often large. _Retained as a diagnostic/coverage contract._
- **Applicability-aware Mondrian conformal** `[LOCAL]` — Enabled only where calibration/test support was sufficient. _Retained with fallback to global calibration._
- **Bidirectional cycle consistency** `[LOCAL]` — Useful explanation signal. _Kept as evidence, not ranking authority._
- **Cycle rerank weight grid** `[REJECT]` — Confirmation produced no added hits or MRR gain. _Rejected from production ranking._

### Graph and non-parametric alternatives

Propagation and kernels were tested as alternate retrieval mechanisms and local rescue signals.

- **Graph & non-parametric alternatives** `[TURN]` — Some restricted routes worked, global graph methods were unstable. _Contributed local auxiliary methods._
- **Graph diffusion** `[REJECT]` — Hub amplification and weak double-cold entry. _Rejected._
- **Multi-hop propagation** `[REJECT]` — Amplified noise and hub effects. _Rejected._
- **Candidate hub normalization** `[REJECT]` — Multiple normalizers failed to produce stable gains. _Rejected._
- **Protein + reaction dual kernel** `[TURN]` — Current-only version failed freezing; budget-limited MARTS E2R succeeded. _Kept only under narrow conditions._
- **Current-only dual kernel** `[REJECT]` — Development gain did not freeze. _Rejected._
- **MARTS dual-kernel rescue** `[LOCAL]` — Passed repeated confirmation. _Retained as a local route._

## 03 · The next problem was no longer coverage. It was how to use extra evidence without damaging a strong base order.

**Problem.** Broad gave us a stable full-space ranking, but EnzGFM, structural models, reaction-center signals, context and TPS knowledge still contained complementary information. Direct continuation and fixed fusion repeatedly traded one regime for another.

**What we learned.** The useful signals differed by direction, support and query. Routing and anchored learning-to-rank worked better than unconditional fusion. External and temporal tests also made clear that validation had to remain separate from model selection.

**Decision.** We moved from “one model should absorb everything” toward an explicit portfolio of experts around a protected broad retriever. That architecture became BiME-Rank.

### Generalization and retention

Attempts to expand Broad while preserving its existing ranking behavior.

- **Budget-aware retrieval routing** `[KEEP]` — Separate production routes outperformed one universal route. _Retained as budget-aware routing._
- **Broad generalization & anti-forgetting** `[TURN]` — Parameter-space consolidation alone did not solve conditional expertise. _Pushed the system toward expert routing._
- **Continuation method scouting** `[CONSIDERED]` — Separated low-cost candidates from high-cost advanced backups. _Recorded as design-space evidence._
- **L2-SP** `[CONSIDERED]` — Documented as a retention baseline; not promoted into the frozen mainline. _Considered route._
- **DER** `[CONSIDERED]` — Studied as a mechanism reference; not promoted into the frozen mainline. _Considered route._
- **FDR** `[CONSIDERED]` — Studied as a retention reference; not promoted into the frozen mainline. _Considered route._
- **Model Soups** `[CONSIDERED]` — Considered alongside WiSE-FT; no independent frozen production branch. _Considered route._
- **RegMean++** `[CONSIDERED]` — Documented as a candidate, not promoted into the frozen production line. _Considered route._
- **Full-evidence directional continuation** `[TURN]` — Expanded coverage but created explicit retention trade-offs. _Established the retention problem._
- **Historical replay anchor** `[LOCAL]` — Useful retention control. _Technique retained locally._
- **Checkpoint blending** `[LOCAL]` — Found useful Pareto points. _Could not express query-specific expertise._
- **RecAdam** `[LOCAL]` — Tested as a low-forgetting continuation route. _Local technique only._
- **Score-retention distillation** `[REJECT]` — No decisive production advantage. _Rejected as final architecture._
- **Bidirectional score distillation** `[REJECT]` — Useful experiment, no final architectural win. _Rejected._
- **Margin-MSE retention** `[REJECT]` — No decisive production advantage. _Rejected._
- **Fisher consolidation** `[REJECT]` — Did not outperform ranking-level organization. _Rejected._
- **RegMean consolidation** `[REJECT]` — Did not become the production solution. _Rejected._
- **TIES merging** `[REJECT]` — Less stable than routing experts at score/rank level. _Rejected._
- **AdaMerging** `[REJECT]` — Did not solve bidirectional conditionality robustly. _Rejected._
- **LoRA continuation** `[CONSIDERED]` — Documented as a candidate; not promoted into the frozen production line. _Kept as literature/engineering option._
- **Task-vector merging** `[CONSIDERED]` — Considered lower priority; not a frozen main experiment. _Recorded as a considered route._
- **Curvature-aware regularization** `[CONSIDERED]` — Considered advanced backup only. _Not promoted into production experiments._
- **Post-hoc domain expert routing** `[TURN]` — Showed expert value depends on query region. _Direct ancestor of query applicability._
- **Low-similarity novelty branch** `[REJECT]` — Could not survive clean frozen confirmation. _Rejected._
- **Novel-reaction replay** `[REJECT]` — Unstable. _Rejected._
- **Novelty expert + similarity route** `[REJECT]` — No robust general gain. _Rejected._

### Candidate expert evidence

Functional, structural and mechanistic signals tested as additions to Broad.

- **Expert evidence families** `[KEEP]` — Produced functional, structural and mechanistic experts. _Feeds BiME and BRIDGE._
- **EnzGFM native baseline** `[KEEP]` — Strong baseline and expert source. _Retained._
- **EnzGFM + RDKit** `[LOCAL]` — Useful on cleanroom slices. _Continued to RDKit+._
- **EnzGFM with reaction features** `[KEEP]` — Passed broad cleanroom protocols in selected roles. _Retained as expert evidence._
- **Equal-block feature fusion** `[LOCAL]` — Useful controlled fusion. _Local implementation detail._
- **Reaction-center mechanistic branch** `[TURN]` — Standalone mechanism ranking failed; bounded residual succeeded. _Key ancestor of bounded corrections._
- **Direct reaction-center fusion** `[REJECT]` — Failed the hard-slice promotion gate. _Rejected._
- **Identity-preserving center residual** `[TURN]` — Stable enough to confirm. _Moved to bounded V3._
- **Bounded reaction-center correction** `[KEEP]` — Passed frozen confirmation. _Direct BRIDGE design ancestor._
- **Shortlist pair reranking** `[TURN]` — Localized computation and reduced damage to tail order. _Led to difficulty-aware and anchored reranking._
- **Pairwise shortlist correction** `[LOCAL]` — Useful development signal. _Kept as a local mechanism._
- **Bounded shortlist correction** `[LOCAL]` — Passed a fold-specific confirmation. _Strengthened the bounded-correction principle._
- **Difficulty-aware shortlist routing** `[KEEP]` — Confirmed. _Kept as a routing precursor._
- **Reaction-center shortlist gate** `[REJECT]` — Comprehensive integration was rejected. _Specific gate dropped; routing idea survived._
- **Functional-prototype residual** `[REJECT]` — Failed the formal selector. _Rejected._
- **CLIPZyme structural branch** `[TURN]` — Useful only when native inputs/support were respected. _Became an availability-aware structural expert._
- **Native CLIPZyme baseline** `[KEEP]` — Established a fair structural baseline. _Retained as expert ancestry._
- **CLIPZyme encoder substitution** `[REJECT]` — Changed model semantics; reverted. _Rejected._
- **Structure-support audits** `[KEEP]` — Identified which candidates could be scored faithfully. _Retained as availability contract._
- **Directed CLIPZyme fallback** `[KEEP]` — Passed fallback contract tests. _Direct predecessor of missing-neutral expert semantics._
- **ReactZyme transfer branch** `[TURN]` — Native adapter reproduced author semantics, retention failed. _Useful negative evidence for universal transfer._
- **Native molecule-bag adapter** `[LOCAL]` — Confirmed adapter behavior. _Kept for reproducible comparison._
- **ReactZyme retention policy** `[REJECT]` — Retention gate rejected it. _Rejected._

### From score fusion to learned routing

How independent evidence was combined before BiME-Rank.

- **Raw score addition** `[REJECT]` — Score scales were incompatible. _Rejected._
- **Tied-rank percentile fusion** `[LOCAL]` — Useful in selected routes. _Kept locally._
- **Reciprocal Rank Fusion** `[LOCAL]` — Stable for selected E2R Top-10 routes. _Retained locally._
- **Dual-neural E2R rank fusion** `[LOCAL]` — Survived repeated confirmation for Top-10. _Retained as a budget-specific route._
- **Fixed three-source fusion** `[TURN]` — Development improved; frozen performance fell. _Proved complementarity does not imply global fixed weights._
- **Early TPS LambdaRank candidate stacking** `[REJECT]` — Too little/unstable data in the early TPS setting. _Rejected then later revisited on broad clean data._
- **Expert candidate union** `[KEEP]` — Improved support without requiring one universal retriever. _Retained in BiME._
- **R2E similarity router** `[KEEP]` — Confirmed deterministic routing. _Retained._
- **Joint reliability gate** `[LOCAL]` — Improved safety constraints. _Fed expert admission logic._

### Frozen evidence that constrained the design

- **External & temporal stress tests** — Several routes failed only when moved to external/fresh support.
- **Leakage-safe nested cleanroom selection** — Became mandatory selection discipline.
- **Pure EnzymeCAGE same-support baseline** — Established auditable apples-to-apples comparison.
- **Enzyme-405 same-support comparison** — Provided fair external comparison.
- **Orphan-335 retrieval stress test** — Provided an external broad-retrieval challenge.
- **TIGER reaction-novel baseline** — Useful for fair comparison.
- **Fresh temporal transfer test** — Transfer failures exposed support/alignment limits.
- **TPS MARTS R2E symmetry confirmation** — Passed its dedicated confirmation.
- **Strict temporal benchmark** — Became part of final model evaluation discipline.

## 04 · Expert admission was useful, but “globally admitted expert” was still too coarse.

**Problem.** BiME organized multiple capabilities, candidate union, cost and fallback. Its remaining weakness was conceptual: an expert could pass globally and still be harmful for a particular query or direction.

**What we learned.** Anchored E2R rescue protected a strong baseline; generic CAGE degraded ranking while seed and structural context helped only under specific support. These results pointed toward conditional expert rights rather than one global admission decision.

**Decision.** Two routes were tested from here: unify the whole interaction into one relational model (FIBRE), or keep Broad authoritative and make expert rights increasingly local. The FIBRE detour was large enough to deserve its own chapter.

### Protect the strong direction-specific baseline

E2R experiments showed why unconstrained expert fusion could not own the entire ranking.

- **E2R expert-fusion branch** `[TURN]` — Unrestricted fusion damaged a very strong EnzGFM base. _Forced explicit base protection._
- **Four-expert E2R LambdaRank** `[REJECT]` — Rejected after head-ranking degradation. _Rejected._
- **Baseline-anchored E2R rescue** `[TURN]` — Safer than unrestricted fusion. _Led to Anchored LambdaMART._
- **Anchored E2R learning-to-rank** `[KEEP]` — Passed confirmation. _Direct ancestor of BRIDGE base-order protection._

### Which experts deserve ranking authority?

Formal admission, context evidence and structural experts exposed the difference between availability and usefulness.

- **Formal expert admission** `[KEEP]` — Made expert inclusion auditable. _Retained conceptually._
- **CLIPZyme structural expert** `[KEEP]` — Passed admission. _Retained._
- **Seed context expert** `[KEEP]` — Conditionally admitted. _Retained._
- **Multi-seed context** `[KEEP]` — Validated in BiME. _Retained as conditional context._
- **Homology context expert** `[REJECT]` — Failed external retention. _Rejected._
- **Reciprocal consistency expert** `[REJECT]` — Failed external retention. _Rejected._
- **Generic EnzymeCAGE expert** `[TURN]` — Internal OOF ranking degraded. _Critical evidence that structural expertise is conditional._

### Execution and incumbent protection

Cost-aware execution and strong-baseline absorption became explicit system policies.

- **Cost-aware hierarchical execution** `[KEEP]` — Cheap experts search broadly; expensive experts run on shortlists. _Retained in final execution philosophy._
- **Strong-baseline absorption policy** `[KEEP]` — Formalized conservative integration. _Direct BRIDGE precursor._

## 05 · FIBRE tested whether one relational geometry could replace the multi-module stack.

**Problem.** The expert stack felt inelegant, so we tried to represent biological relations, mechanisms and heterogeneous evidence inside a unified relational formulation.

**What we learned.** The detour produced useful ideas—explicit scientific evidence, frozen-core gating, pluggable adapters and missing-neutral fallback—but repeated conditional-mode and relational-core experiments did not displace Broad under strict generalization tests.

**Decision.** The failed replacement clarified the final abstraction: preserve the global Broad order and let heterogeneous evidence act through explicit, query-conditioned interfaces.

### Relation geometry

Ways to encode biological relation, catalytic context and interaction structure.

- **Biological relation stratification** `[HISTORICAL]` — Established FIBRE relation semantics. _Historical foundation._
- **Mechanistic relation strata** `[HISTORICAL]` — Integrated into FIBRE observations. _Historical._
- **Partial biological relation** `[HISTORICAL]` — Implemented in the FIBRE observation model. _Historical._
- **Context-restricted FIBRE domain** `[HISTORICAL]` — Improved semantic honesty. _Conceptually survived as applicability._
- **Tensor-product atlas field** `[HISTORICAL]` — Implemented and evaluated. _Historical._
- **Interaction Atlas** `[HISTORICAL]` — Became a full FIBRE formulation. _Historical._
- **Atlas gluing loss** `[REJECT]` — Cross-expert consistency assumptions were too strong. _Removed._
- **Catalytic kernel formulation** `[HISTORICAL]` — Improved theory, not empirical ranking dominance. _Historical._
- **Conditional catalytic kinetics** `[HISTORICAL]` — Used to refine theory. _Historical._
- **Hierarchical mode uncertainty** `[HISTORICAL]` — Theoretical refinement. _Historical._

### Conditional modes

A dense series of attempts to reconcile asymmetric experts inside one probabilistic/conditional formulation.

- **FIBRE conditional modes** `[HISTORICAL]` — Became the largest FIBRE model-selection subtree. _Historical selection hub._
- **Cross-expert consistency penalty** `[REJECT]` — Deleting the penalty simplified the model without losing performance. _Removed._
- **Fully symmetric joint potential** `[REJECT]` — Lost real direction-specific information. _Rejected._
- **Geometric-mean two-sided gates** `[REJECT]` — Unstable across directions. _Rejected._
- **Symmetric linear expert mixture** `[REJECT]` — Did not resolve directional degradation. _Rejected._
- **Log-partition expert aggregation** `[REJECT]` — No stable gain. _Rejected._
- **Entropy-weighted KL barycenter** `[REJECT]` — No stable gain. _Rejected._
- **Molecular joint-potential compensation** `[REJECT]` — Insufficient. _Rejected._
- **Expert-disagreement variance fallback** `[REJECT]` — R2E degraded. _Rejected._
- **Normalized conditional expert mixture** `[REJECT]` — R2E degraded strongly. _Rejected._
- **Dimension-scaled consistency** `[REJECT]` — No advantage over deleting consistency. _Rejected._
- **Directional linear conditional expectation** `[LOCAL]` — Most stable FIBRE conditional mode. _Historical local winner._
- **Log-Mean-Exp aggregation** `[REJECT]` — Development acceptable, frozen evaluation failed. _Rejected._
- **Second-order variance correction** `[REJECT]` — Failed development R2E. _Rejected._
- **Reaction-center mode posterior** `[REJECT]` — Development improved, frozen R2E degraded. _Rejected._

### Explicit scientific evidence

Move structure, context and mechanism outside latent geometry and require evidence admission.

- **Anchored scientific-evidence layer** `[TURN]` — This formulation was substantially more stable. _Major surviving FIBRE idea._
- **Structure scientific evidence** `[KEEP]` — Passed evidence admission. _Survived as structural expert semantics._
- **Known-positive context evidence** `[KEEP]` — Stable conditional capability. _Survived._
- **Reaction-center scientific evidence** `[LOCAL]` — Informative pair evidence but ranking admission was weaker. _Retained mostly as evidence._
- **Evidence admission and calibration** `[TURN]` — Tightened expert permission semantics. _Survived into BRIDGE gating._

### Adaptive expert mixtures

Query-conditioned mixtures worked safely only when the strong core was protected.

- **Heterogeneous Conditional Modes** `[TURN]` — Unified execution worked; universal FIBRE core still did not dominate. _Modularity survived._
- **Query-adaptive expert mixture** `[TURN]` — The safe version required freezing the core. _Led toward explicit applicability._
- **End-to-end bidirectional gate** `[REJECT]` — Damaged R2E. _Rejected._
- **E2R-only gate** `[REJECT]` — Shared training still hurt R2E. _Rejected._
- **Frozen-core post-hoc gate** `[TURN]` — Preserved R2E exactly; E2R gain was insufficient for promotion. _Validated the safe gating pattern._

### Relational-core replacement

A modern relational core was implemented and tested as a possible universal ranker.

- **ERAM relational core** `[TURN]` — Implemented, trained and served. _Still failed to displace Broad._
- **UniMol feature preparation** `[HISTORICAL]` — Implemented. _Historical._
- **Relational main model** `[REJECT]` — Strict double-cold/temporal evaluation did not justify replacing Broad. _Rejected as universal core._
- **Strict temporal + double-cold relational evaluation** `[KEEP]` — Supported the decision to keep Broad as base. _Retained as evidence._

### Pluggable evidence

Modularity and open-world fallback survived even though the unified core did not.

- **Pluggable expert adapters** `[TURN]` — Implemented successfully. _Directly survived into BRIDGE modularity._
- **Frozen context plugin** `[TURN]` — Validated plug-in execution. _Survived._
- **Open-world fallback** `[KEEP]` — Preserved broad fallback. _Survived as missing-neutral behavior._

## 06 · BRIDGE makes ranking authority explicit: Broad orders globally; experts earn bounded local correction rights.

**Problem.** The question after FIBRE was no longer how to create a more unified representation. It was how to allocate ranking authority without losing broad coverage or useful specialist evidence.

**What we learned.** Query-conditioned routing, permission levels and family-specific validation showed that the same evidence can be valuable locally and harmful globally. Missing or inapplicable evidence must remain neutral.

**Decision.** BRIDGE keeps Broad as the universal base order and treats functional, structural, mechanistic, context and domain specialists as gated pair evidence with bounded influence.

### Reframe experts around a protected base

The final design principles that turned historical expert experiments into BRIDGE semantics.

- **Experts as pair evidence** `[KEEP]` — Created a clean correction interface. _Retained._
- **Rebind expert evidence to Broad Core** `[KEEP]` — Broad became the explicit base again. _Retained._
- **Calibrate Broad before evidence fusion** `[KEEP]` — Reduced arbitrary evidence-scale effects. _Retained._
- **Directional score evidence** `[KEEP]` — EnzGFM evidence showed direction-specific admission behavior. _Retained._
- **Expert-type hierarchy** `[KEEP]` — Made responsibilities explicit. _Retained._
- **Query-conditioned expert routing** `[TURN]` — Established query-conditioned expert weighting. _Refined into explicit applicability._
- **Expert permission levels** `[KEEP]` — Produced a safer permission model. _Direct BRIDGE ancestor._

### Validate specialists only in their own applicability domain

Generic structural authority failed; family and TPS specialists returned under explicit gates.

- **CAGE family-response branch** `[TURN]` — Family-tuned CAGE improved several domains despite generic CAGE failure. _Became gated family specialists._
- **P450 CAGE specialist** `[KEEP]` — Clear local improvement. _Retained under family gate._
- **Phosphatase CAGE specialist** `[KEEP]` — Clear local improvement. _Retained under family gate._
- **Terpene CAGE specialist** `[KEEP]` — Local improvement. _Retained under family gate._
- **TPS specialist correction** `[KEEP]` — Activated sparsely on TPS-applicable queries. _Retained as a gated specialist._
- **Integrated gated domain specialists** `[KEEP]` — Improved local domains without reducing broad coverage. _Immediate predecessor of BRIDGE._
- **Layered EnzymeCAGE full-suite comparison** `[KEEP]` — Showed candidate coverage and expert scoreability are distinct constraints. _Supports the final narrative._

## BRIDGE today

Current composition, separated from the historical route that produced it.

- **Base:** Broad Retrieval — Universal candidate generator and stable global ranker. It remains valid when every optional expert is silent.
- **Control:** Applicability + permission layer — For each query and direction, decide whether an expert is relevant, supported and allowed to modify rank; unavailable experts contribute exactly zero.
- **Functional / evolutionary evidence:** EnzGFM with reaction features. General learned functional/evolutionary evidence, admitted only where it adds clean ranking information.
- **Structural evidence:** CLIPZyme, cached pocket / structure support. Structure is evidence with explicit support boundaries, not a universal replacement ranker.
- **Mechanistic evidence:** bounded reaction-center correction, TPS specialist. Mechanistic signals can adjust a local shortlist only when their applicability tests pass.
- **Context evidence:** known-positive seed context, multi-seed context. Context is optional query evidence. When no seed is supplied, the channel is simply silent.
- **Family-specific CAGE specialists:** P450 fine-tuned EnzymeCAGE, phosphatase fine-tuned EnzymeCAGE, terpene fine-tuned EnzymeCAGE. Activation uses candidate-independent generic→family EnzymeCAGE reaction-response features, a reaction-only applicability selector, and reaction-family / pair-level family agreement. The checkpoints remain specialists, never general CAGE ranking authority.
- **Correction:** Bounded correction interface — Experts act as pair evidence with direction-specific rights. Candidate scope, shortlist depth and correction magnitude constrain how much the base order can move.

Formula:

`S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)`

## Parallel project tracks

### Real user requests turned retrieval into COMPASS.

Users changed scope, constraints and follow-up tasks conversationally. That pushed the project from semantic routing to a bounded agent and persistent research workspace.

- **User-driven semantic scope switching** — Semantic intent switching and context continuation were added before the formal agent.
- **Bounded scientific agent** — Created the first end-to-end scientific agent harness with verified recovery paths.
- **Persistent research workspace** — Made scientific objects reusable across conversations and downstream analyses.
- **Starase Navigator** — Became the mature pre-COMPASS application identity.
- **COMPASS** — Current user-facing scientific agent identity.

### Retrieval results had to become an executable wet-lab campaign.

Ranked candidates were translated into auditable panels, balanced plate assignment, randomized wells and a feedback contract for later learning.

- **Retrieval-to-wet-lab decision branch** — Candidate source, exploitation, uncertainty and diversity became explicit experimental roles.
- **Discovery panel construction** — Produced auditable per-reaction candidate panels rather than blindly taking the first K scores.
- **MILP inter-plate balancing** — The solver reached optimal plate assignments and sharply reduced between-plate covariate ranges.
- **Hungarian well-position randomization** — Balanced within-block well placement after plate assignment.
- **Wet-lab feedback contract** — Separated usable experimental feedback from ambiguous assay failures.

## Source of truth

The full experiment inventory remains in `scripts/engineering_lineage/lineage_data.py`.
This generated presentation intentionally changes the **semantic role** of records without deleting the underlying history.
