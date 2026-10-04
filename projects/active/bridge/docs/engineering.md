# BRIDGE Engineering Atlas

The Engineering history is organized as five macro stages. FIBRE is a detour between BiME and the dominant final BRIDGE-formation stage.

## A · From bounded structure ranking to open retrieval

Two compact loops establish the problem and the Broad retrieval base. The early work stays intentionally small in the atlas.

### Can structure ranking scale to a real library?

**Learn:** Hard candidate gates cap recall

- **Design:** audit pocket robustness
- **Build:** full-library scoring + transfer
- **Test:** rank + gate coverage
- **Learn:** missed candidates stay lost

### Can unseen reactions and proteins enter ranking directly?

**Learn:** Broad becomes the open-world base order

- **Design:** molecular-input scoring
- **Build:** registry + dual tower
- **Test:** double-cold stress tests
- **Learn:** protect Broad order

### Full record

- **EnzymeCAGE** `[ROOT]` — It worked inside a bounded support/candidate regime but exposed coverage, scoreability and open-world limits. _Root of the whole Engineering lineage._
- **Pocket robustness audit** `[TURN]` — On a 50-reaction slice, first-pocket and official-pocket Top-5/Top-10 were both 54%/68%; multi-pocket aggregation did not improve them, while 52.46% of pairs had their highest structure score outside pocket 1. _Established that richer local pocket evidence exists, but naive aggregation is not the missing ranking principle._
- **Full-library structural pair screen** `[TURN]` — For 10 TPS reactions against 1,391 proteins, 13,790 structural pair scores still yielded zero Top-1/5/10 hits, MRR about 0.0037 and median best-positive rank 387. _Separated pairwise compatibility scoring from large-candidate retrieval._
- **Reaction-similarity transfer** `[TURN]` — The 10-query screen improved MRR to about 0.0622 and median best-positive rank to 16.5, proving that candidate-space structure matters. _Created the first successful retrieval gate, and exposed its eventual recall ceiling._
- **Closed candidate-pool system** `[TURN]` — Strong for in-database completion; every downstream step depended on the positive entering the pool. _Made candidate coverage the first major problem definition._
- **High-recall candidate union** `[TURN]` — Efficient, but excluded positives became unrecoverable. _Forced open candidate-space thinking._
- **CAGE pool-internal structural ranking** `[LOCAL]` — Important early signal; limited by support coverage and raw-score behavior. _Returned later as family-specific structural expertise._
- **Conditional CAGE rescue** `[LOCAL]` — Useful as a bounded rescue rather than a universal score. _Early availability-aware structure pattern._
- **Direct raw CAGE probability fusion** `[REJECT]` — Sigmoid saturation and ties manufactured unstable fine order. _Rejected; structure had to be handled more cautiously._
- **Tree-based score fusion** `[REJECT]` — Useful on the old fixed-pool task, brittle outside it. _Did not become the broad core._
- **Main ranking + rescue slots** `[TURN]` — Practical and interpretable. _Early ancestor of anchored / bounded correction._
- **Candidate-gate recall ceiling** `[TURN]` — Only 720 of 1,640 known enzyme records entered the pool: about 43.98% relation coverage. Any missed positive became unrecoverable downstream. _Turned open-world coverage from an implementation detail into the next scientific problem._
- **Open-world retrieval problem** `[TURN]` — The task changed from candidate filtering to bidirectional open retrieval. _Spawned candidate-universe, representation, specialization, uncertainty and workflow branches._
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
- **Broad Retrieval** `[KEEP]` — Provided stable open-world ranking and a fallback that does not require optional experts. _Became the universal base order for BRIDGE._
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
- **Evidence, applicability & uncertainty** `[KEEP]` — Created diagnostics that stayed orthogonal to ranking. _Several concepts later helped expert gating._
- **Grouped reliability calibration** `[KEEP]` — Produced protocol-bound reliability tiers. _Retained as a risk layer._
- **Selective abstention** `[KEEP]` — Useful application behavior. _Retained outside core ranking._
- **Candidate Evidence Passport** `[LOCAL]` — Made decisions auditable. _Retained as provenance/interpretation._
- **Open-world applicability proxy** `[TURN]` — Separated applicability from activity probability. _Conceptually anticipated expert applicability._
- **Conformal Retrieval Sets** `[LOCAL]` — All global calibration combinations passed finite-sample tolerance; sets were often large. _Retained as a diagnostic/coverage contract._
- **Applicability-aware Mondrian conformal** `[LOCAL]` — Enabled only where calibration/test support was sufficient. _Retained with fallback to global calibration._
- **Bidirectional cycle consistency** `[LOCAL]` — Useful explanation signal. _Kept as evidence, not ranking authority._
- **Cycle rerank weight grid** `[REJECT]` — Confirmation produced no added hits or MRR gain. _Rejected from production ranking._
- **Graph & non-parametric alternatives** `[TURN]` — Some restricted routes worked, global graph methods were unstable. _Contributed local auxiliary methods._
- **Graph diffusion** `[REJECT]` — Hub amplification and weak double-cold entry. _Rejected._
- **Multi-hop propagation** `[REJECT]` — Amplified noise and hub effects. _Rejected._
- **Candidate hub normalization** `[REJECT]` — Multiple normalizers failed to produce stable gains. _Rejected._
- **Protein + reaction dual kernel** `[TURN]` — Current-only version failed freezing; budget-limited MARTS E2R succeeded. _Kept only under narrow conditions._
- **Current-only dual kernel** `[REJECT]` — Development gain did not freeze. _Rejected._
- **MARTS dual-kernel rescue** `[LOCAL]` — Passed repeated confirmation. _Retained as a local route._

## B · Stabilize Broad before adding more capability

Broad is now the incumbent. The next loop asks whether it can expand to new domains without losing the ranking behavior that made open retrieval work.

### Can Broad adapt without forgetting?

**Learn:** A strong base should be protected, not repeatedly rewritten

- **Design:** extend without forgetting
- **Build:** replay + retention methods
- **Test:** external + temporal tests
- **Learn:** route around the incumbent

### Full record

- **Broad Retrieval** `[KEEP]` — Provided stable open-world ranking and a fallback that does not require optional experts. _Became the universal base order for BRIDGE._
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
- **External & temporal stress tests** `[KEEP]` — Several routes failed only when moved to external/fresh support. _Kept as selection discipline._
- **Leakage-safe nested cleanroom selection** `[KEEP]` — Became mandatory selection discipline. _Retained._
- **Pure EnzymeCAGE same-support baseline** `[KEEP]` — Established auditable apples-to-apples comparison. _Retained as baseline evidence._
- **Enzyme-405 same-support comparison** `[KEEP]` — Provided fair external comparison. _Retained as benchmark evidence._
- **Orphan-335 retrieval stress test** `[KEEP]` — Provided an external broad-retrieval challenge. _Retained as stress evidence._
- **TIGER reaction-novel baseline** `[LOCAL]` — Useful for fair comparison. _Benchmark-only._
- **Fresh temporal transfer test** `[REJECT]` — Transfer failures exposed support/alignment limits. _Used to constrain claims, not as a production route._
- **TPS MARTS R2E symmetry confirmation** `[KEEP]` — Passed its dedicated confirmation. _Retained as capability evidence._
- **Strict temporal benchmark** `[KEEP]` — Became part of final model evaluation discipline. _Retained._

## C · Turn heterogeneous evidence into explicit experts

Functional, structural and ranking loops run in parallel. Their shared result is BiME-Rank: Broad stays protected while additional capabilities are admitted as experts.

### Can functional models add broad evidence?

**Learn:** Functional evidence helps, but remains optional

- **Design:** add functional evidence
- **Build:** EnzGFM + reaction features
- **Test:** frozen retrieval tests
- **Learn:** expert, not base ranker

### Can structure and mechanism improve ranking safely?

**Learn:** Structure and mechanism help only inside supported regions

- **Design:** add structural evidence
- **Build:** bounded local corrections
- **Test:** support + fresh transfer
- **Learn:** gate by applicability

### Can multiple experts improve ranking without breaking Broad?

**Learn:** BiME stabilizes a portfolio, but global admission is still too coarse

- **Design:** protect Broad + add experts
- **Build:** union + anchored LambdaRank
- **Test:** frozen expert admission
- **Learn:** query-specific usefulness

### Full record

- **Broad Retrieval** `[KEEP]` — Provided stable open-world ranking and a fallback that does not require optional experts. _Became the universal base order for BRIDGE._
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
- **Rank fusion & expert routing** `[TURN]` — Fixed fusion repeatedly failed; learned/anchored routing worked better. _Direct parent of BiME-Rank._
- **Raw score addition** `[REJECT]` — Score scales were incompatible. _Rejected._
- **Tied-rank percentile fusion** `[LOCAL]` — Useful in selected routes. _Kept locally._
- **Reciprocal Rank Fusion** `[LOCAL]` — Stable for selected E2R Top-10 routes. _Retained locally._
- **Dual-neural E2R rank fusion** `[LOCAL]` — Survived repeated confirmation for Top-10. _Retained as a budget-specific route._
- **Fixed three-source fusion** `[TURN]` — Development improved; frozen performance fell. _Proved complementarity does not imply global fixed weights._
- **Early TPS LambdaRank candidate stacking** `[REJECT]` — Too little/unstable data in the early TPS setting. _Rejected then later revisited on broad clean data._
- **Retrieval expert portfolio selection** `[TURN]` — Produced a manageable expert set. _Led to R2E LambdaRank._
- **Expert candidate union** `[KEEP]` — Improved support without requiring one universal retriever. _Retained in BiME._
- **R2E LambdaRank stack** `[KEEP]` — Passed frozen confirmation. _Main direct parent of BiME-Rank._
- **R2E similarity router** `[KEEP]` — Confirmed deterministic routing. _Retained._
- **Joint reliability gate** `[LOCAL]` — Improved safety constraints. _Fed expert admission logic._
- **BiME-Rank** `[KEEP]` — Strong multi-expert predecessor, but expert value was still mostly admitted globally/task-wise. _Direct predecessor of BRIDGE._
- **E2R expert-fusion branch** `[TURN]` — Unrestricted fusion damaged a very strong EnzGFM base. _Forced explicit base protection._
- **Four-expert E2R LambdaRank** `[REJECT]` — Rejected after head-ranking degradation. _Rejected._
- **Baseline-anchored E2R rescue** `[TURN]` — Safer than unrestricted fusion. _Led to Anchored LambdaMART._
- **Anchored E2R learning-to-rank** `[KEEP]` — Passed confirmation. _Direct ancestor of BRIDGE base-order protection._
- **Formal expert admission** `[KEEP]` — Made expert inclusion auditable. _Retained conceptually._
- **CLIPZyme structural expert** `[KEEP]` — Passed admission. _Retained._
- **Seed context expert** `[KEEP]` — Conditionally admitted. _Retained._
- **Multi-seed context** `[KEEP]` — Validated in BiME. _Retained as conditional context._
- **Homology context expert** `[REJECT]` — Failed external retention. _Rejected._
- **Reciprocal consistency expert** `[REJECT]` — Failed external retention. _Rejected._
- **Generic EnzymeCAGE expert** `[TURN]` — Internal OOF ranking degraded. _Critical evidence that structural expertise is conditional._
- **Cost-aware hierarchical execution** `[KEEP]` — Cheap experts search broadly; expensive experts run on shortlists. _Retained in final execution philosophy._
- **Strong-baseline absorption policy** `[KEEP]` — Formalized conservative integration. _Direct BRIDGE precursor._

## D · FIBRE detour

A concentrated attempt to replace the expert stack with one relational core. The detour contributes useful principles, then returns to Broad.

### Relational formulation

**Learn:** Useful geometry emerges, but forced symmetry and many aggregations fail

- **D:** formalize relations
- **B:** geometry + conditionals
- **T:** compare formulations
- **L:** retain directional evidence

### Scientific evidence

**Learn:** Structure, context and reaction-center signals work better as admitted evidence

- **D:** anchor explicit evidence
- **B:** structure + context + center
- **T:** calibrate evidence support
- **L:** missing evidence stays neutral

### Can the relational core replace Broad?

**Learn:** No: retain plugins and fallback, abandon global replacement

- **D:** query-adaptive relations
- **B:** post-hoc gate + ERAM + plugins
- **T:** strict replacement test
- **L:** restore Broad authority

### Full record

- **FIBRE unified-model detour** `[HISTORICAL]` — Produced a coherent research line but never replaced Broad as the universal ordering core. _Archived; several design conclusions survived into BRIDGE._
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
- **Anchored scientific-evidence layer** `[TURN]` — This formulation was substantially more stable. _Major surviving FIBRE idea._
- **Structure scientific evidence** `[KEEP]` — Passed evidence admission. _Survived as structural expert semantics._
- **Known-positive context evidence** `[KEEP]` — Stable conditional capability. _Survived._
- **Reaction-center scientific evidence** `[LOCAL]` — Informative pair evidence but ranking admission was weaker. _Retained mostly as evidence._
- **Evidence admission and calibration** `[TURN]` — Tightened expert permission semantics. _Survived into BRIDGE gating._
- **Heterogeneous Conditional Modes** `[TURN]` — Unified execution worked; universal FIBRE core still did not dominate. _Modularity survived._
- **Query-adaptive expert mixture** `[TURN]` — The safe version required freezing the core. _Led toward explicit applicability._
- **End-to-end bidirectional gate** `[REJECT]` — Damaged R2E. _Rejected._
- **E2R-only gate** `[REJECT]` — Shared training still hurt R2E. _Rejected._
- **Frozen-core post-hoc gate** `[TURN]` — Preserved R2E exactly; E2R gain was insufficient for promotion. _Validated the safe gating pattern._
- **ERAM relational core** `[TURN]` — Implemented, trained and served. _Still failed to displace Broad._
- **UniMol feature preparation** `[HISTORICAL]` — Implemented. _Historical._
- **Relational main model** `[REJECT]` — Strict double-cold/temporal evaluation did not justify replacing Broad. _Rejected as universal core._
- **Strict temporal + double-cold relational evaluation** `[KEEP]` — Supported the decision to keep Broad as base. _Retained as evidence._
- **Pluggable expert adapters** `[TURN]` — Implemented successfully. _Directly survived into BRIDGE modularity._
- **Frozen context plugin** `[TURN]` — Validated plug-in execution. _Survived._
- **Open-world fallback** `[KEEP]` — Preserved broad fallback. _Survived as missing-neutral behavior._

## E · Build BRIDGE from locally valid expert authority

This is the main final stage. Query permission, family CAGE, TPS specialization and bounded integration develop as sibling loops, then converge into BRIDGE.

### Query applicability and permission

**Learn:** Expert authority becomes query- and direction-specific

- **D:** rebind experts to Broad
- **B:** directional router + expert types
- **T:** availability ≠ usefulness
- **L:** only applicable experts act

### Family-specific CAGE

**Learn:** Generic CAGE fails globally; family CAGE works locally

- **D:** specialize CAGE by family
- **B:** fine-tune family specialists
- **T:** family-scoped evaluation
- **L:** activate local family expert

### TPS specialist

**Learn:** Mechanistic TPS evidence stays sparse and local

- **D:** reuse TPS mechanism evidence
- **B:** gate TPS correction
- **T:** matched TPS queries only
- **L:** silent outside TPS niche

### Integrated bounded correction

**Learn:** All specialists merge through one bounded correction interface

- **D:** combine admitted evidence
- **B:** preserve Broad outside shortlist
- **T:** full-suite comparison
- **L:** global Broad + local gains

### Full record

- **Return to Broad as ordering core** `[TURN]` — The architecture was reframed around ranking authority rather than one representation. _Starts the final BRIDGE line._
- **Experts as pair evidence** `[KEEP]` — Created a clean correction interface. _Retained._
- **Rebind expert evidence to Broad Core** `[KEEP]` — Broad became the explicit base again. _Retained._
- **Calibrate Broad before evidence fusion** `[KEEP]` — Reduced arbitrary evidence-scale effects. _Retained._
- **Directional score evidence** `[KEEP]` — EnzGFM evidence showed direction-specific admission behavior. _Retained._
- **Expert-type hierarchy** `[KEEP]` — Made responsibilities explicit. _Retained._
- **Query-conditioned expert routing** `[TURN]` — Established query-conditioned expert weighting. _Refined into explicit applicability._
- **Expert permission levels** `[KEEP]` — Produced a safer permission model. _Direct BRIDGE ancestor._
- **Query-level expert applicability** `[KEEP]` — Explained generic-expert failures and specialist successes consistently. _Core BRIDGE principle._
- **CAGE family-response branch** `[TURN]` — Family-tuned CAGE improved several domains despite generic CAGE failure. _Became gated family specialists._
- **P450 CAGE specialist** `[KEEP]` — Clear local improvement. _Retained under family gate._
- **Phosphatase CAGE specialist** `[KEEP]` — Clear local improvement. _Retained under family gate._
- **Terpene CAGE specialist** `[KEEP]` — Local improvement. _Retained under family gate._
- **TPS specialist correction** `[KEEP]` — Activated sparsely on TPS-applicable queries. _Retained as a gated specialist._
- **Integrated gated domain specialists** `[KEEP]` — Improved local domains without reducing broad coverage. _Immediate predecessor of BRIDGE._
- **Layered EnzymeCAGE full-suite comparison** `[KEEP]` — Showed candidate coverage and expert scoreability are distinct constraints. _Supports the final narrative._
- **BRIDGE** `[KEEP]` — Combines broad coverage, expert modularity, missing-neutral semantics, specialist locality and protected base ranking. _Current method._

## Parallel tracks

### Wet-lab execution

- **Retrieval-to-wet-lab decision branch** — Candidate source, exploitation, uncertainty and diversity became explicit experimental roles.
- **Discovery panel construction** — Produced auditable per-reaction candidate panels rather than blindly taking the first K scores.
- **MILP inter-plate balancing** — The solver reached optimal plate assignments and sharply reduced between-plate covariate ranges.
- **Hungarian well-position randomization** — Balanced within-block well placement after plate assignment.
- **Wet-lab feedback contract** — Separated usable experimental feedback from ambiguous assay failures.

### COMPASS workflow

- **User-driven semantic scope switching** — Semantic intent switching and context continuation were added before the formal agent.
- **Bounded scientific agent** — Created the first end-to-end scientific agent harness with verified recovery paths.
- **Persistent research workspace** — Made scientific objects reusable across conversations and downstream analyses.
- **Starase Navigator** — Became the mature pre-COMPASS application identity.
- **COMPASS** — Current user-facing scientific agent identity.

## BRIDGE today

`S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)`
