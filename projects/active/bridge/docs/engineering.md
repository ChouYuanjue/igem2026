# BRIDGE Engineering Story

The public Engineering page reveals one primary DBTL loop at a time. Each loop may contain recursively nested subloops; raw experiment records are evidence at the leaves rather than the primary navigation.

## 01 · The first failure was upstream of ranking.

EnzymeCAGE could judge pairs inside a bounded candidate system, but a positive excluded by the gate could never be recovered downstream.

### Can better ranking rescue a bounded candidate system?

**Outcome:** Candidate eligibility becomes the dominant bottleneck.
**Why next:** Replace fixed gate membership with retrieval from molecular inputs.

- **Design:** Audit structural ranking inside the existing candidate system.
- **Build:** Run full-library structural scoring and candidate rescue.
- **Test:** Measure ranking and candidate-gate coverage.
- **Learn:** A missed positive never reaches the upper ranker.

#### Structure ranking audit

**Outcome:** Structure remained useful evidence, but not a complete full-library ranking solution.
**Why next:** Try using reaction neighbourhoods to rescue candidate generation.

- **Design:** Check whether pocket choice explains ranking weakness.
- **Build:** Run the structural scorer across the full TPS library.
- **Test:** Compare pocket robustness with full-library ranking.
- **Learn:** Pair compatibility does not become useful library ranking by itself.

#### Candidate-pool rescue

**Outcome:** Candidate rescue improved ranks locally but preserved a hard recall ceiling.
**Why next:** Remove gate membership from scoreability entirely.

- **Design:** Use reaction similarity to transfer candidates from nearby known reactions.
- **Build:** Construct the rescued closed candidate pool.
- **Test:** Measure how many known positives survive the gate.
- **Learn:** The gate still excludes positives before any upper ranker can see them.

## 02 · Broad entered below CAGE before it became the base order.

The initial goal was conservative: let Broad open the candidate universe while CAGE remained the ideal upper reranker. That architecture exposed a second ceiling in the upper layer itself.

### Can Broad handle recall while CAGE keeps ranking authority?

**Outcome:** Ranking responsibility begins moving into Broad itself.
**Why next:** Keep Broad's order and make extra evidence conditional.

- **Design:** Replace the hard gate while keeping CAGE above it.
- **Build:** Train open Broad retrieval and feed its candidates upward.
- **Test:** Compare Broad candidate reach with generic CAGE scoreability.
- **Learn:** Broad reaches farther than the generic CAGE upper layer can support.

#### Build a useful Broad order

**Outcome:** Broad is more than a recall-only candidate generator.
**Why next:** Audit whether CAGE can still remain the upper authority.

- **Design:** Score arbitrary reactions and proteins from molecular inputs.
- **Build:** Train representations and hard-negative-aware retrieval.
- **Test:** Stress the ranking under cold and shifted regimes.
- **Learn:** Broad develops standalone ordering value.

##### Open molecular representation

**Outcome:** The retrieval space is open rather than relation-gated.
**Why next:** Improve the quality of the open-space order.

- **Design:** Represent proteins and reactions directly from molecular inputs.
- **Build:** Train a multi-positive bidirectional dual tower.
- **Test:** Check double-unseen retrieval rather than relation-table membership.
- **Learn:** Any new reaction or protein can now receive a score.

##### Make the open order useful

**Outcome:** Broad develops a stable full-space ordering signal.
**Why next:** Test whether CAGE can still own the upper-layer decision.

- **Design:** Reduce false negatives and mine informative hard negatives.
- **Build:** Add hard-negative curricula, domain adaptation and ranking objectives.
- **Test:** Stress the order across cold and domain-shift regimes.
- **Learn:** Broad is no longer only a recall mechanism; its ranking has standalone value.

#### CAGE as the ideal upper reranker

**Outcome:** Broad can reach much farther than a generic CAGE upper layer can reliably score.
**Why next:** Let Broad keep more ranking responsibility and treat extra evidence conditionally.

- **Design:** Keep CAGE above Broad and let it make the final structural judgement.
- **Build:** Feed Broad candidates into generic CAGE where structural inputs are available.
- **Test:** Compare candidate reach with upper-layer scoreability on the layered suite.
- **Learn:** The upper layer has its own coverage and support ceiling.

#### Retention guardrails

**Outcome:** Broad protection becomes a guardrail, not a separate system identity.
**Why next:** Externalize complementary capabilities when appropriate.

- **Design:** Add new domains without erasing the incumbent Broad behaviour.
- **Build:** Try replay, blending, regularization and score retention.
- **Test:** Compare external and temporal retrieval after adaptation.
- **Learn:** Some capabilities are safer to route around Broad than to force into one parameter state.

## 03 · Broad became the base order; other capabilities became experts.

Once Broad itself had a meaningful order, the engineering problem changed from replacing the ranker to deciding when additional evidence deserves to modify that order.

### How should heterogeneous evidence modify a strong Broad order?

**Outcome:** BiME organizes experts around Broad, but global admission remains too coarse.
**Why next:** Ask whether one relational object can explain the expert stack.

- **Design:** Externalize complementary capabilities around Broad.
- **Build:** Develop functional, structural and learned fusion experts.
- **Test:** Confirm expert value by direction, support and context.
- **Learn:** Expert usefulness is conditional rather than universal.

#### Functional / evolutionary evidence

**Outcome:** Functional evidence is complementary and conditional.
**Why next:** Admit it as an optional expert.

- **Design:** Test whether learned enzyme-function representations add complementary evidence.
- **Build:** Add RDKit and reaction features around EnzGFM.
- **Test:** Evaluate on frozen external retrieval tasks.
- **Learn:** The signal helps as an expert but does not replace Broad globally.

##### EnzGFM feature adaptation

**Outcome:** The adapted feature view is retained as an expert input.
**Why next:** Pass the surviving signal to expert admission.

- **Design:** Start from the native EnzGFM signal.
- **Build:** Add RDKit descriptors and reaction features.
- **Test:** Compare the feature variants under the same frozen task.
- **Learn:** Reaction-aware features are the useful retained variant.

#### Structure / mechanism evidence

**Outcome:** Structure and mechanism become local expert evidence.
**Why next:** Combine experts only through protected fusion and routing.

- **Design:** Search for structural and mechanistic signals Broad does not encode directly.
- **Build:** Develop reaction-center and CLIPZyme branches.
- **Test:** Audit support, transfer and local ranking gains.
- **Learn:** These signals are valuable only inside explicit applicability regions.

##### Reaction-center correction

**Outcome:** Reaction-center evidence survives only as a bounded local correction.
**Why next:** Treat structural/mechanistic evidence as applicability-limited expertise.

- **Design:** Inject local reaction-center mechanism into ranking.
- **Build:** Try direct fusion, identity-preserving residual and bounded correction.
- **Test:** Check whether local gains survive without damaging the base order.
- **Learn:** The useful form is bounded and local, not globally authoritative.

##### CLIPZyme structural branch

**Outcome:** CLIPZyme is usable only with explicit support semantics.
**Why next:** Route structural evidence rather than treating it as universal.

- **Design:** Test a learned structural expert alongside Broad.
- **Build:** Audit encoder substitution, support coverage and fallback behaviour.
- **Test:** Measure gains only where author-supported inputs exist.
- **Learn:** Structural expertise must expose support and fallback explicitly.

#### Fusion and routing

**Outcome:** BiME is the first stable expert system over Broad.
**Why next:** Ask whether the experts can be unified under one relational object.

- **Design:** Move from fixed score addition to learned, protected expert fusion.
- **Build:** Develop R2E, anchored E2R and formal expert admission.
- **Test:** Compare expert value across direction, support and context.
- **Learn:** The system needs explicit expert routing around a Broad incumbent.

##### R2E learned fusion

**Outcome:** R2E confirms that expert value can be learned directionally.
**Why next:** Build an anchored E2R counterpart and formal admission.

- **Design:** Learn a ranker over a selected expert portfolio.
- **Build:** Union expert candidates and train R2E LambdaRank.
- **Test:** Compare learned fusion against fixed score rules.
- **Learn:** Learned fusion helps, but routing must respect direction and support.

##### E2R anchored rescue

**Outcome:** E2R establishes the protected-incumbent principle.
**Why next:** Formalize which experts are allowed into the system.

- **Design:** Extend expert fusion to E2R without losing the strong base.
- **Build:** Try four-expert fusion, then baseline-anchored rescue and learning-to-rank.
- **Test:** Compare free fusion against the anchored form.
- **Learn:** Anchoring to Broad is essential when the base is already strong.

##### Formal expert admission

**Outcome:** BiME can organize multiple experts, but expert authority remains too coarse.
**Why next:** Search for a cleaner relational formulation and, later, query-level authority.

- **Design:** Require every extra capability to justify entry.
- **Build:** Evaluate structural, context, homology, reciprocal and CAGE experts.
- **Test:** Use frozen direction-specific expert confirmation.
- **Learn:** Expert usefulness is heterogeneous; admission cannot be one global yes/no forever.

###### CLIPZyme admission

**Outcome:** CLIPZyme is admitted conditionally.
**Why next:** Apply the same discipline to other context experts.

- **Design:** Treat CLIPZyme as a candidate structural expert.
- **Build:** Expose structural support and fallback semantics.
- **Test:** Confirm gains only where structure is supported.
- **Learn:** Structural expertise is admissible only with scoped support.

###### Seed-context admission

**Outcome:** Seed context becomes a conditional expert.
**Why next:** Keep missing context neutral.

- **Design:** Use known-positive context when it exists.
- **Build:** Add seed and multi-seed context features.
- **Test:** Evaluate only queries carrying valid context.
- **Learn:** Context is useful but intrinsically optional.

###### Generic CAGE admission

**Outcome:** The generic CAGE expert is rejected.
**Why next:** Later revisit CAGE at family scope rather than globally.

- **Design:** Test whether CAGE can return as a generic expert above Broad.
- **Build:** Route generic CAGE through the same admission framework.
- **Test:** Run frozen OOF expert confirmation.
- **Learn:** Generic CAGE does not earn universal expert authority.

## 04 · FIBRE tested a cleaner abstraction without replacing the main line.

BiME made the stack work, but left a scientific question: what common relation are all these experts estimating? FIBRE explored that question as a side branch, then returned useful principles to the Broad-centered design.

### Can one relational core replace the expert stack and Broad order?

**Outcome:** The replacement fails, while several interfaces survive.
**Why next:** Return to Broad and allocate expert authority locally.

- **Design:** Model enzyme–reaction matching as one relational problem.
- **Build:** Explore conditional modes, explicit evidence and adaptive relational cores.
- **Test:** Use strict temporal and double-cold replacement tests.
- **Learn:** Keep evidence admission, plugins and fallback; restore Broad authority.

#### Relational formulation

**Outcome:** The relation abstraction is useful, but the ranking core still needs conditional evidence.
**Why next:** Explore conditional expert modes.

- **Design:** Describe enzyme–reaction matching as biological relation rather than only rank score.
- **Build:** Try tensor/product geometry and catalytic kernels.
- **Test:** Check whether the geometry yields a stable conditional ranking object.
- **Learn:** Geometry helps organize the problem but does not solve ranking alone.

##### Conditional-mode search

**Outcome:** The useful mathematics is directional and conditional, not universally symmetric.
**Why next:** Move unsupported science into explicit evidence channels.

- **Design:** Find a common mathematical form for heterogeneous expert value.
- **Build:** Explore symmetric, probabilistic and directional conditional forms.
- **Test:** Evaluate candidate forms under frozen directional ranking.
- **Learn:** A directional conditional view survives; many elegant symmetric forms do not.

###### Symmetric formulations

**Outcome:** Symmetric formulations are rejected.
**Why next:** Test probabilistic and directional alternatives.

- **Design:** Force both retrieval directions into a shared symmetric potential.
- **Build:** Try consistency, geometric gates, symmetric linear mixtures and scaling fixes.
- **Test:** Compare frozen directional retrieval under the symmetric forms.
- **Learn:** Forced symmetry erases useful directional differences.

###### Probabilistic aggregations

**Outcome:** Probabilistic aggregations fail to become the main conditional rule.
**Why next:** Prefer the simpler directional conditional expectation.

- **Design:** Aggregate experts with probabilistic or entropy-based rules.
- **Build:** Try variance fallback, normalized mixtures and second-order forms.
- **Test:** Compare against the simpler directional baseline.
- **Learn:** Extra aggregation complexity does not justify itself.

###### Directional conditional expectation

**Outcome:** Directional conditional expectation is retained.
**Why next:** Anchor the relational core in explicit scientific evidence.

- **Design:** Allow expert value to differ by retrieval direction.
- **Build:** Use directional linear conditional expectation and test local compensations.
- **Test:** Compare against symmetric and higher-order alternatives.
- **Learn:** Directional conditioning is the stable local survivor.

#### Scientific evidence admission

**Outcome:** Evidence admission and missing-neutral semantics survive into BRIDGE.
**Why next:** Use those interfaces around, rather than instead of, a global base order.

- **Design:** Represent physical or biochemical cues as explicit evidence.
- **Build:** Separate structure, known context and reaction-center evidence.
- **Test:** Calibrate whether each evidence source is actually supported.
- **Learn:** Missing or unsupported evidence should contribute neutrally.

##### Structure evidence

**Outcome:** Structure stays optional and support-aware.
**Why next:** Keep unsupported structure neutral.

- **Design:** Expose structure as an explicit scientific channel.
- **Build:** Materialize available structural evidence.
- **Test:** Check support coverage before use.
- **Learn:** No structure means no structural correction.

##### Known-positive context

**Outcome:** Context remains a conditional evidence source.
**Why next:** Route it only when available.

- **Design:** Use known-positive context when present.
- **Build:** Encode context as its own evidence channel.
- **Test:** Evaluate only where context exists.
- **Learn:** Context cannot be assumed globally.

##### Reaction-center evidence

**Outcome:** Reaction-center evidence stays local.
**Why next:** Feed it through evidence admission.

- **Design:** Expose reaction-center information explicitly.
- **Build:** Attach center evidence to pair scoring.
- **Test:** Validate center support before correction.
- **Learn:** Mechanistic evidence is local and optional.

#### Adaptive relational core

**Outcome:** The replacement hypothesis fails, while interfaces survive.
**Why next:** Use these principles to allocate local authority over Broad.

- **Design:** Turn FIBRE principles into an adaptive relational main model.
- **Build:** Combine query-adaptive routing, ERAM and pluggable adapters.
- **Test:** Challenge the replacement under strict temporal and double-cold tests.
- **Learn:** Keep routing, plugins and fallback; restore Broad as global order.

##### Query-adaptive mixture

**Outcome:** Query-adaptive routing works best around a frozen core.
**Why next:** Test a modern relational main model without sacrificing fallback.

- **Design:** Let expert weights depend on the current query.
- **Build:** Try end-to-end, E2R-only and frozen-core post-hoc gates.
- **Test:** Compare which routing scheme preserves the base while adding local gains.
- **Learn:** Frozen-core post-hoc gating is the stable form.

##### ERAM relational core

**Outcome:** ERAM does not displace Broad under strict replacement tests.
**Why next:** Keep plugins and open-world fallback around the incumbent instead.

- **Design:** Test an ERAM-style relational core as the new main model.
- **Build:** Prepare UniMol features and train the relational main model.
- **Test:** Use strict temporal and double-cold evaluation.
- **Learn:** The relational main model cannot justify replacing Broad.

##### Pluggable expert adapters

**Outcome:** The plugin/fallback interface survives the FIBRE detour.
**Why next:** Return to Broad as ordering authority.

- **Design:** Keep new evidence sources detachable from the core.
- **Build:** Implement frozen-context plugin and open-world fallback.
- **Test:** Verify the system remains valid when optional evidence is absent.
- **Learn:** Plugins plus neutral fallback are more robust than replacing the global core.

## 05 · BRIDGE decides who may change Broad's order, where, and by how much.

The final design keeps Broad globally valid, gives specialists bounded ranking rights, and separates training-time recall from genuinely new runtime context before either is allowed to influence the query.

### Who may alter Broad's order for this query?

**Outcome:** BRIDGE is a global Broad order plus query-gated local authority and provenance-aware dual memory.
**Why next:** The current architecture closes after ranking authority and context authority are separated explicitly.

- **Design:** Rebind every expert to Broad and separate context that the model already saw from context that arrived after training.
- **Build:** Add query-conditioned expert permission, provenance-aware dual memory and domain specialists.
- **Test:** Validate specialists in their applicability domains and learn both memory authorities only on firewalled validation data.
- **Learn:** Broad stays global; specialists get bounded local rights, while remembered facts and new support receive distinct query-specific authority.

#### Permission model

**Outcome:** Permission becomes the control plane over optional evidence.
**Why next:** Attach domain specialists behind this control plane.

- **Design:** Reframe expert outputs as evidence relative to Broad.
- **Build:** Add directional score evidence, expert types and routing.
- **Test:** Separate availability, usefulness and query applicability.
- **Learn:** Expert authority is query- and direction-specific.

##### Rebind evidence to Broad

**Outcome:** All expert evidence is defined relative to a valid Broad base.
**Why next:** Decide when an expert is allowed to act.

- **Design:** Treat expert outputs as optional pair evidence.
- **Build:** Calibrate Broad before evidence fusion.
- **Test:** Check that missing evidence leaves the Broad score valid.
- **Learn:** Broad must remain the valid default score everywhere.

##### Query-conditioned routing

**Outcome:** Query applicability becomes the permission layer.
**Why next:** Specialize experts inside their own applicability domains.

- **Design:** Separate expert availability from expert usefulness.
- **Build:** Add expert types and explicit permission levels.
- **Test:** Evaluate whether the current query should activate each expert.
- **Learn:** Authority must be decided per query and direction.

#### How should known positives influence a model that may already have seen them?

**Outcome:** BRIDGE separates remembered training facts from genuinely new runtime evidence.
**Why next:** Feed both memories into the final bounded authority model without changing Broad candidate coverage.

- **Design:** Split context by provenance: training-time relations are recall; post-training relations are new evidence.
- **Build:** Pair a validation-gated long-term reciprocal memory with a separate validation-gated episodic support update.
- **Test:** Use validation-only authority learning, enforce no double counting, and keep frozen outer few-shot evaluation separate from zero-shot BRIDGE metrics.
- **Learn:** Context authority depends on when the relation became known and on the current query; fixed α and fixed rank windows have no universal meaning.

##### Training-graph long-term memory

**Outcome:** Training-time relations become long-term recall, not fresh evidence.
**Why next:** Handle relations that appeared only after training as a different memory type.

- **Design:** Separate relations already present during training from genuinely new runtime evidence.
- **Build:** Keep clean2023 exact enzyme–reaction neighborhoods as read-only reciprocal recall outside the parameter model.
- **Test:** Learn query-conditioned recall authority only on the firewalled validation split and verify that training positives never vote twice as runtime seeds.
- **Learn:** A relation already absorbed by training should be recalled once, with query-specific authority rather than a fixed global weight or rank band.

##### Post-training episodic memory

**Outcome:** Post-training positives become query-local episodic support.
**Why next:** Combine both memory types with the same provenance-aware BRIDGE authority model.

- **Design:** Treat database additions and user-confirmed positives that were absent from training as runtime support.
- **Build:** Pool support in the frozen Broad space and make one temporary query update; registered and open-world supports share the same support-only interface.
- **Test:** Freeze a 0–1 trust gate on validation episodes, then check frozen few-shot generalization and support-only open-world entity handling.
- **Learn:** New context can adapt the current query without entering the candidate pool or receiving one universal seed weight.

#### Family-specific CAGE

**Outcome:** CAGE completes its role transition: upper ranker → rejected generic expert → family specialist.
**Why next:** Let query permission activate the right family expert.

- **Design:** Revisit CAGE at family scope after generic admission failed.
- **Build:** Train P450, phosphatase and terpene specialists.
- **Test:** Evaluate each specialist only in its applicability domain.
- **Learn:** CAGE regains authority only as a family-specific specialist.

##### P450 CAGE specialist

**Outcome:** P450 earns local structural authority.
**Why next:** Keep the same rule for other responsive families.

- **Design:** Specialize CAGE for P450 queries.
- **Build:** Fine-tune the P450 structural expert.
- **Test:** Evaluate only inside the P450 applicability domain.
- **Learn:** Family-local CAGE improves where generic CAGE did not.

##### Phosphatase CAGE specialist

**Outcome:** Phosphatase earns local structural authority.
**Why next:** Retain family-scoped activation.

- **Design:** Specialize CAGE for phosphatase queries.
- **Build:** Fine-tune the phosphatase structural expert.
- **Test:** Evaluate only inside the phosphatase domain.
- **Learn:** The family-specific expert gives a strong local gain.

##### Terpene CAGE specialist

**Outcome:** Terpene CAGE remains family-scoped.
**Why next:** Merge only through query permission.

- **Design:** Specialize CAGE for terpene queries.
- **Build:** Fine-tune the terpene structural expert.
- **Test:** Evaluate only inside the terpene family.
- **Learn:** Even the weaker family gain is safer as a local specialist than a global expert.

#### TPS specialist

**Outcome:** TPS returns as a local mechanistic expert.
**Why next:** Combine it with other admitted specialists through the same bounded interface.

- **Design:** Reuse TPS-specific mechanism after the task became broad.
- **Build:** Gate a bounded TPS correction.
- **Test:** Activate only on matched TPS queries.
- **Learn:** TPS is now one biochemical specialist rather than the whole task.

#### Bounded integration

**Outcome:** The final interface is a Broad base plus query-gated bounded corrections.
**Why next:** This closes the engineering loop in BRIDGE.

- **Design:** Combine only evidence that has passed query-level permission.
- **Build:** Apply bounded corrections while leaving Broad valid outside the local scope.
- **Test:** Run the layered full-suite comparison.
- **Learn:** Broad stays global; specialists contribute bounded local gains.

## BRIDGE today

`S_BRIDGE(q,e) = S_Broad(q,e) + Σ_k g_k(q) Δ_k(q,e)`
