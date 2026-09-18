# Product–Research Observation Workflow

## One method, three operating contexts

The method is not a frozen database and it is not a collection of fast/full models.

A deployment is one realization of the same construction:

molecular observations -> (M_R,g_R),(M_E,g_E) -> Omega -> F_Omega(r,e).

The scarce object is the verified enzyme–reaction correspondence Omega.
Sequence, molecular structure, protein structure, pocket geometry, catalytic local
contexts and literature are observations of the two factor spaces. Missing an
observation means that coordinate is unobserved; it is never negative evidence.

The same method is used in three contexts:

1. **research evaluation** — test whether the construction generalizes without
   revealing held-out pair labels;
2. **reproducible domain expansion** — rebuild the factor manifolds and pair
   registry for another enzyme/reaction domain;
3. **product inference** — serve low-latency queries against a versioned reference
   atlas and the latest verified production correspondence set.

These contexts differ in what data are visible and when expensive measurements
are acquired. They do not define different scoring models.

## 1. Separate molecular observations from pair labels

There are two fundamentally different data planes.

### 1.1 Label-blind molecular plane

Protein examples include primary sequence, global sequence embedding,
family-applicable catalytic local sequence context, existing structure,
whole-structure 3Di, detected pockets, pocket-local sequence state,
pocket-local 3Di, and pocket-distribution OT geometry.

Reaction examples include canonical reactant/product structure,
reaction-difference representation, reactant/product molecular neighbourhoods,
atom mapping, reaction-centre transitions, and curated mechanistic context.

These observations may be built before looking at enzyme–reaction pair labels.

### 1.2 Sparse correspondence plane

A pair (r,e) enters Omega only as a verified positive biochemical association
with provenance. No synthetic negative pair is required by the correspondence field.

Pair acquisition is expensive supervision: curated databases, papers,
collaborator records and wet-lab confirmation. Adding one verified positive must
therefore be useful without global retraining.

For the current field, a new positive performs an exact pointwise-min update of
the joint and marginal distance transforms. It is mathematically identical to
rebuilding the enlarged correspondence field from scratch.

## 2. Research evaluation contract

Research evaluation must never use held-out pair labels to construct training
Omega, choose geometry, choose acquisition depth, or tune ranking.

Two regimes should be reported explicitly.

### 2.1 Pair-cold / label-blind transductive geometry

All declared label-blind molecular observations may define the factor manifolds,
while pair labels are split by protein and reaction clusters.

This asks: given a molecular universe whose entities are observable but whose
enzyme–reaction associations are sparse, can known pairs recover unseen
correspondences?

This is the current canonical MARTS development setting.

### 2.2 Strict inductive atlas transfer

For each held-out protein/reaction fold, those entities are removed while the
reference atlas is built. A held-out entity is then attached using only its own
query-to-reference measurements.

This asks: can a genuinely new molecule be attached to a previously built atlas
without rebuilding the whole manifold or seeing its labels?

Strict inductive transfer is an audit, not a replacement for the canonical
pair-cold result. Both must report the exact data visibility contract.

## 3. Production contract

Production is not a benchmark. It should not pretend that already verified
biochemistry is unknown.

A production snapshot contains a versioned immutable reaction atlas, a versioned
immutable protein atlas, the best currently verified positive set Omega_prod,
provenance for every positive and molecular measurement, and cached measurements
for registered entities.

All verified production positives may participate in the production
correspondence field. Evaluation results remain tied to frozen Omega_train and
cannot be recomputed under Omega_prod and reported as benchmark numbers.

New wet-lab positives may update Omega_prod immediately without model retraining.
A verified positive whose protein lies outside the focused reference atlas may
still contribute immediately when its sequence is already available: attach that
protein out of sample and apply the same exact positive-set update. This avoids
turning an atlas boundary into an artificial evidence boundary. New molecular
entities are promoted into a new reference-atlas version only by an offline
rebuild.

## 4. Reference atlas versus online query extension

The expensive operation is **atlas construction**, not ordinary prediction.

### Offline reference build

For an entity collection:

1. resolve stable reaction/protein identities;
2. acquire all scientifically valid molecular observations allowed by the build;
3. construct each measurement map independently;
4. build the partially observed factor geometry;
5. freeze entity order, local scales, graph topology and provenance;
6. publish an immutable atlas version.

The current low-latency deployment bundle is
data/terpene_correspondence_deployment_atlas_v2.
It stores factor graph edge lengths, per-measurement availability and local
scales. It is label-free.

### Online external query

Never rebuild the reference N x N graph for one request.

For a new point q:

1. acquire permitted query-side observations;
2. calculate only query-to-reference distances in observed coordinates;
3. estimate a query-local intrinsic scale;
4. attach q to the frozen atlas at the intrinsic sqrt(N) chart scale;
5. run one single-source geodesic calculation;
6. evaluate the same correspondence field against Omega_prod.

The generic implementation is
projects/active/fibre/geometry/extension.py.

Reference-reference geometry remains unchanged. More query information can
refine the attachment without creating another retrieval model.

## 5. AI-selected observation filtration: latency without model fragmentation

Use a nested observation filtration O0 subset O1 subset O2 subset O3. A deeper
level means more observations of the same molecular point, not a different
predictor. **Observation depth is selected by the semantic planner, not by a
frontend mode switch.**

The semantic planner receives the user's natural-language goal plus verified
reaction/protein context. It returns an intent-level `analysis_depth`:

- `standard` — ordinary interactive candidate discovery where cheap molecular
  observations are sufficient;
- `deep` — the task materially benefits from structural, pocket, mechanistic or
  unusually careful evidence acquisition.

These are execution-plan fields, not concepts the user must know or select.

### Cached observations override cost

If a measurement is already materialised for a reference/registered entity, use
it immediately regardless of the selected depth. A zero-latency structural
coordinate should not be discarded merely to imitate a lower-information run.

### Standard interactive acquisition

Protein: primary sequence and global ESM-C, plus directly available local
sequence context.

Reaction: canonical reaction structure, DRFP, and reactant/product neighbourhood.

Do not synchronously predict a missing 3D structure.

### Deep evidence acquisition

When semantic intent justifies it, additionally attempt reusable structure lookup,
whole-structure 3Di, pocket detection, pocket-local sequence, pocket 3Di / pocket
OT, atom mapping, reaction-centre mechanism evidence, and curated/literature
evidence.

Deep work remains progressive. Return the first valid ranking before genuinely
long stages finish; a later run revision may refine query attachment if new factor
geometry actually becomes available. Reaction-centre evidence is currently
**evidence only** in canonical ranking because development ablation showed a
significant E2R regression when it was forced into g_R.

### Planning is not execution

The runtime must distinguish all of the following states:

- `reuse_cached` — materialised before the request and available now;
- `provided` — supplied as part of the request;
- `planned_now` — selected by the semantic acquisition plan but not yet executed;
- `computed_now` — actually materialised successfully for this query;
- `execution_failed` — selected but not successfully acquired;
- `defer` — useful additional evidence that is not part of the current execution;
- `blocked_by_dependency` — cannot be acquired until another observation exists.

A planned observation must never be rendered as though it already affected the
ranking.

### Candidate-side cached evidence

Observation acquisition is not only a query-side concern. For an
application-domain reference candidate, molecular measurements that already exist
in the frozen reference collection are reused at zero query-time acquisition cost.
The product therefore reports aggregate Top-K coverage of global sequence state,
family/motif context, whole-structure 3Di, pocket-local sequence state, pocket 3Di
and pocket OT when those views are present. Missing candidate views remain neutral.

This coverage report is role-aware: protein structural/pocket views that belong to
the canonical protein factor are reported as ranking geometry; reaction-centre
transition observations remain mechanism evidence unless a future development
audit justifies promoting them. The frontend reports coverage in scientific terms
(e.g. structure or pocket evidence available for N/K candidates), not internal
atlas coordinates.

### Reproduce / domain-build

Reproducible domain rebuilding is a research/batch operation, not a chat mode. It
may include de-novo structure prediction, batch pocket detection, all structure
comparisons, atom mapping for the full reaction registry, rebuilding factor
atlases, and rebuilding cold splits/evaluation artifacts. It is never exposed as
a synchronous frontend choice.

## 6. AI-native product UI contract

Natural language is the primary control surface. The user should describe the
scientific goal; the product should not ask them to choose a model, atlas,
candidate-universe implementation, route identifier, or observation-depth mode.

The semantic route planner decides two scientific properties after entity
verification:

1. **retrieval scope** — `broad` versus `application_domain`;
2. **analysis depth** — `standard` versus `deep`.

For a verified terpene-synthase / terpenoid-catalysis target,
`application_domain` maps internally to the strongest currently validated focused
retrieval realization. For broader or insufficiently supported contexts it maps
to the broad retrieval realization. The mapping is an implementation detail and
may evolve without changing the user interaction.

The semantic LLM owns intent interpretation. Deterministic code that runs before
or after it may parse exact scientific identifiers and validate execution safety,
but it must not infer user intent from keyword gates. In particular:

- exact RHEA / UniProt ID parsing is identity resolution, not semantic routing;
- Reaction SMILES / FASTA detection is structured-input parsing, not semantic
  routing;
- enum validation, seed identity checks, candidate masks and provenance checks are
  post-semantic guardrails;
- literal word lists must not decide whether the user wants factual lookup versus
  predictive candidate discovery.

The normal frontend therefore contains no Standard/Deep selector and no route or
model catalog. After execution it may explain:

- the AI-selected scientific search scope in human terms;
- why that scope was chosen;
- which molecular evidence was cached, computed, planned, unavailable or deferred;
- whether verified positives were used as context;
- the actual high-level search/evidence steps used for this request.

Internal backend names and route IDs remain available only in machine provenance
and developer audit artifacts. Public ranking HTTP endpoints also ignore legacy
`route_mode` / `observation_mode` override fields: every user-facing ranking
request enters the intelligent semantic planner. Deterministic default routing
remains available only to internal workers whose task semantics were already
established by the agent or research workflow.

## 7. Product run lifecycle

The public candidate-discovery path is LLM-first at the task level and
verification-first at the scientific-identity level:

1. `interpret_user_goal` — the agent LLM decides whether the request is factual
   database lookup, exploratory candidate discovery, route design, pathway analysis
   or another supported scientific task;
2. `resolve_and_verify_entity` — deterministic/database tools establish reaction,
   protein, sequence or structure identity;
3. `semantic_retrieval_plan` — using the original request **and verified target
   context**, the route LLM chooses broad/application-domain scope and
   standard/deep evidence depth;
4. `validate_plan` — deterministic code validates enums, seed identities, masks,
   taxonomy and other execution constraints without re-interpreting intent;
5. `inspect_cached_observations`;
6. `acquire_selected_query_observations`;
7. `attach_query_to_reference_atlas` when the query is not already a reference
   state;
8. `evaluate_correspondence_or_broad_field`;
9. `return_ranked_hypotheses`;
10. `assemble_biological_witnesses`;
11. optionally continue genuinely long deep observations and publish an auditable
    refined run revision.

The frontend shows only the actual plan chosen for the current request. It does
not expose a menu of repository routes or pipelines.

A run records the semantic retrieval scope, semantic analysis depth, internal
backend provenance, atlas version, Omega snapshot/version, exact observations
actually used, query hash, candidate-set hash, and code/builder provenance. A
refined revision preserves its predecessor so ranking changes are auditable.

## 8. Reproducible application-domain expansion

A third party must not need our frozen MARTS matrices to reproduce the method.

The transferable recipe is:

1. **Entity registry** — collect stable protein sequences and reaction structures.
   Aliases are provenance, not separate molecular states.
2. **Label-blind measurement acquisition** — compute scientifically meaningful
   available views; every builder writes availability and provenance.
3. **Positive-pair registry** — curate verified enzyme–reaction positives with
   source/evidence level. Do not manufacture negatives merely to fit the method.
4. **Cold split construction** — cluster proteins and reactions independently;
   freeze development/test cells before comparing method variants.
5. **Factor-manifold build** — build M_R and M_E from molecular observations only.
6. **Correspondence field** — construct F_Omega from training positives.
7. **Evaluation** — report bidirectional R2E/E2R under one field; optionally run
   strict out-of-sample atlas-transfer audit.
8. **Deployment packaging** — publish local scales, graph edge lengths, entity
   order, observation availability and provenance.
9. **Production evolution** — add verified positives to Omega_prod immediately;
   accumulate new entities/measurements and periodically publish a new atlas.

The expensive parts are measurement acquisition and pair curation. The
mathematical training/update rule itself remains portable.

## 9. What “use all available information” means

It does **not** mean forcing every measurement into a ranking score.

Every valid observation gets one of three roles:

1. **factor geometry** — coherently measures molecular state and has a
   non-destructive geometric role;
2. **biological/mechanistic evidence** — useful for interpreting a hypothesis but
   not currently justified as a ranking metric;
3. **provenance/annotation** — context that should not pretend to be a distance or
   label.

Promoting an observation from evidence to factor geometry requires both a
mathematically natural construction and non-destructive development evidence.
This is why protein pocket/structure observations enter current multiresolution
g_E, whereas reaction-centre observations remain mechanistic evidence for now.

## 10. Current implementation status

Implemented:

- FIBRE correspondence field and bidirectional evaluator;
- exact no-retrain positive-seed update;
- multiresolution protein geometry and global reaction chemistry geometry;
- strict inductive atlas-transfer audit with zero train/test protein and reaction
  overlap in all nine double-cold cells;
- self-contained `data/terpene_correspondence_deployment_atlas_v2/` containing
  production query-extension sufficient statistics;
- exact online registered-state lookup and out-of-sample reaction/protein
  attachment through one `CorrespondenceGeometryService`;
- canonical reaction encoding parity and online ESM-C/reference parity checks;
- production Omega updates including externally supplied verified protein seeds;
- truthful observation planning/execution states;
- AI-native semantic routing from natural-language goal + verified target context;
- internal broad/application-domain backend mapping with no architecture-facing
  frontend selector;
- conversation-first frontend with no Standard/Deep selector and no route catalog;
- actual-run search-plan explanation and executed-observation provenance;
- candidate-side cached structure/pocket/chemistry coverage for focused Top-K results;
- source-fingerprinted runtime provenance and explicit frontend asset cache versioning;
- static AI-native contracts plus focused, broad and deep regression/live-smoke suites.

Strict inductive audit authority:
`results/terpene_product_correspondence_strict_inductive_v1/summary.json`. Current
summary metrics are E2R MRR 0.06236 / median rank 38 and R2E MRR 0.06060 / median
rank 126; all declared train/test entity overlaps are zero. This audit does not
replace the canonical label-blind transductive-side-information development
result.

Remaining engineering work:

- asynchronous executors for genuinely missing external-protein deep structural
  observations (structure -> pocket -> 3Di / OT);
- ongoing monitoring of live semantic routing behavior as the production LLM evolves;
- optional refined-run push/progress transport for long deep acquisitions.

Until the first item is complete, the observation plan remains intentionally
honest: missing deep structural views can be planned/deferred, but are not marked
`computed_now` and do not silently alter online ranking.

