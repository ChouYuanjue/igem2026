# Catalyst Finder

A fully isolated conversational interface for the deployed terpene retrieval system. It does not import, build, mount, or modify `frontend/terpene_portal`.

## Product flow

The main surface is one conversation rather than separate parameter forms. A user can describe either direction in natural language:

- **Reaction → enzyme**: describe substrate/product, optional positive enzymes, taxonomy preference, shortlist size, and whether close homolog families should be allowed.
- **Enzyme → reaction**: describe a protein/enzyme in natural language or provide a UniProt/local ID, then ask for likely reactions, optional known-activity expansion, or known-activity masking.
- **Whole pathway compatibility**: describe a multi-step reaction chain in natural language (or simply type a chain such as `A → B → C`). Each reaction is still verified against Rhea. If a step has no enzyme specified, the existing R2E production ranker supplies candidates and a global compatibility layer jointly chooses the enzyme set.

The interface follows a human-in-the-loop agent pattern:

1. Parse the user's intent and biological entities.
2. Resolve reaction descriptions against real Rhea records.
3. Resolve enzyme/protein descriptions against the deployed 2,085-protein model catalog and UniProt REST results.
4. Pause in the conversation for the user to confirm the exact Rhea / protein records.
5. Send only confirmed IDs to constrained LangGraph route planners.
6. Run the existing production `RetrievalEngine`.
7. Render the actual route as a module chain, not just a route ID, next to the ranked results.
8. Keep a lightweight explicit “continue from previous run” context chip for follow-up requests such as “排除近缘，再给 Top 20”.

The LLM is not a source of truth for Rhea IDs or protein accessions. It normalizes names and proposes intent-level route choices; deterministic database lookup and LangGraph guardrails control what reaches the production runtime.

## Natural-language protein resolution

`/api/resolve-protein` and `/api/agent/resolve` support descriptions such as:

```text
丹参中的 miltiradiene synthase KSL1
```

The DeepSeek layer normalizes the description to search terms, then the server searches the deployed model catalog and UniProt. The user confirms the selected result before it is used.

For R2E, a confirmed positive-enzyme description can become a Few-shot seed even when the user never typed its accession literally. If the confirmed UniProt protein is outside the deployed 2,085-protein universe, the isolated server can create a temporary verified seed CSV and use the repository's existing temporary-candidate extension path; arbitrary user model paths are never exposed.

For E2R, a model-catalog protein is ranked by ID. A verified external UniProt entry is fetched for its real amino-acid sequence and submitted through the existing external-protein E2R path.

## Routing behavior

Default R2E remains fixed to Top-10, all enzyme candidates, Zero-shot, homolog-allowed retrieval.

Intelligent R2E can interpret:

- Top-3 / 5 / 10 / 20
- all / eukaryote / prokaryote candidate scope
- confirmed user positive enzymes or explicitly requested catalog-known positives
- homolog-allowed vs remote-family cross-cluster discovery

Remote-family discovery uses the repository's MMseqs2 50% sequence-identity cluster boundary with >=80% coverage. It is a transparent post-score novelty overlay, not a fabricated production route suffix.

Intelligent E2R can interpret:

- Top-3 / 5 / 10 / 20
- ordinary Zero-shot annotation
- explicitly requested expansion from catalog-known reactions (`fewshot`)
- explicitly requested masking of catalog-known reactions for novel-activity discovery (`masked`)

## Run

```bash
cd /home/s241850073/igem2026
.venv/bin/pip install -r scripts/catalyst_finder/requirements.txt
bash scripts/catalyst_finder/manage.sh start
```

Default port: `8791`.

Runtime state and secrets live under the git-ignored directory:

```text
results/catalyst_finder_runtime/
```

Configure DeepSeek without placing the key in shell history:

```bash
bash scripts/catalyst_finder/manage.sh configure-key
```

## Main endpoints

```text
GET  /api/status
GET  /api/routes
POST /api/agent/resolve
POST /api/resolve
POST /api/resolve-protein
POST /api/rank
POST /api/rank-reactions
POST /api/route/design
POST /api/pathway/analyze
POST /api/feedback
```

## Tests

```bash
.venv/bin/python -m unittest \
  scripts.catalyst_finder.test_serve \
  scripts.catalyst_finder.test_routing_graph \
  scripts.catalyst_finder.test_route_view \
  scripts.catalyst_finder.test_agent_flows -v
```

The original portal bridge regression suite should remain green:

```bash
.venv/bin/python -m pytest \
  projects/active/terpene_screening/tests/test_terpene_portal_bridge.py -q
```


## Product entry points and progressive capability disclosure

The public interface is organized around the **scientific question**, not around internal model routes or parameter panels. The first screen intentionally keeps only four primary tasks:

1. reaction → enzyme;
2. enzyme → reaction;
3. biosynthetic route design;
4. whole-pathway enzyme compatibility.

A compact capability ribbon makes the broader scope visible without forcing a new user to configure it: database evidence, unrecorded associations, known-activity expansion, remote-family search, route feasibility, and pathway compatibility. An opt-in “More common tasks and constraints” guide then exposes **23 natural-language templates** grouped by the same four scientific tasks. The templates cover result scope, Top 3/5/10/20, eukaryote/prokaryote filtering, known-enzyme references, remote-family filtering, external UniProt proteins, route priorities, predicted-route exploration, one-pot/sequential/in-vivo pathways, and explicit pH/temperature/cofactor conditions.

These templates are **examples, not new modes**. Clicking one only inserts a natural-language request into the composer; it never auto-submits and never creates hidden stateful priority/pathway selectors. The semantic router and existing guardrails remain the source of truth. This progressive-disclosure structure is deliberate: the four main cards preserve a low-friction first use, while the expandable guide lets experienced users discover the system's real breadth.

The Chinese interface uses product-language terms rather than untranslated internal jargon. In particular:

- `已知证据` = a pair supported by the integrated database evidence sources;
- `新关联候选` = a model-ranked association absent from the currently integrated evidence sources;
- `模型探索` = model/rule-based exploration as a process description.

The English word `discovery` must not appear as a Chinese-interface label, summary, status, or result description. Scientific proper names and identifiers such as Rhea, UniProt, Swiss-Prot, E. coli, MDF, FBA and pH remain unchanged.

## Known evidence and discovery policy

Catalyst Finder is evidence-first. Database-recorded reaction–enzyme associations are presented as the primary factual result, independent of whether the current neural candidate universe covers that entity. Neural retrieval is a separate discovery layer for associations that are not recorded in the integrated evidence sources.

- no special request → `allow_known`: show recorded database evidence first, then return Top-K **unrecorded discovery candidates**; recorded items do not consume discovery slots;
- “只看已记录 / known only” → `known_only`: show database-recorded evidence only; the neural discovery table is omitted;
- “排除已知 / discovery only” → `exclude_known`: rank only unrecorded discovery candidates while keeping recorded evidence available as a clearly separated reference section.

The frontend does not expose a stateful result-scope selector. It offers lightweight text actions that insert the corresponding natural-language request into the composer; they never execute the task or mutate route settings directly. Known evidence and discovery candidates are never mixed in one ranking table.

“Recorded” means the pair is supported by the project association catalog or the official Rhea/Swiss-Prot mapping. The primary evidence card shows the source record directly. When the current neural model also has a retrieval score for that recorded entity, the score appears as secondary metadata; model-coverage explanations are kept out of the main reading path. Unrecorded candidate tables label the value as a retrieval score and use it for relative ranking within that candidate set.

The same evidence/discovery split is symmetric for reaction→enzyme and enzyme→reaction queries. The official Rhea/Swiss-Prot mapping is indexed in both directions.

## Intent boundary: one reaction vs route design

Catalyst Finder treats endpoint wording as a semantic contract before LLM entity parsing:

- **`底物 → 产物` / `把 A 转化为 B` / `目标反应`** means one reaction and routes to `reaction_to_enzyme`;
- **`起始前体 → 目标产物`** means route endpoints and routes to `route_design` even if the user omits the word “route”;
- **two or more explicit arrows (`A → B → C`)** mean the user already supplied a fixed multi-step pathway and route to `pathway_compatibility`;
- generic `A 到 B` without reaction-role or route-generation wording is deliberately not promoted to route design by a regex; the ordinary intent parser handles it.

The route-design and pathway starter cards carry internal task hints, while the visible control remains “自动判断”; no new mode selector is exposed. These internal hints are soft: if the user rewrites a starter into an obviously different task (for example, changes the route starter to “把 GPP 转化为 beta-myrcene，找候选酶”), the explicit text contract overrides the stale starter hint. The visible reaction/enzyme expert selectors remain hard user choices.

## Candidate route design and ranking

Route design is another **natural-language task**, not a new parameter panel. Requests such as “推荐从 GPP 到 beta-myrcene 的几条路线并排序”, “优先少几步”, or “优先容易找到酶的路线” are normalized into a guarded `route_design` intent. The language model may normalize source/target/host names and ranking intent, but it never invents intermediate reactions, Rhea IDs or ChEBI IDs. The confirmation card only verifies the source/target entities.

The production route designer has two strictly separated evidence layers:

1. **Known-biochemistry layer (`rhea_full_graph_v1`)** — official Rhea release TSVs are cached under `results/catalyst_finder_runtime/cache/route_design/rhea/`. Directed reaction SMILES, ChEBI structures, direction metadata and Swiss-Prot mappings are converted into a broad biochemical graph. Currency cofactors are excluded from main-chain shortcuts, and RDKit structure continuity is used only to choose the likely main substrate/product connection inside a Rhea hyper-reaction. Every returned step keeps its Rhea ID for audit. Candidate simple paths are generated with NetworkX `shortest_simple_paths`.
2. **Explicit prediction layer (`MINE/Pickaxe`)** — a pinned upstream MINE-Database snapshot lives under `external_repos/route_design/MINE-Database/`; its exact commit is recorded in `MINE-Database.UPSTREAM_COMMIT`. It is never imported into the web-service process. `pickaxe_worker.py` runs it in a subprocess with worker-only dependencies under `results/catalyst_finder_runtime/route_design/pickaxe_site/`. It performs one bounded generation with the bundled MetaCyc generalized rules. A predicted product is useful only when it can be mapped back to a Rhea/ChEBI compound and then connected to the target through the known Rhea graph. Predictions that duplicate an existing direct Rhea edge are not reported as novel. Predicted bridge routes are displayed in a separate exploratory list and never mixed into the known-route ranking.

Known-route ranking is interpretable and relative. Candidate generation still starts with path length, Swiss-Prot enzyme availability, main-transformation structural continuity, direction evidence and current project-model coverage, but **final Top-K is no longer cut off at that stage**. A larger preliminary pool is passed through the feasibility layer first. Natural language can emphasize `short`, `enzyme_available`, `project_covered`, `thermodynamic` (MDF / ΔG) or `host_flux` (FBA / host-product-flux); there is still no frontend priority selector. General words such as “可实现性” remain `balanced` unless the user explicitly names a priority.

### Real thermodynamic layer

The isolated thermodynamic runtime uses `equilibrator-api==0.7.0` and `equilibrator-pathway==0.7.1` under `results/catalyst_finder_runtime/route_feasibility/thermo_site/`. Before calculation, the main-chain route projection is discarded and the service reconstructs the **complete directed Rhea hyper-reaction** from the official `rhea-reaction-smiles.tsv`; every participant must map exactly back to an official Rhea/ChEBI structure. The worker then:

- resolves those ChEBI identifiers through eQuilibrator;
- requires the `PhasedReaction` to be balanced;
- reports per-step standard transformed ΔG′° and physiological ΔG′ with uncertainty where available;
- computes whole-route Max-min Driving Force (MDF) using `equilibrator-pathway`.

The deployed eQuilibrator cache currently uses its observed default aqueous conditions: **pH 7.5, pMg 3.0, ionic strength 0.25 M, 298.15 K (25 °C)** and eQuilibrator's default metabolite-concentration bounds (typically 1 µM–10 mM with curated cofactor exceptions). These conditions are returned with the API result and rendered in the frontend. A positive MDF means the package found a concentration assignment within those bounds with positive minimum driving force; it is **not** a prediction of enzyme activity or pathway yield. Missing/unresolved/unbalanced reactions remain `unknown`.

`equilibrator-pathway 0.7.1` currently has a result-object edge case for single-reaction MDF: optimization succeeds but `PathwayMdfSolution` indexes a scalar physiological ΔG as a vector. `route_thermo_worker.py` does not downgrade the package; only for that specific single-reaction `IndexError`, it calls the package's own `_conc_constraints` and `_thermo_constraints` with the same CLARABEL MDF objective and reads the solved objective directly. Multi-step pathways use the public `mdf_analysis()` result normally.

### E. coli host-feasibility layer

The isolated host runtime uses `COBRApy==0.32.1` and a real cached BiGG **iML1515** model (2712 reactions, 1877 metabolites, 1516 genes) under `results/catalyst_finder_runtime/route_feasibility/`. The E. coli route-start pool is generated from **cytosolic ChEBI annotations in that actual model** rather than the old Pickaxe example CSV whenever the FBA runtime is ready.

For every candidate all-Rhea route, `route_fba_worker.py` maps complete Rhea/ChEBI stoichiometry into the iML1515 cytosol. A non-native required participant is created as an internal metabolite **without an exchange**, so it cannot appear from nowhere. The worker then adds the candidate reactions and a target demand, introduces one common route-flux variable `z`, and requires **every candidate step plus target output to carry at least `z`**. It maximizes `z` while preserving at least 10% and 50% of baseline wild-type growth. This prevents a native bypass from making an unused candidate route look feasible.

A route is hard-filtered only when FBA completed successfully and its route-supported flux is zero. Missing mapping / missing worker / solver failure remains unknown and is not labelled infeasible. The displayed FBA capacity is a **stoichiometric flux capacity, not a predicted titer, enzyme kinetic rate or fermentation yield**.

Final ranking keeps the original base route score and adds candidate-set-relative MDF and, for E. coli, route-supported FBA components. Completed zero-flux E. coli routes are removed before final Top-K; completed negative-MDF routes are strongly demoted but kept auditable. Missing evidence never receives a feasibility bonus. `score` / `final_score` remains only a relative prioritization number.

A selected all-Rhea route can be inserted back into the composer with “填入这条路线继续评估酶兼容性”. This action only fills natural language; it does not auto-submit. The next task reuses the existing `pathway_compatibility-v1` workflow. Predicted Pickaxe steps do not get this handoff because they first require independent reaction validation.

To recreate the optional route runtimes without modifying the main `.venv` dependency set:

```bash
# rule-based predicted bridge exploration
scripts/catalyst_finder/setup_route_explorer.sh

# eQuilibrator cache + COBRApy + BiGG iML1515 (~1.4 GB thermo cache)
scripts/catalyst_finder/setup_route_feasibility.sh
```

Heavy route packages are imported only in subprocess workers. The pinned MINE source, eQuilibrator cache, COBRApy model/runtime and Rhea cache are independent from `frontend/terpene_portal` and `scripts/terpene_portal`. They are not automatically upgraded at service startup.

## Whole-pathway enzyme compatibility

Pathway compatibility is intentionally **not** another parameter panel. In automatic direction mode, natural-language pathway intent or a chain containing at least two arrows is resolved as `pathway_compatibility`. The confirmation card only asks the user to verify each Rhea step and any enzyme they explicitly named.

After confirmation, `POST /api/pathway/analyze`:

1. reuses the deployed Reaction→Enzyme production ranking for every unspecified step;
2. keeps a small local candidate set per step;
3. retrieves curated UniProtKB evidence for pH dependence, temperature dependence, cofactors, activity regulation, and subcellular location;
4. computes theoretical pI from the UniProt sequence with Biopython as an auxiliary physical-property clue;
5. performs a bounded global candidate-combination rerank, keeping the single-step model score as the dominant signal while penalizing explicit multi-enzyme condition conflicts;
6. reports shared condition windows only when **every** selected enzyme has the relevant evidence. Missing data is reported as unknown, never as compatibility.

This layer does not claim to predict precipitation or long-term stability. Protein concentration, pI, ionic strength, buffer chemistry, solvent, substrates/products, tags, aggregation state and time still require wet-lab mixture/stability tests. SABIO-RK and BRENDA are treated as future pluggable evidence sources rather than hard dependencies: the current deployment has no BRENDA credentials and the currently documented SABIO endpoint was not reachable from the model server during integration testing.

## Feedback

The isolated app exposes `POST /api/feedback`. The frontend feedback dialog can collect a coarse usefulness rating, category, free-text comment, optional contact information, and a small task-context snapshot. Records are appended server-side to `results/catalyst_finder_runtime/feedback.jsonl`; this runtime path is not part of the frontend source tree.

Feedback is deliberately **write-only from the public app**. There is no unauthenticated GET endpoint for messages or contact details. The runtime directory is mode `700`, and `feedback.jsonl` is forced to mode `600` after writes because the optional contact field may contain personal information.

Server-side reporting commands:

```bash
# aggregate counts + recent messages (contact redacted by default)
scripts/catalyst_finder/manage.sh feedback-summary 10

# recent records as JSONL, still redacting contact by default
scripts/catalyst_finder/manage.sh feedback-tail 20

# machine-readable aggregate + recent records
scripts/catalyst_finder/manage.sh feedback-json 20

# advanced filters, e.g. the last 7 days
.venv/bin/python scripts/catalyst_finder/feedback_report.py --days 7 --json

# only a trusted server operator should request contact values
.venv/bin/python scripts/catalyst_finder/feedback_report.py --days 7 --include-contact --json
```

`feedback_report.py` skips malformed JSONL lines while reporting their count, so one damaged line does not block aggregation.
