# Catalyst Finder

Catalyst Finder is the current conversational research interface for enzyme–reaction retrieval, database evidence, literature/structure inspection, route design, and pathway compatibility. The public product is organized around scientific questions rather than internal model switches.

The frontend is intentionally isolated from retired portal implementations. Production code lives under `frontend/catalyst_finder/` and `scripts/catalyst_finder/`; runtime models, caches, databases, secrets, and generated results live outside Git under ignored `data/` and `results/` paths.

## Core interaction model

`POST /api/agent/resolve` is the single natural-language entry point. A model-led scientific harness composes explicit tools for entity resolution, verified database relations, literature/structure evidence, model candidate search, route design, and pathway compatibility.

Natural-language intent is not recovered with keyword/regex classifiers. DeepSeek maps user language to bounded semantic choices such as retrieval direction, result scope, Few-shot/Zero-shot policy, taxonomy scope, or candidate universe. Deterministic Python code is responsible for invariants: identifier parsing, database verification, candidate-universe membership, confirmation provenance, and mutually exclusive policy combinations.

The LLM is never trusted to invent Rhea, UniProt, ChEBI, PMID/PMCID/DOI, Pfam, or other scientific identifiers. Identifiers used by downstream tools must come from verified tool output or explicit structured input.

## Retrieval directions and positive-context policy

Catalyst supports both directions symmetrically:

- **Reaction → enzyme (R2E)** uses verified positive enzymes as **protein-space Few-shot anchors**.
- **Enzyme → reaction (E2R)** uses verified recorded activities as **reaction-space Few-shot anchors**.

When verified database positives exist, the normal production policy is Few-shot in either direction. If no verified positives exist, the run is Zero-shot. A user can explicitly request Zero-shot and disable positive-context guidance.

User-supplied positives extend the verified database seed set only after server-side resolution and confirmation. Confirmation is bound to the current session, target entity, displayed verification card, and—for external protein seeds—the verified sequence digest. A client cannot turn an arbitrary ID into a “confirmed positive” merely by POSTing a field named `confirmed_*`.

### Result scopes

The result scope is independent of whether verified positives are used as Few-shot context:

- `separate_known` — **default**. Database-recorded associations are shown as factual evidence; model Top-K contains unrecorded candidates. Verified positives may still guide Few-shot retrieval.
- `exclude_known` — return only unrecorded model hypotheses, while keeping recorded evidence separately available. This is an output filter, not a request to disable Few-shot.
- `known_only` — when the user explicitly asks to rank only verified-known associations by the model, the verified-known set becomes an exact candidate subset and is scored **Zero-shot**. The same items are not simultaneously used as Few-shot anchors.
- `rank_with_known` — retrospective mixed ranking of recorded and unrecorded candidates in one list. This is deliberately **Zero-shot** so known and unknown entries receive the same scoring treatment. It is useful for checking whether the model independently recovers known biology and is not a normal default.

A dedicated TPS candidate universe is another specialist mode. It is used only when the user explicitly requests the TPS-specialized search scope; it is not selected merely because a reaction “looks terpene-like”. Its value is that the underlying retrieval components were specialized/evaluated for that domain.

Other advanced constraints—eukaryote/prokaryote scope, remote-family/cross-cluster search, Top-3/5/10/20 budgets—remain parameters of the same candidate-search capability rather than separate scientific tools.

## Model-score presentation

The default UI displays one compact **model score on a 0–100 scale**. This is a display-oriented evidence-strength score already produced by the retrieval engine; it does not change candidate order and it is not a calibrated activity probability.

The UI does **not** min-max normalize each query so that every Top-1 becomes 100. Consequently, Top-1 candidates from two queries on the same route can still have visibly different support. Different route families—especially Zero-shot vs Few-shot and direct neural scores vs RRF—are not interpreted as a single globally comparable probability scale.

Raw retrieval score, score source, route ID, candidate universe, applicability/reliability metadata, and the display-score formula are kept in the existing technical-details disclosure rather than repeated in the main result surface.

## Verified evidence and research workspace

The research workspace can combine multiple evidence modules without forcing every request to run all of them:

- **UniProtKB** — protein identity, annotation, catalytic activity, cofactors, cross-references, curated publication links.
- **Rhea** — verified reaction identity, participants, direction and official protein mappings.
- **InterPro/Pfam and functional scopes** — verified membership and aggregated recorded relations.
- **RCSB PDB / AlphaFold DB** — experimental/predicted structure evidence.
- **Europe PMC** — database-linked references and broader biomedical search.
- **OpenAlex** — complementary scholarly discovery beyond the biomedical index.

Finite source sets are returned completely and paginated inside their collapsed panel; the backend no longer silently truncates a finite list before the frontend paginates it. Remote search providers use real cursor pagination.

Literature is intentionally layered instead of presented as one misleading “complete” count:

1. database-linked/curated references (high precision);
2. broad Europe PMC search;
3. OpenAlex scholarly discovery.

Cross-provider literature entities are deduplicated by stable identity (PMID/DOI where available). An OpenAlex-discovered work can be enriched with Europe PMC content while retaining provenance from both sources.

### Transient external failures

UniProt, Europe PMC, and OpenAlex are live sources. Catalyst performs bounded retries for transient network/429/5xx failures and persists the **last successful response** under the ignored Catalyst runtime cache. A transient failure may therefore return a recent successful snapshot marked with:

- `source_freshness = stale_cache`
- `stale_cache_age_seconds`
- a non-sensitive `live_fetch_error` type

Live data is always preferred. Stale fallback is never used for deterministic 4xx/not-found failures, and the cache is not treated as a local mirror of the upstream database.

## Grounded synthesis

Scientific prose after tool use is grounded in the run-scoped evidence ledger and structured result. Once a scientific tool has been attempted, the assistant cannot fall back to ordinary model-memory prose.

The synthesis layer checks requested-entity completeness and protects high-risk scientific qualifiers. Unsupported identifiers, subcellular locations, inhibition categories, full-text claims, and similar sensitive assertions trigger constrained rewriting rather than being silently accepted. Correction/erratum records remain distinct from the scientific articles they modify.

## Bilingual behavior

Chinese and English UI sessions are separate. Product labels, accessibility text, backend UI labels, error fallbacks, controller summaries, and grounded synthesis follow `ui_language`; switching languages does not reuse the other language's conversational session.

Scientific proper names and original source content—e.g. `UniProtKB`, `Rhea`, `Europe PMC`, paper titles, species names, database IDs—remain in their canonical/original form rather than being artificially translated.

Static tests guard the English/Chinese slots against accidental cross-language product copy.

## Route design and pathway compatibility

**Route design** accepts source/target chemistry in natural language, resolves the endpoints, and searches the verified Rhea graph. Known Rhea routes and explicitly predicted MINE/Pickaxe bridges remain separate evidence layers. Optional thermodynamic (eQuilibrator/MDF) and E. coli host-flux (COBRApy/iML1515) analyses are isolated worker runtimes and do not run inside the web process.

**Pathway compatibility** evaluates a fixed multi-step pathway. Unspecified enzymes can be supplied by the production R2E ranker, while curated UniProt evidence is used to assess pH, temperature, cofactor and location compatibility. Missing evidence remains unknown; it is not converted into a positive compatibility claim.

## Public routes vs internal implementation

`GET /api/routes` is a **Catalyst product projection**, not a dump of every research/CLI switch in the repository. Publicly meaningful route capabilities are exposed with short product names. Manual model overrides, temporary-universe engineering switches, batch-only overlays, CAGE rescue internals, conformal/reliability internals, hard-negative/dual-kernel components and similar implementation details remain technical execution metadata rather than user-selectable scientific abilities.

The route-catalog implementation itself lives in `scripts/catalyst_finder/route_catalog.py`; Catalyst has no runtime dependency on the retired portal source tree.

## Runtime and Git boundaries

The service is normally deployed on port `8791` through the user systemd service / management script.

```bash
cd /home/s241850073/igem2026
bash scripts/catalyst_finder/manage.sh start
bash scripts/catalyst_finder/manage.sh status
```

Runtime state and secrets live under ignored paths, primarily:

```text
results/catalyst_finder_runtime/
```

Configure DeepSeek without placing the key in tracked source:

```bash
bash scripts/catalyst_finder/manage.sh configure-key
```

`data/`, `results/`, local reports, model weights, external runtime caches, generated candidate libraries, and retired local-only code are deliberately ignored by Git. A clean source checkout can run source-level tests without the deployment assets; tests that explicitly validate provisioned deployment assets skip when those assets are absent.

## Main endpoints

```text
GET  /api/status
GET  /api/routes
GET  /api/capabilities
POST /api/agent/resolve
POST /api/session/view-context
POST /api/rank
POST /api/rank-reactions
POST /api/route/design
POST /api/pathway/analyze
POST /api/feedback
```

`/api/status` is the source of truth for the currently deployed build revision, candidate-universe sizes, evidence-graph size, default retrieval policy, DeepSeek availability, and capability version. README text intentionally does not hard-code changing universe sizes.

## Tests

Run the Catalyst suite:

```bash
.venv/bin/python -m pytest -q scripts/catalyst_finder
```

Run the active production-retrieval tests:

```bash
.venv/bin/python -m pytest -q projects/active/terpene_screening/tests
```

Before deployment, also run:

```bash
python3 -m compileall -q scripts/catalyst_finder

git diff --check

git ls-files -ci --exclude-standard
```

The last command should print nothing: ignored runtime assets must not remain tracked.

## Feedback and logs

`POST /api/feedback` appends runtime feedback to the ignored Catalyst runtime directory. Public feedback is write-only; contact information is not exposed through a public read endpoint.

HTTP access logs redact common sensitive query parameters (`auth`, `token`, `access_token`, `api_key`, `key`, `secret`) while retaining non-sensitive request context.
