# Experimental Touch Cascade

Independent evidence-acquisition and T0–T5 tagging module. It is intentionally **not** part of the R2E model/ranking implementation.

## Goal

Keep the biological output simple (`T0`–`T5`) while making evidence acquisition staged by cost:

1. **Stage 1 — cheap registry pass, whole library.** Exact-sequence public presence plus lightweight UniProt summary. Most candidates stop here as T0/T1. High-recall evidence signals and model-priority focus candidates are promoted.
2. **Stage 2 — structured evidence pass, shortlist only.** Rich UniProt experimental fields and PDB verification. Candidates with sufficient structured evidence are finalized; linked-publication candidates and high-priority focus candidates can be promoted.
3. **Stage 3 — deep targeted pass, small shortlist only.** Exact-sequence UniProt-linked PubMed evidence is treated as identity-confirmed; extra alias-search papers require an explicit candidate accession/locus/protein-ID match before they can auto-upgrade. PubMed search/article results are cached, and article fetches are batched. Ambiguous literature becomes a review item rather than a guessed T assignment.

`evidence_stage` is operational provenance, **not another biological tier**. The user-facing biological level remains only T0–T5.

## Current NJU cascade geometry

Profile `configs/experimental_touch/profiles/nju_lab_gbk_20260812.yaml` uses the 17 unique reaction rankings as an input-only focus list:

- Stage 2 force-focus: Top 200 per reaction group = 2,859 unique candidates.
- Stage 3 force-focus: Top 50 per reaction group = 773 unique candidates.
- Existing structured evidence signals are OR'ed into the Stage 2 queue.

The current NJU run has completed all three stages:

- input: 131,532
- Stage 1 finalized: 128,375
- Stage 1 -> Stage 2: **3,157** (2.40% of the library)
- Stage 2 finalized: 2,039
- Stage 2 -> Stage 3: **1,118** (0.85% of the library)
  - 773 are forced Top-50 model-priority candidates
  - 345 additional candidates were retained because the structured pass found linked-publication/other escalation signals
- Stage 3 finalized: **1,118 / 1,118**
- Stage 3 unresolved review: **0**
- literature-driven tag upgrades relative to the validated v2 snapshot: **12**
- final strict T2+ pool: **37**
- final tag counts: T0=31,171; T1=100,324; T2=1; T3=7; T4=8; T5=21

Thus the expensive literature layer was reduced from 131,532 proteins to 1,118 while the Top-200/Top-50 model-priority candidates could not be accidentally filtered out just because public annotation was sparse. Stage 3 is an explicit command rather than an implicit part of ranking, so expensive evidence acquisition stays auditable and independently rerunnable.

## Database separation

The module has three distinct roles. They must never be collapsed into one database.

### 1. Candidate source — read only

For NJU this is the existing candidate CSV/metadata under:

`local_candidate_libraries/nju_lab_gbk_20260812/candidates/`

The cascade reads candidate IDs/sequences and source metadata. It never updates the candidate source, `normalized/library.sqlite`, or any production database.

### 2. Public evidence store — reusable across runs

Current path:

`local_candidate_libraries/experimental_touch_evidence/uniprot_2026_02/evidence.sqlite`

Contains public evidence keyed by sequence checksum/public accession/PDB/publication. It does not contain R2E scores or reaction rankings. Candidate-specific publication mappings are kept in the run database; public literature rows use public keys such as `uniprot:<accession>` or `seqmd5:<checksum>`.

### 3. Run-state database — one library/run

Current bootstrap:

`local_candidate_libraries/experimental_touch_runs/nju_lab_gbk_20260812/bootstrap_v2_20260813/run.sqlite`

Contains candidate hashes, focus membership, stage decisions, final/provisional T tags and review overrides. It does not contain full protein sequences or production registry rows.

The profile loader enforces `allowed_evidence_root` and `allowed_run_root`; a runtime database configured outside those roots fails before execution. The production/upstream database repository is explicitly forbidden as a runtime destination.

## Quick switching

Nothing in the Python code is NJU-specific. Switch candidate libraries/evidence snapshots by switching profile:

```bash
PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile configs/experimental_touch/profiles/nju_lab_gbk_20260812.yaml doctor
```

A new candidate library should get its own profile and `run_root`. A public evidence snapshot may be shared deliberately by pointing several profiles to the same evidence database; candidate run state remains separate.

Template:

`configs/experimental_touch/profiles/template.yaml`

## Commands

Initialize a fresh staged run:

```bash
PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile <profile.yaml> init --run-id <run_id>

PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile <profile.yaml> stage1 --run-id <run_id>

PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile <profile.yaml> stage2 --run-id <run_id>

PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile <profile.yaml> stage3 --run-id <run_id>
```

Inspect queue/tier counts:

```bash
PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile <profile.yaml> status --run-id <run_id>
```

The existing v2 full scan can be migrated without trusting candidate-level aggregate evidence as accession-level evidence:

```bash
PYTHONPATH=src ./.venv/bin/python -m experimental_touch_cascade.cli \
  --profile configs/experimental_touch/profiles/nju_lab_gbk_20260812.yaml \
  bootstrap-v2 --run-id bootstrap_v2_20260813 \
  --v2-tags local_candidate_libraries/nju_lab_gbk_20260812/candidates/experimental_touch_v2.tsv.gz
```

Only exact sequence -> UniParc/accession mappings are imported into the reusable public evidence DB. Aggregated legacy UniProt evidence is retained only as the bootstrap run's current tag, so it is never incorrectly attributed to every linked accession.

## Manual review contract

Stage 3 never upgrades an ambiguous paper merely because experimental keywords occur. If the paper cannot be tied to an exact candidate alias/accession, the run gets `REVIEW`. A reviewed CSV can then be applied with:

```text
candidate_id,touch_level,evidence_ref,note,reviewer
...
```

via `apply-overrides`. This keeps expensive judgment limited to the small unresolved queue and makes every override auditable.

## Legacy v2

The old all-131k full-scan implementation is retained as `legacy_v2_fullscan.py` only for reproducibility. The private-library path `profile_all_candidates_touch_v2.py` is now a thin compatibility wrapper; future work should use the cascade CLI.
