# BRIDGE canonical dataset v1

The canonical BRIDGE benchmark uses one frozen enzyme-reaction training graph, one shared development set and one common formal test set for R2E and E2R.

| Dataset | Relations | Published HF file |
| --- | ---: | --- |
| Training | 218,537 | `benchmarks/main/train.csv` |
| Shared development | 5,216 | `benchmarks/main/validation.csv.gz` |
| Formal test | 21,505 | `benchmarks/main/test.csv.gz` |

**Current published host:** [`ChouYuanjue/atlas-bridge`](https://huggingface.co/ChouYuanjue/atlas-bridge), model snapshot `6b0850a8aac6357869037912a51fbdd66dc0fca4`. The repository already contains all three membership files, the 23,773-relation parent inventory at `benchmarks/main/support/bridge_relation_unseen_max_v3/targets.csv`, the Rhea extension at `benchmarks/main/support/bridge_rhea_longitudinal_growth_v4/targets.csv`, and the evaluation ranking caches. Their published bytes have been checked against the source SHA-256 locks. HF's validation gzip differs at the container level from a freshly generated gzip; the decompressed CSV identity is fixed in this record.

**Official GitLab:** `benchmarks/bridge/canonical-v1/` holds this release definition, `manifest.json`, the split-generation command and source-only regression tests. The published CSVs are retrieved from the HF snapshot above. We deliberately do not commit a second copy of upstream-derived relations to GitLab.

## Exact split recipe

From the 23,773 frozen parent associations, deterministic protein-query hashing gives 2,268 development and 21,505 test associations. Another 2,948 nonoverlapping Rhea relations complete the shared development set. That set has 3,900 learning and 1,316 inner-validation associations. All three sets have zero identical positive associations in common. Development and test have disjoint protein-query IDs; 516 reaction-query IDs occur in both.

The generator uses only the Python standard library. Download the source CSVs from the pinned HF snapshot, and run:

```sh
python3 scripts/maintenance/rebuild_bridge_canonical_splits.py \
  --train /path/to/benchmarks/main/train.csv \
  --parent /path/to/benchmarks/main/support/bridge_relation_unseen_max_v3/targets.csv \
  --rhea /path/to/benchmarks/main/support/bridge_rhea_longitudinal_growth_v4/targets.csv \
  --output /tmp/bridge-canonical-v1
```

This produces `train.csv.gz`, `validation.csv.gz`, `test.csv.gz` and `rebuild-report.json`. Source byte checksums, all split counts, query separation and exact **decompressed** CSV SHA-256 must match `manifest.json`. Omit `--output` for an in-memory verification. The locally generated gzip containers are deterministic (`mtime=0`); the scientific CSV content is exactly the same as HF's snapshot.

This split is the published ranking evaluation protocol. The neural relational model's separate training-internal double-cold folds remain part of its training recipe; their fold sizes are not the 5,216/21,505 evaluation populations.

## Long-term release layout

The preferred destination for a separately citable dataset is a **Hugging Face Dataset** repository `ChouYuanjue/atlas-bridge-benchmarks` (planned, not yet created), using:

```text
canonical-v1/
  splits/
    train.csv.gz
    validation.csv.gz
    test.csv.gz
  sources/
    parent_targets.csv.gz
    rhea_targets.csv.gz
  ranks/
    r2e/
    e2r/
  manifest.json
```

The current HF model snapshot remains usable and must not be broken by this migration. A future dataset revision will be recorded with its exact commit hash and every relevant file SHA-256 before any download path changes. Our trained encoders, gates and application resources remain in the existing HF **model** repository under `weights/`, `evidence/`, `universe/` and `application/`.

Upstream association data and external checkpoints retain their original attribution and redistribution terms; our Apache-2.0 software license does not override those terms. A dedicated Dataset repository should carry its own provenance and license notices. The source-only GitLab CI checks versioned split identity without downloading data; the canonical reproduction command separately validates the 23 frozen evaluation assets and recomputes all reported metrics.
