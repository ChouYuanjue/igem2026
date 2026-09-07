# Local historical experiment archive

`archive/experiments/` is **not a current result source**. It contains local experiment outputs moved out of `results/` during the 2026-09-07 BiME-Rank release cleanup so that historical sweeps, smoke tests, rejected variants, and obsolete retention runs are less likely to be mistaken for current evidence.

The move is intentionally conservative:

- no archived payload was deleted;
- current/canonical assets remain at their original paths;
- candidates with a live/static dependency or uncertain identity were left under `results/`;
- the reviewed move list is `reproducibility/bime_rank/archive_moves.json`;
- the candidate/blocked audit is `reproducibility/bime_rank/archive_plan.json`;
- historical payloads are ignored by Git; only provenance, verification tools, and this README are versioned.

Many old result JSON files contain their original absolute/relative paths. Historical replay therefore requires restoring a moved directory to its original `results/...` location rather than teaching current production code to read from `archive/`.

Read-only verification:

```bash
.venv/bin/python scripts/maintenance/manage_bime_archive.py verify
```

Preview a restore without changing files:

```bash
.venv/bin/python scripts/maintenance/manage_bime_archive.py restore --path results/<historical_run>
```

Apply a restore only when deliberately reproducing that historical run:

```bash
.venv/bin/python scripts/maintenance/manage_bime_archive.py restore --path results/<historical_run> --apply
```

Do not use archive timestamps or names such as `production_*` to select a model or score. Current BiME-Rank claims are resolved only through `reproducibility/bime_rank/canonical.json`.
