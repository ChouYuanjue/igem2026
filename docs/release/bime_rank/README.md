# BiME-Rank judge release

This directory is the tracked judge-facing release source for BiME-Rank.

## Authority boundary

`BiME_Rank_Judge_Report.tex` is a **narrative presentation artifact**, not the numeric source of truth. Current claims must resolve through `reproducibility/bime_rank/canonical.json`; the report is validated against those frozen claim primaries before release.

The imported narrative source was `BiME_Rank_Judge_Report_20260907_v9_1.tex` from the project Library (SHA-256 `8d1ea50f98b6a998fe22d3072ca9d95a2fca2774e470af96f29f7e9e64621c0f`). The bibliography source was `BiME_Rank_Judge_Report_20260907_v9.bib` (SHA-256 `6c87ca7f4574acf59eaa9cd7e36f0f61b6fa912354555e25364b4d17e074278e`).

The release copy intentionally differs from that imported TeX only where the repository evidence required normalization. In the 2026-09-07 release pass, the stale CLIPZyme R2E table values were updated from the frozen `clipzyme_r2e` primary, and the bibliography filename was made repository-stable. See `release_manifest.json` for provenance.

## Releasing safely

Run the following checks from the repository root; none performs model inference or benchmark evaluation:

```bash
.venv/bin/python scripts/maintenance/resolve_bime_asset.py --verify
.venv/bin/python scripts/maintenance/manage_bime_archive.py verify
.venv/bin/python scripts/maintenance/validate_bime_assets.py
.venv/bin/python scripts/maintenance/validate_bime_judge_report.py
```

Do not copy numbers from old PDFs, archived notes, filenames containing `current`, or historical `production_*` directories. If a claim changes after explicit review, update its canonical primary first, rebuild the asset index, then update this narrative and make the validator pass.

Generated PDF/build auxiliaries are verification artifacts and are not committed. The tracked release contract is the TeX source, bibliography, manifest, and canonical evidence graph.
