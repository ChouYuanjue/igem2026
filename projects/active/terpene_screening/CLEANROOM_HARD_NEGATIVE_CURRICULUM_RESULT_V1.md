# Cleanroom hard-negative curriculum v1 — development result

Protocol was frozen in `CLEANROOM_HARD_NEGATIVE_CURRICULUM_V1.json` before the new runs. Selection used clean2023 strict double-cold folds 0/1/2 only; Enzyme-405 was not used.

| Candidate | Mean joint percentile | Worst-fold percentile | Mean SR@10 | Mean MRR | Mean MAP | Mean AUROC | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| fixed_hard80 | **0.7583** | 0.575 | 0.5680 | **0.2372** | **0.2325** | **0.6586** | **Yes** |
| curriculum_start0 | 0.6833 | **0.625** | **0.5741** | 0.2340 | 0.2293 | 0.6553 | No |
| curriculum_start32 | 0.5500 | 0.425 | 0.5629 | 0.2349 | 0.2284 | 0.6484 | No |
| curriculum_start16 | 0.5083 | 0.250 | 0.5600 | 0.2327 | 0.2248 | 0.6471 | No |

The simple hard-negative warm-up hypothesis is rejected for this recipe. `curriculum_start0` slightly improves mean SR@10 but loses enough early-rank and global ranking quality that the preregistered joint selector retains the fixed 80-hard/8-random schedule. Untouched folds 3/4 are not consumed for a method that failed development selection.
