# Cleanroom Horizyn MLNCE v1 — development result

Protocol was frozen in Git at `e742edf` before the multifold development run. The experiment reused `dayhofflabs/horizyn`'s official `FullBatchMLNCELoss` directly, while keeping Catalyst's existing cleanroom protein/reaction features unchanged. Selection used only clean2023 strict double-cold folds 0/1/2; no outer benchmark or Enzyme-405 label was read.

| Candidate | Mean joint percentile | Mean SR@1 | Mean SR@3 | Mean SR@5 | Mean SR@10 | Mean DCG@10 | Mean EF@1% | Mean MRR | Mean MAP | Mean AUROC | Mean NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| existing cleanroom baseline | **0.9833** | 0.0864 | **0.2905** | **0.4030** | **0.5680** | **0.4445** | **29.53** | **0.2372** | **0.2325** | **0.6586** | **0.2644** |
| official Horizyn MLNCE | 0.5167 | **0.0881** | 0.2387 | 0.3427 | 0.4962 | 0.4022 | 24.41 | 0.2117 | 0.2113 | 0.5882 | 0.2314 |

The MLNCE-only hypothesis is rejected. It gives a very small mean SR@1 improvement but materially degrades SR@3/5/10, DCG/EF, MRR/MAP, AUROC and NDCG. Therefore the strong Horizyn paper result cannot be attributed to its loss alone in Catalyst's multi-positive cleanroom setting. Do not spend outer benchmarks on this candidate.

The next Horizyn-derived experiment, if any, must isolate representation changes (ProtT5 and/or RDKit+) rather than combining them with this rejected loss. Externally pretrained Horizyn retrieval checkpoints remain excluded from clean generalization claims because their relation/entity exposure is not compatible with our clean boundary.
