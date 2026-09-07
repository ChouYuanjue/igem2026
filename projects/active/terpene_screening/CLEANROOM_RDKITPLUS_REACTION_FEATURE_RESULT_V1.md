# Cleanroom RDKit+ reaction representation v1 — development result

The experiment was frozen at `383ac94` before development results. It changes only the unlabeled reaction representation: the existing 2115-d Catalyst DRFP+categorical vector is concatenated with the official Horizyn 1024-d standardized RDKit+ structural Morgan fingerprint. Protein features, model architecture beyond input width, training loss, sampling, clean2023 folds and evaluation metrics remain unchanged.

| Candidate | Joint percentile | SR@1 | SR@3 | SR@5 | SR@10 | DCG@10 | EF@1% | MRR | MAP | AUROC | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.500 | 8.64% | 29.05% | 40.30% | 56.80% | 0.4445 | 29.53 | 0.2372 | 0.2325 | 0.6586 | 0.2644 |
| RDKit+ augmented | **1.000** | **11.34%** | **33.65%** | **45.54%** | **61.55%** | **0.5066** | **33.24** | **0.2717** | **0.2618** | **0.6923** | **0.2982** |

RDKit+ wins every metric family on all three development folds. This is the first representation-level intervention in the current iteration with a large, directionally consistent gain. The next step is frozen outer confirmation on reaction-cold and double-cold full-universe protocols. Outer models must preserve the existing base checkpoint by zero-initialized input expansion rather than restarting from random initialization.
