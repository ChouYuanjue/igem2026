# RecAdam upstream notice

`recadam.py` is vendored from the official RecAdam repository:

- Repository: `https://github.com/Sanyuan-Chen/RecAdam`
- Upstream commit used: `505ba3c265d5b6b90996dddd254f3eb38adaabae`
- Paper: Chen et al., *Recall and Learn: Fine-tuning Deep Pretrained Language Models with Less Forgetting*, EMNLP 2020.
- License: Apache-2.0; the upstream license is retained as `RECAdam_LICENSE`.

The optimizer source is kept as upstream code. Catalyst-specific scheduling and parameter matching live in `train_general_evidence_retriever.py` rather than modifying the vendored optimizer.
