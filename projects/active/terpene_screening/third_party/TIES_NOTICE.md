# TIES-Merging upstream notice

`ties_merge.py` extracts the generic trim / sign-election / disjoint-mean task-vector
merging primitives from the official TIES-Merging implementation:

- Paper: Yadav et al., *TIES-Merging: Resolving Interference When Merging Models*, NeurIPS 2023.
- Repository: `https://github.com/prateeky2806/ties-merging`
- Upstream commit: `44e7891fc84f3de7e4caa52664cd864ca3715e91`
- Relevant source: `src/utils/merge_utils.py`
- License: BSD-3-Clause; retained as `TIES_LICENSE`.

Catalyst only reuses the model-agnostic task-vector operations. Checkpoint loading,
seed matching, directional-tower safety checks, retrieval evaluation and model
provenance are Catalyst-specific.
