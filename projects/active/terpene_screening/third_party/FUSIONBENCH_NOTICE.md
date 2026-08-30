# FusionBench upstream notice

`fusionbench_fisher.py` contains the generic Fisher-weighted parameter-merging
helpers adapted by extraction from FusionBench:

- Repository: `https://github.com/tanganke/fusion_bench`
- Upstream commit: `54c9e8c9d9621620c720452cd8533332a32d3689`
- Source: `fusion_bench/method/fisher_merging/fisher_merging.py`
- License: MIT; retained as `FUSIONBENCH_LICENSE`.

The upstream project credits MergeLM for the original implementation. We use
FusionBench as the directly reused source because its repository carries an
explicit MIT license. Catalyst-specific retrieval losses and Fisher data
sampling remain outside this vendored module.
