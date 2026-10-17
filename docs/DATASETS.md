# BRIDGE dataset release

The canonical [dataset record](../benchmarks/bridge/canonical-v1/README.md) fixes 218,537 training, 5,216 shared development and 21,505 formal test relations, as well as all source file digests and query-level split parameters.

The current data files are already published at [`ChouYuanjue/atlas-bridge`](https://huggingface.co/ChouYuanjue/atlas-bridge), `benchmarks/main/`, pinned to immutable commit `6b0850a8aac6357869037912a51fbdd66dc0fca4`. The official iGEM GitLab preserves the lightweight `benchmarks/bridge/canonical-v1/manifest.json` and `scripts/maintenance/rebuild_bridge_canonical_splits.py`. The Python standard-library generator reproduces all three membership CSVs exactly from their published sources.

Our full model and ranking-cache asset registry is `assets/full_reproduction.lock.json`. A dedicated HF Dataset repository can later host the complete benchmark without disrupting the currently published HF model repository.
