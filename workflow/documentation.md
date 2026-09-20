# Portable FIBRE workflow

The workflow rebuilds a frozen FIBRE reference bundle from a declared
enzyme-reaction dataset. It is intentionally separate from the currently
deployed Starase Navigator candidate universe.

## Steps

1. validate the protein, reaction and positive-support tables;
2. compute or import one global protein coordinate;
3. compute or import one global reaction coordinate;
4. build or import each sparse factor affinity graph;
5. convert affinities to normalized graph lengths;
6. compute positive-support-to-all geodesic transforms;
7. emit an immutable reference bundle and provenance manifest.

The workflow does not materialize a reaction-by-protein score matrix. Query
sections are evaluated lazily from support distances.

## Scientific versus execution configuration

The YAML config contains dataset paths, feature/encoder choices and graph
construction policy. CPU threads, GPU assignment, memory, scheduler queues and
cluster/cloud execution belong to Snakemake execution options or profiles.

## Scale modes

dense_exact is the canonical small-universe route. blockwise_exact preserves
the same single-view self-tuned affinity with bounded peak RAM. faiss_hnsw is
an explicitly approximate opt-in backend and is recorded as exact=false in the
bundle manifest. A user-supplied sparse affinity is also accepted as an
extension contract.

## Test profiles

`.test/config.yaml` uses tiny tracked precomputed vectors and is the CI/catalog
smoke test. `.test/config_fresh.yaml` recomputes ESM-C and DRFP features from
the source tables and is the local clean-room integration test when model
weights are available.
