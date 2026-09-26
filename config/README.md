# Legacy geometry workflow configuration

> This directory documents the retained portable product-manifold/correspondence workflow. It is preserved for historical reproducibility and compatibility and is not the current FIBRE — Factorized Interaction Basis for Reaction–Enzyme — method definition. For the current method, start with projects/active/fibre/docs/method.md and projects/active/fibre/docs/catalytic_kernel_foundation.md.


The YAML config describes scientific inputs and model-building choices only.
CPU threads, memory, cluster partitions, GPU assignment and scheduling belong
to Snakemake CLI options or execution profiles.

## Required dataset tables

proteins.csv
- protein_id: stable unique identifier inside this build.
- sequence: amino-acid sequence.
- optional aliases/provenance columns are preserved.

reactions.csv
- reaction_id: stable unique identifier inside this build.
- reaction_smiles: directed reaction SMILES containing >>.
- optional aliases/provenance columns are preserved.

positive_pairs.csv
- protein_id, reaction_id: accepted positive correspondence support.
- unknown pairs are not negatives.
- runtime few-shot positives do not enter this table unless deliberately promoted and rebuilt.

## Feature modes

For proteins, esmc reuses the production FIBRE ESM-C mean-embedding encoder;
precomputed accepts an aligned numeric CSV.

For reactions, drfp reuses the production canonicalization + DRFP encoder;
precomputed accepts an aligned numeric CSV.

A precomputed feature CSV starts with protein_id or reaction_id, followed by numeric feature columns.

## Large-database graph mode

The default geometry backend is auto. It uses the existing dense exact
construction for small universes and switches to a bounded-memory exact
construction above geometry.dense_limit. Both routes implement the same
single-view self-tuned FIBRE affinity and are marked exact=true in manifest.json.
The bounded-memory route removes the N x N RAM requirement but remains O(N^2)
in compute, so it is not intended to make million-entity exact construction cheap.

For very large universes there are two explicit choices. Set
geometry.backend: faiss_hnsw to use approximate HNSW candidate retrieval; the
canonical FIBRE endpoint energy is then evaluated inside that candidate pool,
and manifest.json records exact=false plus the approximation parameters. Or
supply a sparse SciPy NPZ affinity graph through geometry.protein_affinity
and/or geometry.reaction_affinity, using an HPC/ANN workflow appropriate to the
deployment scale.

For large faiss_hnsw builds, set geometry.graph_k explicitly. The canonical
ceil(sqrt(N)) graph is intentionally not silently carried into million-entity
deployment because its edge count grows as N^(3/2). An explicit smaller graph_k
therefore defines a distinct approximate build and is recorded in provenance.

In every case FIBRE stores graph lengths and support-to-all geodesic transforms,
not a dense reaction-by-protein field or dense factor all-pairs geodesics.
Exact and approximate builds are different scientific artifacts and their
manifests must not be treated as interchangeable.

## Frozen product queries versus rebuilding

A reference build is immutable. A new query molecule does not need to be
appended to the dataset and does not trigger retraining. When a factor was
built from the portable normalized feature coordinate, the reference bundle
also stores its local self-tuning scales. A new feature vector can be attached
out-of-sample to the frozen graph and evaluated with the same FIBRE section.

Example:

    python -m projects.active.fibre.portable.reference_query \
      --bundle results/my_build/reference_bundle \
      --reaction-feature query_reaction.npy --top-k 20

The analogous --protein-feature route ranks reactions. The query is not written
into positive_pairs.csv and does not mutate the reference bundle.

Rebuild only when the reference universe, encoder/coordinate definition, graph
policy, or deliberately promoted accepted positive support changes. A runtime
few-shot seed is session evidence by default; promotion into positive_pairs.csv
is an explicit data-governance action followed by a new build.

Custom precomputed affinity graphs do not automatically promise feature-space
out-of-sample attachment because their metric may differ from the bundled
feature metric. Such releases must provide a compatible attachment
implementation instead of silently mixing geometries.

## Optional biological coordinates

Global sequence/reaction features are the portable minimum. Extra coordinates
enter the same factor geometry; they are not independent score experts.

Feature-valued views are CSV files containing protein_id or reaction_id plus
numeric columns. They may contain any non-empty subset of the reference IDs.
A missing row means that coordinate was not observed.

Distance-valued views preserve geometries that should not be forced into a
vector representation, for example Foldseek-derived structural diffusion,
pocket optimal transport, or reaction-centre Wasserstein distance. Each view is
a square .npy distance matrix aligned to the reference table plus a boolean
availability .npy. Only the observed submatrix must be finite, symmetric,
non-negative and zero-diagonal; missing endpoints are ignored.

Both types can coexist. The reference bundle stores each view's availability,
self-tuning local scale and input hash. A product query may likewise provide
only the views it actually has. Feature views use --protein-view/--reaction-view
NAME=PATH; distance views use --protein-distance/--reaction-distance NAME=NPY
with a query-to-reference distance vector.

### Fresh reaction-centre reconstruction

The current changed-atom representation is now reproducible on a new reaction
table. First create a pinned atom-mapping registry. The existing resumable
RXNMapper path accepts a new dataset without the historical entries table:

    python -m projects.active.fibre.runtime.reaction_mapping \
      --reactions data/my_dataset/reactions.csv \
      --derive-entries \
      --runtime external_runtime/rxnmapper \
      --output-dir data/my_dataset/rxnmapper

Then either enable precompute.reaction_center in the YAML or run:

    python -m projects.active.fibre.portable.reaction_center \
      --reactions data/my_dataset/reactions.csv \
      --mapped-reactions data/my_dataset/rxnmapper/mapped_reactions.csv \
      --output-dir data/my_dataset/reaction_center

This recomputes exact equal-mass changed-atom W2 and explicit transition-token
Jaccard distances. Mapping failures remain missing coordinates. RXNMapper
confidence is retained as mapping provenance and is never used as a geometric
weight.

The machine-readable maturity inventory is
projects/active/fibre/release/coordinate_registry.yaml. It distinguishes
ready portable builders from coordinates that still require a dataset adapter
and current-dataset-only orchestration.

Before launching a rebuild, inspect the resolved work plan:

    python -m projects.active.fibre.portable.build_plan \
      --config config/fibre.example.yaml

The plan states which coordinates will be freshly recomputed, which are supplied
artifacts, the graph backend, and semantic warnings.

## Execution resources

The YAML file contains scientific choices only. CPU threads, RAM, GPU
assignment, cluster partitions and scheduler options belong to Snakemake CLI
arguments or execution profiles. The same scientific config can therefore run
on a workstation or cluster without changing the scientific object.
