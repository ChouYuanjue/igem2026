# FIBRE rebuild and product contract

This contract separates a scientific reference build from a deployed query
service. A Starase Navigator product release may bind one frozen FIBRE bundle
to a particular candidate universe while the FIBRE method remains rebuildable
on a different enzyme-reaction dataset.

## Reference build

The minimum source dataset consists of proteins, reactions and accepted
positive_pairs. proteins provides stable protein_id values and amino-acid
sequences. reactions provides stable reaction_id values and directed reaction
SMILES. positive_pairs contains accepted positive protein-reaction
observations. Unknown pairs are not negatives.

Factor geometry is label-free. positive_pairs enters only correspondence
support. A build records input hashes, graph backend and exact/approximate
status, graph policy, feature assets, support indices and support-to-all
geodesic transforms.

Changing a reference entity, feature/encoder definition, graph policy or
promoted positive support creates a new reference build. A new build must not
overwrite the provenance of an old one.

## Product query

A query against a frozen build is not training. The reference side is
precomputed. Query-side work computes the same feature coordinate, attaches
the query to the frozen factor graph using reference local scales, computes one
query-to-reference geodesic transform and evaluates the existing FIBRE section
against frozen positive support.

## Few-shot evidence

A verified session protein-reaction pair is runtime evidence. It may be used by
an explicitly seed-aware route, but it is not silently merged into the
reference support table. Promotion requires an explicit decision to update the
source dataset and create a new reference build.

## Scale policy

dense_exact is the canonical small-universe implementation. blockwise_exact
computes the same single-view geometry with bounded peak memory but remains
quadratic in compute. faiss_hnsw is an opt-in approximate large-universe
backend and is always identified as exact=false. A user-supplied sparse
affinity is also allowed, with its builder and provenance becoming part of the
release contract.

Approximate and exact builds are different scientific artifacts even when they
share the same source tables.

## Optional observations

Structure, pocket, motif, reaction-centre, mechanism and assay/context
measurements are optional biological coordinates. They are used only where
semantically valid and observed. Missing coordinates are unresolved; absence
is not negative evidence. Expensive coordinates should normally be precomputed
on the reference side and cached by entity/tool version, while query-side
computation remains lazy.

## Historical assets

Historical BiME/FIBRE experiment assets remain reproducibility evidence. They
may seed or validate a new implementation, but a portable build must be able
to recompute every required current artifact from the declared source dataset
and pinned external/foundation dependencies. A historical result file is not a
substitute for a missing current builder.
