# Structural Geometry Coverage Audit

## Decision

Pocket / pocket-3Di / pocket-OT are **not part of the current product-manifold mainline**. This is a coverage decision, not a negative judgment on those representations.

The product-manifold method permits structural information only as an observed coordinate of the same protein metric `g_E`. Missing structure must remain missing geometry, not a gate, penalty, fallback, or alternative score expert. A structural coordinate is therefore useful only when it is sufficiently represented inside the local protein charts on which the field is solved.

## Current assets

The canonical retrieval universe contains `185,918` proteins. The currently materialized P2Rank asset contains `1,287` proteins with successful pockets and no failed structures/pockets in that targeted subset, i.e. only about `0.69%` of the full protein universe.

The intended derived output directories

- `data/terpene_3di_v1`
- `data/terpene_pocket_3di_v1`
- `data/terpene_pocket_ot_v1`

are currently absent. Their builders are present in the repository, but these representations are not yet materialized as canonical full-support assets.

## Label-blind chart occupancy

Using the same 432-node numerical protein charts as the clean-dev evaluator, without using dev labels:

- fold 1: pocket nodes mean `6.40`, median `0`, p90 `1`; `534/611` charts contain no pocket node, and only `47/611` contain at least five;
- fold 2: pocket nodes mean `4.40`, median `0`, p90 `1`; `596/669` charts contain no pocket node, and only `51/669` contain at least five;
- requiring both pocket and the existing structure coordinate is even sparser: `566/611` fold-1 charts and `604/669` fold-2 charts contain none.

By contrast, the existing CLIP structure coordinate is already dense in these charts: fold-1 mean `370.7/432`, fold-2 mean `362.5/432`, with zero charts lacking the view.

## Consequence for the mathematical object

Adding the present pocket assets to `g_E` would mostly define a metric correction on isolated or nearly isolated observed points rather than a local structural geometry. Any practical attempt to make it matter would therefore require a structure-availability gate or a special-case branch, which violates the one-object mainline.

The representation becomes eligible later only after an unlabeled coverage audit shows that it forms genuine local neighbourhoods on a substantial fraction of canonical charts. At that point it should modify the same protein metric continuously on edges where both endpoints are observed; it must not become an independent score or route.
