# Biological Support-Geometry Audit

## Purpose

This audit asks what the ranking-neutral support coordinates mean. It deliberately does **not** fit a confidence model, choose a threshold, or alter the canonical v8 ranking.

The explanation probe is ranking-identical to canonical v8 on the first 8 fold-0 queries: reciprocal rank, AP, ROC-AUC, NDCG@10, Hit@10/20/50, and best-positive rank all have exact per-query difference zero.

## Unselected Top-1 witness audit

The first eight fold-0 queries were taken in their existing deterministic order; no case was selected for a favorable explanation. The examples range from strong precedents to weak extrapolation.

Strong-support examples include:

- `RHEA:10048 -> O31631`: the local positive relation contains the same protein on another related reaction; the nearest witness has sequence identity 1.0.
- `RHEA:10248 -> P53369`: again the same protein occurs as a known positive for a neighbouring reaction.
- `RHEA:10020 -> P0DPE4`: nearest witness `RHEA:34111 -> Q9KRL3` has global sequence identity about 0.84, ESM-C cosine about 0.997, CLIP structure cosine about 0.992, and mapped reaction-centre similarity 0.80.

A weak-support example is `RHEA:10164 -> A2SSV1`: its nearest witness has DRFP similarity about 0.03, mapped reaction-centre similarity 0.20, and global sequence identity about 0.14. The report preserves this weak evidence rather than converting it into a favorable narrative.

## Support geometry is not calibrated confidence

A broader audit used the actual canonical-v8 Top-1 predictions for the first 64 fold-0 queries. The reaction- and protein-axis locality coordinates were constructed without held-out labels; labels were inspected only afterward for this audit.

Results:

- reaction locality vs reciprocal rank Spearman: `0.0472`;
- protein locality vs reciprocal rank Spearman: `0.0126`;
- sum of the two locality coordinates vs reciprocal rank Spearman: `0.0003`.

Therefore these coordinates must **not** be described as a confidence or uncertainty score. Their role is descriptive: they locate the prediction relative to known biochemical support on the two factors of the product manifold.

Artifact: `results/product_manifold_applicability_audit_v1/fold0_q64.summary.json`.

## Evidence regimes that can be reported without thresholds

The support relation supplies provenance facts that are biologically meaningful without becoming ranking rules. In the same 64-query audit, `60.94%` of Top-1 candidates had the exact candidate protein among the local known-positive precedents. Such a prediction can be described as **reaction-side functional transfer for an already observed enzyme**.

When no exact-protein precedent exists, the report instead shows the nearest distinct proteins and their sequence/structure relationships. This is **protein-side transfer from distinct enzyme precedents**. No identity cutoff is required to distinguish these two provenance regimes.

These descriptions say where the hypothesis comes from; they do not assert that one regime is more accurate than the other.

## Reporting rule

Use the following language:

- `support geometry` or `evidence regime` for the continuous reaction/protein locality coordinates;
- `same-protein precedent` when the candidate itself occurs in a geometrically local known positive pair;
- `distinct-protein precedents` otherwise;
- `extrapolative support` only as a geometric description of weak/distant precedents, never as a calibrated probability of failure.

Do not use `confidence score`, `uncertainty score`, or a high/medium/low cutoff unless a separate frozen calibration study is performed.
