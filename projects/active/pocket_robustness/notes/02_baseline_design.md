# Baseline Design

## Baseline 0: Official EnzymeCAGE

- Pocket source: upstream EnzymeCAGE default route, usually AlphaFill when
  available or the repository's documented fallback
- Score generation: official EnzymeCAGE inference
- Aggregation method: none, one pocket hypothesis
- Expected insight: reference behavior and reproducibility target

## Baseline 1: P2Rank Top-1

- Pocket source: P2Rank top-ranked pocket
- Score generation: EnzymeCAGE inference using top-1 pocket input
- Aggregation method: none or `max` over a single pocket
- Expected insight: practical fallback behavior when AlphaFill is unavailable

## Baseline 2: P2Rank Top-K Max Aggregation

- Pocket source: P2Rank top-k pockets
- Score generation: EnzymeCAGE score per pocket hypothesis
- Aggregation method: maximum pocket score per enzyme
- Expected insight: whether any plausible predicted pocket can rescue ranking

## Baseline 3: P2Rank Top-K Mean Aggregation

- Pocket source: P2Rank top-k pockets
- Score generation: EnzymeCAGE score per pocket hypothesis
- Aggregation method: arithmetic mean
- Expected insight: whether overall pocket compatibility is stable across
  multiple hypotheses

## Baseline 4: P2Rank Top-K Rank-Weighted Aggregation

- Pocket source: P2Rank top-k pockets
- Score generation: EnzymeCAGE score per pocket hypothesis
- Aggregation method: weight = `1 / pocket_rank`
- Expected insight: whether detector rank is a useful prior for retrieval

## Baseline 5: P2Rank Top-K Softmax-Pooling Aggregation

- Pocket source: P2Rank top-k pockets
- Score generation: EnzymeCAGE score per pocket hypothesis
- Aggregation method: softmax over pocket scores with configurable temperature
- Expected insight: smooth version of "best pocket wins" that still uses all
  hypotheses

## Future: fpocket Top-1 and Top-K

- Pocket source: fpocket geometry-derived pockets
- Score generation: EnzymeCAGE score per converted pocket
- Aggregation method: same as P2Rank top-k
- Expected insight: detector disagreement and geometry-only pocket behavior

## Future: ScanNet Residue Prior

- Pocket source: P2Rank/fpocket plus ScanNet residue scores
- Score generation: EnzymeCAGE score plus residue evidence
- Aggregation method: reranking prior or evidence feature outside EnzymeCAGE
- Expected insight: whether residue-level binding evidence corrects bad pockets

## Future: Catalytic-Residue-Aware Rerank

- Pocket source: predicted pockets plus catalytic residue evidence
- Score generation: EnzymeCAGE score plus catalytic residue coverage
- Aggregation method: evidence-aware reranking
- Expected insight: whether failures are caused by missing catalytic residues
