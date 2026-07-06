# P2Rank Top-K Prompt

The purpose of P2Rank top-k aggregation is to test whether EnzymeCAGE retrieval
depends too strongly on one selected pocket hypothesis.

Current aggregation methods:

- `max`
- `mean`
- `rank_weighted`
- `softmax_pool`

Possible extensions:

- confidence-weighted aggregation using P2Rank scores
- calibrated softmax pooling
- reaction-class-specific pocket priors
- residue-evidence-aware aggregation

Rescued and harmed case analysis:

- compare baseline rank and top-k aggregated rank
- mark a case as rescued when a labeled enzyme moves into top-k
- mark a case as harmed when a labeled enzyme drops out of top-k
- record which pocket rank contributed the best score
- inspect whether the selected pocket includes expected catalytic residue
  evidence
