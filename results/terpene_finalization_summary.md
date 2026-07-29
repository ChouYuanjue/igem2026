# Terpene Synthase Retrieval — Current Finalization Summary

- Status: complete
- Canonical report: `docs/terpene_candidate_retrieval_comprehensive_report_zh.md`
- Focused protocol reassessment: `docs/terpene_retrieval_protocol_reassessment_zh.md`
- Tests: 74 passed; 20 known DRFP/NumPy deprecation warnings
- Deployment validation: five neural packages and one dual-kernel sparse package valid
- Registry: 694 external proteins, 240 external reactions
- Discovery audit: 30,822 ranking rows checked, 0 known-association leaks
- Wet-lab procurement: 6 plates, 352 sequence-deduplicated constructs, 184,501 aa
- Git commit: not created in this reassessment

## Evaluation principle

Homology-enabled practical retrieval, exact-new entity retrieval, cluster-cold generalization, and double-cold stress testing are different co-primary tracks. Double-cold is not the single overall accuracy measure.

## Practical and novelty-stratified results

- Current exact-reaction completion R2E Top-10 / Top-20: 48.1% / 57.5%
- Seeded homolog expansion Top-10: 73.7% with 1 seed; 92.8% with 5 seeds
- Exact-new protein E2R Top-10: 72.4%
  - same 50% identity cluster homolog visible in training: 82.6%
  - no same-cluster homolog visible: 38.4%
- Exact-new protein R2E Top-10: 51.2%
  - same-cluster homolog visible: 62.6%
  - no same-cluster homolog visible: 15.5%
- Exact-new reaction R2E / E2R Top-10: 38.0% / 37.9%
- Same-model protein-cluster-cold E2R Top-10: 36.1%
- Same-model reaction-cluster-cold R2E Top-10: 28.7%
- Same-model double-cold R2E / E2R Top-10: 6.4% / 14.1%

## Production routing

- E2R Top-3: freeze-reaction, 5 neighbors, direct weight 0.75
- E2R Top-10: locked neural RRF, primary 0.35 / hard-negative secondary 0.65 / constant 60
- E2R Top-20: primary 0.70 / dual-kernel 0.30 / constant 60
- R2E Top-3: reaction-loss weight 0.75 direct
- R2E Top-10/20: Horizyn exact-residual direct

The independent locked E2R Top-20 confirmation increased Hit@20 from 34.77% to 43.37%, +8.60 percentage points, with paired bootstrap 95% CI +5.02 to +12.54 percentage points.
