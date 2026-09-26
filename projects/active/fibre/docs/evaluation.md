# FIBRE evaluation contract

FIBRE is evaluated as an interaction atlas learned from sparse paired observations and rich per-object molecular views.

## Claim boundary

Only the fibre-reproduction profile may support benchmark-performance claims. It freezes the data snapshot, split, candidate universe, training inputs, model configuration and evaluator. starase-application may use more information, but its outputs are not benchmark evidence.

Strict protein+reaction double-cold evaluation remains the central open-world audit: neither held-out protein nor held-out reaction may occur in the paired training set.

## Atlas admission

A proposed local chart or expert is admitted only if it is evaluated under a frozen protocol.

The evaluation should report:

- full-atlas ranking metrics in both R2E and E2R;
- ablation of the proposed chart while leaving the rest of the atlas unchanged;
- chart applicability / partition-mass distribution;
- overlap consistency with simultaneously active charts;
- performance conditioned on the chart being strongly applicable;
- exact behavior when the chart is unavailable;
- candidate-universe coverage and runtime cost.

A chart is useful only when it provides complementary information without making the global model less reliable outside its biochemical support.

## Universal-coverage checks

Every promoted atlas change preserves:

- one broad raw-input chart for every valid reaction/enzyme pair;
- no candidate removal caused by optional-view absence;
- no abstention requirement;
- raw reaction input remains sufficient for the broad reaction path;
- raw protein sequence remains sufficient for the broad enzyme path;
- unavailable local charts receive zero partition mass and the remaining chart weights renormalize.

These are functional invariants separate from ranking accuracy.

## Multi-view and overlap checks

Different molecular representations may have unrelated latent dimensions. Evaluation therefore does not require embedding-distance agreement.

Instead, when two charts are simultaneously applicable, compare their scalar interaction estimates and report the overlap/gluing loss

\[
\mathcal L_{\mathrm{glue}}
=
\mathbb E
\sum_{\alpha<\beta}
\rho_\alpha\rho_\beta
(K_\alpha-K_\beta)^2.
\]

A useful new representation should either improve the global ranking or add calibrated/complementary information on its support, while maintaining agreement on overlaps.

## TPS family chart

TPS specialization is evaluated as a biochemical chart, not as a TPS-only candidate universe.

A clean TPS admission therefore keeps a broad frozen candidate universe and asks whether the TPS chart improves TPS-relevant queries/candidates inside that same universe. Dataset membership may define an evaluation cohort, but it may not define chart applicability at inference time.

The historical TPS-only and MARTS-specific evaluations remain useful lineage evidence, but they do not by themselves prove broad-universe TPS-chart effectiveness.

## Wet-lab feedback

Train-free experimental updates are evaluated inside the same chart system. Tests verify:

- the observation is projected only into applicable charts;
- update rank and magnitude are controlled;
- every previously scoreable candidate remains scoreable;
- held-out observations improve or correctly revise local interaction estimates without encoder retraining;
- source and assay context remain recoverable.

## Non-promoted bilinear experiment

The earlier global \(I+B\) bilinear correction is retained as a negative/mixed development record, not an atlas component. Its fixed three-fold result is stored at reproducibility/bime_rank/records/FIBRE_CATALYTIC_INTERACTION_RESIDUAL_DEV_V1.json.
