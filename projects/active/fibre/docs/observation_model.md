# Biological observation model

FIBRE separates molecular observations, paired catalytic observations and prediction/calibration state.

## Molecular observations

Reaction-side examples include reaction identity, substrate/product structures, DRFP/RDKit representations, atom mapping, changed bonds/atom states and reaction-centre context.

Protein-side examples include sequence, ESM-C/EnzGFM representations, domains, structures, pockets, motifs, catalytic residues and cofactor annotations.

A missing observation is unknown, not negative.

## Paired catalytic observations

A database association, literature product assignment, quantitative assay and explicitly inactive assay are not interchangeable labels. Every pair observation preserves source and scope.

Positive pair observations may train a frozen interaction model or enter an application session through a train-free finite-rank update. Condition-dependent negative observations remain scoped to their reported assay context and are not permanent global negative edges.

## Provenance

Structured records are preferred when semantics match. Literature extraction retains a recoverable source span. Computed structures and motifs are marked as computed rather than measured.

## Model roles

An observation can serve as:

1. raw input to a broad reaction/enzyme encoder;
2. a coordinate or applicability signal for one or more local interaction charts;
3. a train-free pair update;
4. explanatory evidence only.

Promotion from evidence to a score-bearing role requires fixed-data evaluation. Absence of an optional role cannot shrink the prediction domain.
