# FIBRE interpretability contract

A FIBRE result explains which information supported or contextualized the predicted catalytic interaction.

For a returned pair, expose when available:

- the general interaction score and ranking;
- contributing interaction bases or model views;
- agreement and disagreement among views;
- which family / structure / mechanism charts contributed and their partition weights;
- known positive catalytic precedents and train-free observation updates;
- reaction-centre, sequence, structure, pocket, motif and cofactor observations;
- exact provenance for curated or literature-derived facts;
- assay context and whether it is pair-specific, protein-level or inferred;
- applicability or calibration diagnostics only in their validated scope.

Missing information is reported as missing. It is not converted into a negative score or a reason to refuse prediction.

A combined probability-like confidence is allowed only with a matching frozen calibration. Otherwise Starase reports support, evidence coverage and model agreement separately.

Historical fibre_relation, stratified-correspondence and geometric-uncertainty fields may remain in compatibility APIs while clients migrate, but they are not the current scientific interpretation of FIBRE.
