# TPS-specialized retrieval strategy

## Decision

Large general enzyme–reaction pretraining (ReactZyme, CLIPZyme, MAERM-like assets) is **not** the production mainline. It may be used only as a frozen prior, teacher, candidate generator, or ablation. Promotion requires leakage-controlled gains on TPS development folds and no regression on the frozen strict double-cold cells.

The production research path is TPS-specialized, hierarchical, and mechanism-aware.

## Why generic pretraining is secondary

The deployed task is not broad enzyme annotation. The candidate universe is already enriched for terpene synthases, and the difficult distinctions are among close TPS architectures and homologs with different product skeletons. Generic enzyme–reaction corpora are dominated by unrelated enzyme families and broad reaction-class signals. They may improve coarse compatibility while failing to resolve the local steric/electrostatic determinants of TPS cyclization outcomes.

## Mainline architecture

1. **Hard architecture gate**
   - class I, class II, bifunctional class I/II, OSC/triterpene, microbial-type, prenyltransferase-like/query-outside;
   - enforce complete-domain contracts and substrate carbon-count compatibility before learned ranking.

2. **Mechanistic reaction representation**
   - precursor and carbon count;
   - ionization/protonation initiation mode;
   - first cyclization topology;
   - ring count and product skeleton;
   - rearrangement and stereochemical descriptors when recoverable.

3. **TPS-local protein representation**
   - motif-relative residue windows around DDxxD, NSE/DTE, DxDD and architecture-specific catalytic regions;
   - predicted pocket residues and pocket geometry when structures are available;
   - full-sequence PLM embeddings retained only as global context.

4. **Family-specialized experts**
   - separate experts for plant class I, microbial class I, class II/bifunctional diterpene, and OSC/triterpene architectures;
   - train on within-family hard negatives: same substrate class and architecture, similar sequence, different product skeleton.

5. **Two-stage retrieval**
   - cheap global/phylogenetic scorer retrieves Top-50 or Top-100;
   - a small residue–atom or pocket–reaction cross-attention reranker scores only this shortlist.

6. **Positive-unlabeled and abstention policy**
   - unlabeled pairs are not treated as trustworthy negatives;
   - use group-aware PU masking, uncertainty calibration, and canonical-only fallback when evidence is weak.

## Evaluation contract

- nested exact-reaction validation for the current 1,391-protein library;
- strict protein-cluster × reaction-cluster double-cold evaluation;
- additional leave-one-product-skeleton / leave-one-architecture checks where sample size permits;
- development folds choose all hyperparameters; frozen cells are read once;
- wet-lab hit rate and diversity are the final promotion criteria.

## Role of external general models

A generic model is promoted only if one of the following is demonstrated without leakage:

- improves a TPS-specialized expert as a frozen feature or teacher;
- rescues queries outside the characterized TPS manifold;
- adds unique strict-double-cold hits without degrading calibrated early precision;
- reduces sample complexity when fine-tuning a family-specific expert.

Otherwise it remains an archived experiment.
