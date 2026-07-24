# Terpene synthase retrieval scenarios

The terpene branch is a bidirectional, inductive retrieval system rather than a closed-label classifier. It must support the following scenarios through one shared enzyme encoder, one shared reaction encoder, and a compatible scoring space.

## 1. Reaction-to-enzyme retrieval

### R2E few-shot: known reaction with known catalysts

Input:
- one reaction;
- one or more enzymes already known to catalyze it;
- an enzyme candidate library.

Goal:
- retrieve additional enzymes that may catalyze the reaction;
- exclude the supplied known catalysts from the novelty ranking;
- report Top-3, Top-10 and Top-20 panels.

Primary evaluation:
- hide enzyme positives by MMseqs2 cluster rather than hiding random near-duplicates;
- mask supplied/known catalysts before ranking;
- report Hit@3/10/20, MRR, best-positive rank, positive recall and enrichment.

### R2E zero-shot: reaction with no known catalyst

Input:
- one reaction;
- an enzyme candidate library;
- no catalyst seed.

Goal:
- rank enzymes directly from the reaction representation;
- optionally use other annotated, chemically or mechanistically related reactions as retrieval evidence.

Primary evaluation:
- reaction-cluster-cold and double-cold splits;
- no positive enzyme from the query reaction may appear in training;
- report Hit@3/10/20, MRR, best-positive rank and coverage.

## 2. Enzyme-to-reaction retrieval

### E2R few-shot: known enzyme with known reactions

Input:
- one enzyme;
- one or more reactions already known for that enzyme;
- a reaction candidate library.

Goal:
- retrieve additional reactions or products that the enzyme may catalyze;
- exclude supplied/known reactions from the novelty ranking.

Primary evaluation:
- hide reaction positives by reaction mechanism/scaffold cluster;
- mask supplied/known reactions before ranking;
- report Hit@3/10/20, MRR, best-positive rank and positive recall.

### E2R zero-shot: enzyme with no known reaction

Input:
- one enzyme sequence, and optionally its predicted structure/domain information;
- a reaction candidate library;
- no reaction seed.

Goal:
- rank reactions directly from the enzyme representation.

Primary evaluation:
- protein-cluster-cold and double-cold splits;
- no positive reaction association for the query enzyme may appear in training.

## 3. Open-world library extension

The system must not require an entity to have been present when the model was trained.

### Add an external enzyme

Required behavior:
1. accept a new amino-acid sequence and a temporary identifier;
2. compute its sequence embedding, and later optional structure/domain features;
3. append it to the enzyme index without retraining the model;
4. allow it to participate in reaction-to-enzyme rankings;
5. allow it to act as an enzyme query for enzyme-to-reaction ranking.

Acceptance test:
- hold out complete MMseqs2 clusters during training;
- encode the held-out enzymes only at inference time;
- measure whether relevant held-out enzymes appear in Top-3/10/20.

### Add an external reaction

Required behavior:
1. accept a reaction SMILES or structured substrate/product description and a temporary identifier;
2. compute DRFP/RXNFP, precursor, product-scaffold and later mechanism features;
3. append it to the reaction index without retraining the model;
4. allow it to act as a reaction query for reaction-to-enzyme ranking;
5. allow it to participate as a candidate in enzyme-to-reaction ranking.

Acceptance test:
- hold out complete reaction scaffold/mechanism clusters during training;
- encode held-out reactions only at inference time;
- measure whether relevant reactions appear in Top-3/10/20.

## 4. Architectural consequences

The following are mandatory consequences of the scenarios above:

- Do not implement the final system as a 513-class Rhea classifier or a 1,391-class enzyme classifier.
- Enzyme and reaction encoders must be independently callable.
- Candidate indices must support append/update without model retraining.
- Ranking APIs must support masking supplied known positives.
- Evaluation must be bidirectional and include entity-cold and double-cold protocols.
- Random positive hiding may be retained only as a diagnostic upper bound, not as the main result.
- The final report must distinguish database coverage failure, encoder generalization failure and ranking failure.

## 5. Planned unified API

```text
encode_enzyme(sequence, optional_structure) -> enzyme_vector
encode_reaction(reaction_smiles, optional_mechanism) -> reaction_vector
add_enzyme(temp_id, enzyme_vector)
add_reaction(temp_id, reaction_vector)
rank_enzymes(reaction_vector, known_enzyme_ids=[], top_k=20)
rank_reactions(enzyme_vector, known_reaction_ids=[], top_k=20)
```

This document records the target behavior. Current baseline work on MMseqs2 clustering, ESM-C retrieval, cold splits and dual-tower training remains unchanged and supplies the first implementation of this interface.
