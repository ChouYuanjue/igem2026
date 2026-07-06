# EnzymeCAGE Intro Route

EnzymeCAGE is an enzyme-reaction retrieval and enzyme function prediction model.
It uses enzyme structure and reaction representation to produce a catalytic
compatibility score.

## Why Pocket-Level Modeling

Whole-protein structures contain large amounts of context that may be unrelated
to catalysis. Pocket-level modeling focuses the structural input on the local
environment most likely to interact with the reaction center, cofactor, or
substrate.

This choice reduces irrelevant protein context and makes the model's enzyme side
closer to catalytic evidence rather than global fold evidence alone.

## Pocket Route

The expected route is:

1. Use AlphaFill to transfer homologous ligand or cofactor context and identify
   a catalytic pocket when possible.
2. Use P2Rank as a fallback or demo ligand-binding pocket detector.
3. Convert the selected pocket into a residue graph.
4. Combine geometric pocket representation, pocket attention, reaction-aware
   interaction, ESM-C enzyme embedding, and DRFP reaction fingerprints.
5. Predict enzyme-reaction catalytic compatibility.

## Known Limitation

The active-site evidence is limited by the predicted pocket. If AlphaFill or a
predicted pocket misses catalytic residues, EnzymeCAGE may score the enzyme using
an incomplete or incorrect local structure.

## Our Question

We test pocket hypothesis robustness:

**How much do EnzymeCAGE ranking results change when the pocket source,
selection strategy, or aggregation strategy changes at inference time?**
