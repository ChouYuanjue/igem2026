# Pocket Model Survey

This survey is a lightweight map of methods relevant to pocket hypotheses and
residue evidence. It is not a claim that all methods are immediately compatible
with EnzymeCAGE.

## AlphaFill

- Input: protein structure, homologous templates, ligand and cofactor evidence
- Output: transferred ligands/cofactors and local binding context
- Training paradigm: template or homology-based transfer, not a standard
  supervised pocket predictor
- Requires template: yes
- Confidence: depends on template evidence and alignment context
- EnzymeCAGE integration: high conceptual relevance; this is the primary route
  described by EnzymeCAGE

## P2Rank

- Input: protein structure
- Output: ranked ligand-binding pockets, surface points, centers, scores, and
  residue sets when available
- Training paradigm: machine-learning surface point scoring and clustering
- Requires template: no
- Confidence: yes, score/probability-style pocket ranking
- EnzymeCAGE integration: practical; useful as fallback and top-k hypothesis
  source

## fpocket

- Input: protein structure
- Output: geometry-derived pockets and descriptors
- Training paradigm: geometry-based method using Voronoi tessellation and alpha
  spheres
- Requires template: no
- Confidence: pocket descriptors and scores, method-specific
- EnzymeCAGE integration: moderate; needs adapter to convert pockets into the
  expected pocket input

## DeepSurf

- Input: protein surface representation
- Output: ligand-binding site predictions
- Training paradigm: surface-based deep learning
- Requires template: no
- Confidence: model scores
- EnzymeCAGE integration: moderate; output needs residue or pocket conversion

## MaSIF

- Input: molecular surface patches
- Output: molecular surface interaction fingerprints and local descriptors
- Training paradigm: geometric deep learning over geodesic surface patches
- Requires template: no
- Confidence: task-dependent scores
- EnzymeCAGE integration: exploratory; useful for surface descriptors but not a
  direct pocket replacement without adapter work

## ScanNet

- Input: protein structure and sequence-derived residue context
- Output: residue-level binding site predictions
- Training paradigm: geometric deep learning
- Requires template: no
- Confidence: residue-level prediction scores
- EnzymeCAGE integration: useful as a residue prior or reranking signal

## ReactZyme

- Input: enzyme-reaction data, depending on benchmark/task setup
- Output: enzyme-reaction prediction or benchmark labels
- Training paradigm: enzyme-reaction benchmark and modeling, not pocket-centered
- Requires template: no
- Confidence: model-dependent
- EnzymeCAGE integration: useful broader baseline, not a direct pocket adapter

## GENzyme

- Input: reaction-conditioned enzyme or pocket generation context
- Output: generated enzyme/pocket candidates depending on task setup
- Training paradigm: generative modeling
- Requires template: task-dependent
- Confidence: model-dependent
- EnzymeCAGE integration: useful contrast to retrieval; not a phase-1 dependency
