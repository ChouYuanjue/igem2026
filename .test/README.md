# Portable workflow test data

This directory contains a deliberately tiny synthetic enzyme-reaction dataset
used only to test workflow contracts. Protein sequences, reaction SMILES and
feature vectors are fabricated for software validation; they are not
biological benchmark observations.

`config.yaml` exercises the portable workflow using tracked precomputed
features so CI does not require large foundation-model downloads.

`config_fresh.yaml` exercises fresh feature computation from the same source
tables with ESM-C and DRFP. Generated outputs are ignored by Git.
