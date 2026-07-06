# Data Schema

## Reaction Input CSV

Expected columns:

- `reaction_id`
- `substrate_smiles`
- `product_smiles`
- `reaction_smiles`
- optional metadata columns

EnzymeCAGE demo-style reaction files may instead contain `CANO_RXN_SMILES`.

## Enzyme Structure Manifest

Expected columns:

- `enzyme_id`
- `structure_path`
- `source`
- `species`
- optional `sequence_path`

## Enzyme-Reaction Pair CSV

Expected columns:

- `reaction_id` optional
- `enzyme_id` or `UniprotID`
- `structure_path` optional
- `sequence` when running EnzymeCAGE feature generation
- `CANO_RXN_SMILES` or equivalent reaction column
- optional `label`

## Pocket Manifest CSV

Fixed columns:

- `run_id`
- `enzyme_id`
- `structure_path`
- `pocket_method`
- `pocket_source`
- `pocket_rank`
- `pocket_global_id`
- `pocket_score`
- `pocket_center_x`
- `pocket_center_y`
- `pocket_center_z`
- `pocket_residues`
- `pocket_pdb_path`
- `source_raw_dir`
- `pocket_pdb_mode`

`pocket_global_id` format:

```text
{enzyme_id}__{pocket_source}__rank{pocket_rank}
```

`pocket_pdb_mode` values:

- `cropped_pocket`
- `full_structure_placeholder`
- `external_original`

## Pocket-Level Prediction CSV

Expected normalized columns:

- optional `reaction_id`
- `enzyme_id`
- `pocket_global_id`
- `pocket_source`
- `pocket_rank`
- `pocket_score_original`
- `cage_score`

Accepted raw score aliases include `score`, `pred`, `prediction`,
`catalytic_score`, and `y_pred`.

## Enzyme-Level Prediction CSV

Expected columns:

- optional `reaction_id`
- `enzyme_id`
- `aggregated_score`
- `aggregation_method`
- `n_pockets`
- `best_pocket_global_id`
- `best_pocket_source`
- `best_pocket_rank`
- `best_pocket_cage_score`
- `used_fallback`
- optional `fallback_reason`

## Baseline Matrix Config

Each generated config contains:

- `project`: `name`, `run_id`, `seed`, baseline metadata
- `external`: `enzymecage_root`, `p2rank_home`, `fpocket_bin`
- `data`: `data_mode`, `reaction_csv`, `structure_dir`, `working_dir`, `output_dir`
- `model`: checkpoint directory and filename
- `pocket`: method, source, selection, top-k fields, union sources
- `aggregation`: method, optional temperature, optional source weights
- `evaluation`: label CSV and top-k values

Supported `data_mode` values:

- `demo_mining`: README-style `dataset/demo` mining route. This is blocked when
  `dataset/demo` or its matching checkpoint is absent.
- `official_eval`: direct `python infer.py --config <official config>` from the
  EnzymeCAGE root. This uses official configs and real checkpoint names such as
  `epoch_19.pth`.
- `derived_smallset`: sampled official evaluation rows plus mapped full
  structures, then normal pocket extraction and aggregation. If only
  pre-extracted pocket PDBs are available, this mode is blocked with
  `blocked_missing_full_structure_for_p2rank`.

## Official Smallset Files

`build_official_smallset.py` writes:

- `smallset_pairs.csv`: normalized candidate rows with `reaction_id`,
  `enzyme_id`, `UniprotID`, optional `sequence`, `structure_path`,
  `CANO_RXN_SMILES`, and optional labels.
- `structure_link_report.csv`: per-enzyme mapping status, including whether a
  full structure, only a pre-extracted pocket, or no structure was found.
- `smallset_summary.json`: field detection, sampled counts, blocked reason, and
  generated files.
