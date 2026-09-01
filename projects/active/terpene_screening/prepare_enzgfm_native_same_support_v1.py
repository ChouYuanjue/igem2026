from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import (
    PREPARED, PROTEIN_FEATURE_ROOT, PROTEIN_SEQUENCE_TSV, SELECTION_SALT, TEST_POS, TRAIN_POS,
    bag_feature, normalize_reaction_bag, normalize_sequence, sha256_file, split_is_dev, support_priority,
)


def load_positive(path: Path) -> pd.DataFrame:
    data = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for source_key, value in data.items():
        if len(value) < 2:
            raise ValueError(f"unexpected ReactZyme tuple at {source_key}")
        rows.append({
            "source_key": str(source_key),
            "reaction_bag": normalize_reaction_bag(value[0]),
            "sequence": normalize_sequence(value[1]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    PREPARED.mkdir(parents=True, exist_ok=True)
    train_raw = load_positive(TRAIN_POS)
    test = load_positive(TEST_POS)

    seq_table = pd.read_csv(PROTEIN_SEQUENCE_TSV, sep="\t", dtype=str).fillna("")
    entries = pd.read_csv(PROTEIN_FEATURE_ROOT / "entries.csv", dtype=str)
    if len(seq_table) != len(entries):
        raise AssertionError("protein sequence / feature row count mismatch")
    if list(seq_table.protein_id.astype(str)) != list(entries.Entry.astype(str)):
        raise AssertionError("protein sequence / EnzGFM feature order mismatch")
    seq_to_rows: dict[str, list[int]] = defaultdict(list)
    for i, seq in enumerate(seq_table.sequence.astype(str)):
        seq_to_rows[normalize_sequence(seq)].append(i)

    all_bags = sorted(set(train_raw.reaction_bag) | set(test.reaction_bag))
    bag_to_idx = {b: i for i, b in enumerate(all_bags)}
    feats, audit = [], []
    for i, bag in enumerate(all_bags):
        v, valid, invalid = bag_feature(bag)
        feats.append(v)
        audit.append({"reaction_idx": i, "reaction_bag": bag, "valid_molecules": valid, "invalid_molecules": invalid, "zero_feature": valid == 0})
    np.save(PREPARED / "reaction_features.npy", np.stack(feats).astype(np.float32))
    pd.DataFrame(audit).to_csv(PREPARED / "reaction_entries.csv", index=False)

    def map_rows(frame: pd.DataFrame, split_name: str, allow_missing: bool) -> tuple[pd.DataFrame, list[str]]:
        out, missing = [], []
        for x in frame.itertuples(index=False):
            candidates = seq_to_rows.get(x.sequence, [])
            if not candidates:
                missing.append(x.sequence)
                if allow_missing:
                    continue
                raise AssertionError(f"unmapped {split_name} sequence")
            out.append({
                "source_key": x.source_key,
                "reaction_idx": bag_to_idx[x.reaction_bag],
                "protein_row": min(candidates),
                "sequence": x.sequence,
            })
        return pd.DataFrame(out), missing

    train_mapped, missing_train = map_rows(train_raw, "train", allow_missing=True)
    test_mapped, missing_test = map_rows(test, "test", allow_missing=False)
    train_mapped["is_dev"] = train_mapped.sequence.map(split_is_dev)
    train = train_mapped[~train_mapped.is_dev].drop(columns="is_dev").reset_index(drop=True)
    dev = train_mapped[train_mapped.is_dev].drop(columns="is_dev").reset_index(drop=True)
    all_train = train_mapped.drop(columns="is_dev").reset_index(drop=True)
    if set(train.sequence) & set(dev.sequence):
        raise AssertionError("development protein leakage")

    train.to_csv(PREPARED / "train_pairs.csv", index=False)
    dev.to_csv(PREPARED / "dev_pairs.csv", index=False)
    all_train.to_csv(PREPARED / "all_train_pairs.csv", index=False)
    test_mapped.to_csv(PREPARED / "test_pairs.csv", index=False)

    # Fixed bounded development scoring matrix selected by input IDs/hashes only, never outcomes.
    dev_proteins = dev[["sequence", "protein_row"]].drop_duplicates().copy()
    dev_proteins["priority"] = dev_proteins.sequence.map(lambda x: support_priority("protein", x))
    dev_proteins = dev_proteins.sort_values(["priority", "sequence"]).head(2048).drop(columns="priority")
    dev_reactions = dev[["reaction_idx"]].drop_duplicates().copy()
    rxn_entries = pd.DataFrame(audit)[["reaction_idx", "reaction_bag"]]
    dev_reactions = dev_reactions.merge(rxn_entries, on="reaction_idx", how="left")
    dev_reactions["priority"] = dev_reactions.reaction_bag.map(lambda x: support_priority("reaction", x))
    dev_reactions = dev_reactions.sort_values(["priority", "reaction_bag"]).head(2048).drop(columns="priority")
    dev_proteins.to_csv(PREPARED / "dev_eval_proteins.csv", index=False)
    dev_reactions.to_csv(PREPARED / "dev_eval_reactions.csv", index=False)

    test_proteins = test_mapped[["sequence", "protein_row"]].drop_duplicates().sort_values("sequence")
    test_reactions = test_mapped[["reaction_idx"]].drop_duplicates().merge(rxn_entries, on="reaction_idx", how="left").sort_values("reaction_bag")
    test_proteins.to_csv(PREPARED / "test_proteins.csv", index=False)
    test_reactions.to_csv(PREPARED / "test_reactions.csv", index=False)

    # Support checks only; no model scores or target-performance filtering.
    dev_p_set = set(dev_proteins.protein_row.astype(int))
    dev_r_set = set(dev_reactions.reaction_idx.astype(int))
    dev_eval_pairs = dev[dev.protein_row.astype(int).isin(dev_p_set) & dev.reaction_idx.astype(int).isin(dev_r_set)]
    e2r_q = dev_eval_pairs.protein_row.nunique()
    r2e_q = dev_eval_pairs.reaction_idx.nunique()
    if e2r_q < 200 or r2e_q < 200:
        raise AssertionError(f"internal selection support underpowered: e2r={e2r_q} r2e={r2e_q}")

    summary = {
        "status": "prepared_support_only_no_model_scores",
        "selection_salt": SELECTION_SALT,
        "train_source_rows": int(len(train_raw)),
        "retained_train_val_rows": int(len(all_train)),
        "missing_train_sequences": int(len(set(missing_train))),
        "train_rows": int(len(train)),
        "dev_rows": int(len(dev)),
        "train_unique_sequences": int(train.sequence.nunique()),
        "dev_unique_sequences": int(dev.sequence.nunique()),
        "train_dev_sequence_overlap": 0,
        "unique_reaction_bags_all_inputs": len(all_bags),
        "zero_reaction_features": int(sum(x["zero_feature"] for x in audit)),
        "invalid_molecule_tokens": int(sum(x["invalid_molecules"] for x in audit)),
        "dev_eval_proteins_selected_by_hash": int(len(dev_proteins)),
        "dev_eval_reactions_selected_by_hash": int(len(dev_reactions)),
        "dev_eval_e2r_queries_with_positive": int(e2r_q),
        "dev_eval_r2e_queries_with_positive": int(r2e_q),
        "native_test_rows": int(len(test_mapped)),
        "native_test_unique_sequences": int(test_mapped.sequence.nunique()),
        "native_test_unique_reaction_bags": int(test_mapped.reaction_idx.nunique()),
        "native_test_missing_sequences": len(missing_test),
        "test_labels_read_for_support_and_future_scoring_only": True,
        "test_performance_read": False,
        "test_performance_used_for_model_selection": False,
        "source_sha256": {
            "positive_train_val_seq_smi.pt": sha256_file(TRAIN_POS),
            "positive_test_seq_smi.pt": sha256_file(TEST_POS),
            "protein_sequences.tsv": sha256_file(PROTEIN_SEQUENCE_TSV),
            "protein_feature_entries.csv": sha256_file(PROTEIN_FEATURE_ROOT / "entries.csv"),
            "protein_feature_manifest.json": sha256_file(PROTEIN_FEATURE_ROOT / "manifest.json"),
        },
    }
    (PREPARED / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
