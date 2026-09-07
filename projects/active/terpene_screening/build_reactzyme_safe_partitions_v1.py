from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "projects/active/terpene_screening/REACTZYME_NATIVE_SUPPORT_ADAPTATION_V1.json"
DEFAULT_SOURCE = ROOT / "results/enzymecage_cleanroom_rdkitplus_v1"
DEFAULT_OUTPUT = ROOT / "results/reactzyme_native_support_adaptation_v1/partitions"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_seq(value: object) -> str:
    return "".join(str(value).split()).upper().rstrip("*")


def norm_bag(value: object) -> str:
    return ".".join(sorted(part for part in str(value).replace("*", "C").split(".") if part))


def split_rhea(value: object) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip().startswith("RHEA:")}


def build_protection_sets(root: Path = ROOT) -> tuple[set[str], set[str], dict[str, int]]:
    protein_frame = pd.read_csv(
        root / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv",
        sep="\t", dtype=str,
    ).fillna("")
    seq_to_ids: dict[str, set[str]] = defaultdict(set)
    for protein_id, sequence in protein_frame[["protein_id", "sequence"]].itertuples(index=False):
        seq_to_ids[norm_seq(sequence)].add(str(protein_id))

    enzyme = torch.load(
        root / "data/external/reactzyme/enzyme_smi_split/positive_test_seq_smi.pt",
        map_location="cpu", weights_only=False,
    )
    enzyme_sequences = {norm_seq(value[1]) for value in enzyme.values()}
    protected_proteins: set[str] = set()
    mapped_sequences = 0
    for sequence in enzyme_sequences:
        ids = seq_to_ids.get(sequence, set())
        if ids:
            mapped_sequences += 1
            protected_proteins.update(ids)

    reaction = torch.load(
        root / "data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt",
        map_location="cpu", weights_only=False,
    )
    reaction_bags = {norm_bag(value[0]) for value in reaction.values()}
    molecules = pd.read_csv(root / "data/external/reactzyme/uniprot_molecules.tsv", sep="\t", dtype=str).fillna("")
    matched_entries = set(
        molecules.loc[molecules["molecules"].map(norm_bag).isin(reaction_bags), "uniprot_id"].astype(str)
    )
    uniprot_rhea = pd.read_csv(root / "data/external/reactzyme/uniprot_rhea.tsv", sep="\t", dtype=str).fillna("")
    protected_rhea: set[str] = set()
    for value in uniprot_rhea.loc[uniprot_rhea["Entry"].isin(matched_entries), "Rhea ID"]:
        protected_rhea.update(split_rhea(value))

    clean = pd.read_csv(
        root / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv",
        dtype=str,
    ).fillna("")
    audit = {
        "enzyme_split_pairs": len(enzyme),
        "enzyme_unique_sequences": len(enzyme_sequences),
        "enzyme_mapped_sequences": mapped_sequences,
        "enzyme_protected_protein_ids": len(protected_proteins),
        "reaction_split_pairs": len(reaction),
        "reaction_unique_molecule_bags": len(reaction_bags),
        "reaction_matched_uniprot_entries": len(matched_entries),
        "reaction_protected_rhea_ids": len(protected_rhea),
        "reaction_protected_rhea_ids_present_in_clean2023": len(protected_rhea & set(clean["reaction_id"])),
    }
    return protected_proteins, protected_rhea, audit


def filter_training(frame: pd.DataFrame, policy: str, protected_proteins: set[str], protected_rhea: set[str]) -> pd.DataFrame:
    protein_mask = frame["protein_id"].astype(str).isin(protected_proteins)
    if policy == "enzyme_safe":
        keep = ~protein_mask
    elif policy == "union_safe_max":
        reaction_mask = frame["reaction_id"].astype(str).isin(protected_rhea)
        keep = ~(protein_mask | reaction_mask)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return frame.loc[keep, ["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    expected_hashes = protocol["source_hashes"]
    source_paths = {
        "reactzyme_enzyme_test": ROOT / "data/external/reactzyme/enzyme_smi_split/positive_test_seq_smi.pt",
        "reactzyme_reaction_test": ROOT / "data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt",
        "reactzyme_uniprot_molecules": ROOT / "data/external/reactzyme/uniprot_molecules.tsv",
        "reactzyme_uniprot_rhea": ROOT / "data/external/reactzyme/uniprot_rhea.tsv",
        "reactzyme_cleaned_uniprot_rhea": ROOT / "data/external/reactzyme/cleaned_uniprot_rhea.tsv",
        "general_protein_sequences": ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv",
        "clean2023_pairs": ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv",
    }
    for key, path in source_paths.items():
        actual = sha256_file(path)
        if actual != expected_hashes[key]:
            raise RuntimeError(f"Source hash mismatch for {key}: {actual} != {expected_hashes[key]}")

    protected_proteins, protected_rhea, audit = build_protection_sets(ROOT)
    expected_audit = protocol["expected_static_audit"]
    for key, value in audit.items():
        if int(value) != int(expected_audit[key]):
            raise RuntimeError(f"Static audit mismatch for {key}: {value} != {expected_audit[key]}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "protected_protein_ids.txt").write_text("\n".join(sorted(protected_proteins)) + "\n", encoding="utf-8")
    (args.output_root / "protected_rhea_ids.txt").write_text("\n".join(sorted(protected_rhea)) + "\n", encoding="utf-8")

    manifests = []
    for fold in (0, 1, 2):
        fold_spec = protocol["fixed_folds"][str(fold)]
        source_train = args.source_root / f"fold{fold}/training_pairs.csv"
        source_dev = args.source_root / f"fold{fold}/dev_pairs.csv"
        if sha256_file(source_train) != fold_spec["source_train_sha256"]:
            raise RuntimeError(f"fold{fold} source train hash mismatch")
        if sha256_file(source_dev) != fold_spec["source_dev_sha256"]:
            raise RuntimeError(f"fold{fold} source dev hash mismatch")
        train = pd.read_csv(source_train, dtype=str).fillna("")[["protein_id", "reaction_id"]]
        if len(train) != int(fold_spec["source_train_pairs"]):
            raise RuntimeError(f"fold{fold} source train row mismatch")
        for policy in ("enzyme_safe", "union_safe_max"):
            output = args.output_root / policy / f"fold{fold}"
            output.mkdir(parents=True, exist_ok=True)
            filtered = filter_training(train, policy, protected_proteins, protected_rhea)
            expected_rows = int(fold_spec[f"{policy}_train_pairs"])
            if len(filtered) != expected_rows:
                raise RuntimeError(f"{policy} fold{fold} row mismatch: {len(filtered)} != {expected_rows}")
            if filtered["protein_id"].isin(protected_proteins).any():
                raise RuntimeError(f"{policy} fold{fold} retains protected protein")
            if policy == "union_safe_max" and filtered["reaction_id"].isin(protected_rhea).any():
                raise RuntimeError(f"{policy} fold{fold} retains protected reaction")
            filtered.to_csv(output / "training_pairs.csv", index=False)
            shutil.copyfile(source_dev, output / "dev_pairs.csv")
            if sha256_file(output / "dev_pairs.csv") != fold_spec["source_dev_sha256"]:
                raise RuntimeError(f"{policy} fold{fold} dev bytes changed")
            dev = pd.read_csv(output / "dev_pairs.csv", dtype=str).fillna("")
            if set(filtered.protein_id) & set(dev.protein_id):
                raise RuntimeError(f"{policy} fold{fold} train/dev protein overlap")
            if set(filtered.reaction_id) & set(dev.reaction_id):
                raise RuntimeError(f"{policy} fold{fold} train/dev reaction overlap")
            manifest = {
                "policy": policy,
                "fold": fold,
                "source_training_pairs": str(source_train.resolve()),
                "source_training_sha256": fold_spec["source_train_sha256"],
                "source_dev_pairs": str(source_dev.resolve()),
                "source_dev_sha256": fold_spec["source_dev_sha256"],
                "output_training_sha256": sha256_file(output / "training_pairs.csv"),
                "output_dev_sha256": sha256_file(output / "dev_pairs.csv"),
                "source_train_rows": len(train),
                "retained_train_rows": len(filtered),
                "retained_fraction": len(filtered) / len(train),
                "dev_rows": len(dev),
                "protected_protein_ids": len(protected_proteins),
                "protected_rhea_ids": len(protected_rhea),
                "external_metrics_read": False,
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            manifests.append(manifest)
    payload = {"status": "materialized_from_frozen_protocol", "static_audit": audit, "partitions": manifests}
    (args.output_root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
