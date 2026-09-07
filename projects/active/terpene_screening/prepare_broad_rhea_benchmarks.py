from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = ROOT / "data/external/reactzyme/cleaned_uniprot_rhea.tsv"
DEFAULT_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"
DEFAULT_BASE_TRAIN = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu/training_pairs.csv"
DEFAULT_OUTPUT = ROOT / "results/broad_rhea_fair_benchmarks_v1"

OFFICIAL_SPLITS = {
    "time": (
        ROOT / "data/external/reactzyme/time_split/positive_train_val_time.pt",
        ROOT / "data/external/reactzyme/time_split/positive_test_time.pt",
    ),
    "reaction": (
        ROOT / "data/external/reactzyme/reaction_smi_split/positive_train_val_mol_smi.pt",
        ROOT / "data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt",
    ),
    "enzyme": (
        ROOT / "data/external/reactzyme/enzyme_smi_split/positive_train_val_seq_smi.pt",
        ROOT / "data/external/reactzyme/enzyme_smi_split/positive_test_seq_smi.pt",
    ),
}


def clean_sequence(value: object) -> str:
    return "".join(str(value or "").split()).upper()


def stable_bucket(*values: object, modulo: int = 10) -> int:
    text = "\x1f".join(str(value) for value in values)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


def load_official_sequence_membership(path: Path) -> set[str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {clean_sequence(value[1]) for value in payload.values()}


def load_sequence_to_canonical(universe: Path) -> dict[str, str]:
    frame = pd.read_csv(universe / "protein_sequences.tsv", sep="\t", dtype=str).fillna("")
    mapping: dict[str, str] = {}
    for protein_id, sequence in frame[["protein_id", "sequence"]].itertuples(index=False):
        sequence = clean_sequence(sequence)
        if sequence:
            mapping.setdefault(sequence, str(protein_id))
    return mapping


def load_alias_to_canonical(universe: Path) -> dict[str, str]:
    frame = pd.read_csv(universe / "protein_metadata.csv", dtype=str).fillna("")
    mapping: dict[str, str] = {}
    for record in frame.to_dict("records"):
        canonical = str(record["protein_id"])
        aliases = [canonical, str(record.get("canonical_accession", ""))]
        aliases.extend(str(record.get("aliases", "")).split(";"))
        for alias in aliases:
            alias = alias.strip()
            if alias:
                mapping.setdefault(alias, canonical)
    return mapping


def load_rhea_pairs(raw_path: Path, universe: Path) -> pd.DataFrame:
    sequence_to_canonical = load_sequence_to_canonical(universe)
    reaction_ids = set(pd.read_csv(universe / "reactions.csv", dtype=str)["reaction_id"].astype(str))
    raw = pd.read_csv(
        raw_path,
        sep="\t",
        usecols=["Rhea ID", "Date of creation", "Sequence"],
        dtype=str,
    ).fillna("")
    rows: list[dict[str, object]] = []
    for rhea_text, date_text, sequence_raw in raw.itertuples(index=False):
        sequence = clean_sequence(sequence_raw)
        protein_id = sequence_to_canonical.get(sequence)
        if protein_id is None:
            continue
        try:
            date = int(str(date_text).strip())
        except ValueError:
            date = 0
        for reaction_id in str(rhea_text).split(";"):
            reaction_id = reaction_id.strip()
            if reaction_id in reaction_ids:
                rows.append(
                    {
                        "protein_id": protein_id,
                        "reaction_id": reaction_id,
                        "sequence": sequence,
                        "creation_date": date,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["protein_id", "reaction_id", "creation_date"])
        .drop_duplicates(["protein_id", "reaction_id"], keep="first")
        .reset_index(drop=True)
    )


def load_base_exposure(base_train: Path, universe: Path) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    aliases = load_alias_to_canonical(universe)
    frame = pd.read_csv(base_train, dtype=str).fillna("")
    proteins: set[str] = set()
    reactions: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for entry, reaction_id in frame[["Entry", "rhea_id"]].itertuples(index=False):
        canonical = aliases.get(str(entry).strip())
        reaction_id = str(reaction_id).strip()
        if canonical:
            proteins.add(canonical)
        if reaction_id.startswith("RHEA:"):
            reactions.add(reaction_id)
        if canonical and reaction_id.startswith("RHEA:"):
            pairs.add((canonical, reaction_id))
    return proteins, reactions, pairs


def pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[["protein_id", "reaction_id"]].drop_duplicates().sort_values(["protein_id", "reaction_id"]).reset_index(drop=True)


def write_cell(
    output: Path,
    *,
    name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_proteins: set[str],
    base_reactions: set[str],
    base_pairs: set[tuple[str, str]],
    expected: dict[str, bool],
    source_protocol: str,
    claim_tier: str,
) -> dict[str, object]:
    train_pairs = pair_frame(train)
    test_pairs = pair_frame(test)
    train_pair_set = set(map(tuple, train_pairs[["protein_id", "reaction_id"]].itertuples(index=False, name=None)))
    test_pair_set = set(map(tuple, test_pairs[["protein_id", "reaction_id"]].itertuples(index=False, name=None)))
    train_proteins = set(train_pairs["protein_id"])
    train_reactions = set(train_pairs["reaction_id"])
    test_proteins = set(test_pairs["protein_id"])
    test_reactions = set(test_pairs["reaction_id"])

    exact_overlap = test_pair_set & train_pair_set
    base_pair_overlap = test_pair_set & base_pairs
    base_protein_overlap = test_proteins & base_proteins
    base_reaction_overlap = test_reactions & base_reactions
    audit = {
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_proteins": len(train_proteins),
        "test_proteins": len(test_proteins),
        "train_reactions": len(train_reactions),
        "test_reactions": len(test_reactions),
        "exact_train_test_pair_overlap": len(exact_overlap),
        "test_protein_seen_fraction": float(test_pairs["protein_id"].isin(train_proteins).mean()) if len(test_pairs) else 0.0,
        "test_reaction_seen_fraction": float(test_pairs["reaction_id"].isin(train_reactions).mean()) if len(test_pairs) else 0.0,
        "base_checkpoint_exact_pair_overlap": len(base_pair_overlap),
        "base_checkpoint_test_protein_overlap": len(base_protein_overlap),
        "base_checkpoint_test_reaction_overlap": len(base_reaction_overlap),
    }
    violations: list[str] = []
    if exact_overlap:
        violations.append("exact_train_test_pair_overlap")
    if expected.get("protein_unseen") and audit["test_protein_seen_fraction"] != 0.0:
        violations.append("protein_unseen")
    if expected.get("reaction_unseen") and audit["test_reaction_seen_fraction"] != 0.0:
        violations.append("reaction_unseen")
    if expected.get("base_protein_unseen") and base_protein_overlap:
        violations.append("base_checkpoint_test_protein_overlap")
    if expected.get("base_reaction_unseen") and base_reaction_overlap:
        violations.append("base_checkpoint_test_reaction_overlap")
    if base_pair_overlap:
        violations.append("base_checkpoint_exact_pair_overlap")

    cell = output / name
    cell.mkdir(parents=True, exist_ok=True)
    train_pairs.to_csv(cell / "train_pairs.csv", index=False)
    test_pairs.to_csv(cell / "test_pairs.csv", index=False)
    metadata = {
        "name": name,
        "source_protocol": source_protocol,
        "claim_tier": claim_tier,
        "expected": expected,
        "audit": audit,
        "valid": not violations and len(test_pairs) > 0,
        "violations": violations,
    }
    (cell / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build broad leakage-aware Rhea-level benchmark cells from ReactZyme/UniProt memberships."
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--base-train", type=Path, default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temporal-cutoff-year", type=int, default=2020)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = load_rhea_pairs(args.raw.resolve(), args.universe.resolve())
    base_proteins, base_reactions, base_pairs = load_base_exposure(
        args.base_train.resolve(), args.universe.resolve()
    )
    manifests: list[dict[str, object]] = []

    # Project ReactZyme's author-provided sequence memberships onto the underlying
    # UniProt-Rhea graph. These are explicitly *not* called ReactZyme-native scores:
    # native ReactZyme reaction inputs are molecule bags, whereas Catalyst ranks
    # individual Rhea reactions. The memberships are useful split definitions only.
    memberships: dict[str, tuple[set[str], set[str]]] = {}
    for split, (train_path, test_path) in OFFICIAL_SPLITS.items():
        memberships[split] = (
            load_official_sequence_membership(train_path),
            load_official_sequence_membership(test_path),
        )

    for split in ("time", "enzyme"):
        train_sequences, test_sequences = memberships[split]
        train = pairs[pairs["sequence"].isin(train_sequences)].copy()
        train_reactions = set(train["reaction_id"])
        test = pairs[
            pairs["sequence"].isin(test_sequences)
            & ~pairs["sequence"].isin(train_sequences)
            & pairs["reaction_id"].isin(train_reactions)
            & ~pairs["protein_id"].isin(base_proteins)
        ].copy()
        test = test[
            [pair not in base_pairs for pair in zip(test["protein_id"], test["reaction_id"])]
        ]
        manifests.append(
            write_cell(
                output,
                name=f"reactzyme_{split}_projected_protein_cold",
                train=train,
                test=test,
                base_proteins=base_proteins,
                base_reactions=base_reactions,
                base_pairs=base_pairs,
                expected={"protein_unseen": True, "base_protein_unseen": True},
                source_protocol=f"ReactZyme {split} sequence membership projected to individual Rhea associations",
                claim_tier="core_generalization",
            )
        )

    train_sequences, test_sequences = memberships["reaction"]
    train = pairs[pairs["sequence"].isin(train_sequences)].copy()
    train_reactions = set(train["reaction_id"])
    test = pairs[
        pairs["sequence"].isin(test_sequences)
        & ~pairs["sequence"].isin(train_sequences)
        & ~pairs["reaction_id"].isin(train_reactions)
        & ~pairs["protein_id"].isin(base_proteins)
        & ~pairs["reaction_id"].isin(base_reactions)
    ].copy()
    test = test[
        [pair not in base_pairs for pair in zip(test["protein_id"], test["reaction_id"])]
    ]
    manifests.append(
        write_cell(
            output,
            name="reactzyme_reaction_projected_double_cold",
            train=train,
            test=test,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            expected={
                "protein_unseen": True,
                "reaction_unseen": True,
                "base_protein_unseen": True,
                "base_reaction_unseen": True,
            },
            source_protocol="ReactZyme reaction-similarity membership projected to individual Rhea associations",
            claim_tier="core_generalization",
        )
    )

    cutoff = args.temporal_cutoff_year * 10000 + 1231
    temporal_train = pairs[pairs["creation_date"].between(1, cutoff)].copy()
    temporal_test_all = pairs[pairs["creation_date"] > cutoff].copy()
    temporal_train_proteins = set(temporal_train["protein_id"])
    temporal_train_reactions = set(temporal_train["reaction_id"])

    temporal_protein = temporal_test_all[
        ~temporal_test_all["protein_id"].isin(temporal_train_proteins)
        & temporal_test_all["reaction_id"].isin(temporal_train_reactions)
        & ~temporal_test_all["protein_id"].isin(base_proteins)
    ].copy()
    temporal_protein = temporal_protein[
        [pair not in base_pairs for pair in zip(temporal_protein["protein_id"], temporal_protein["reaction_id"])]
    ]
    manifests.append(
        write_cell(
            output,
            name=f"temporal_post{args.temporal_cutoff_year}_protein_cold",
            train=temporal_train,
            test=temporal_protein,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            expected={"protein_unseen": True, "base_protein_unseen": True},
            source_protocol=f"UniProt creation date <= {args.temporal_cutoff_year} train, later proteins test; reaction seen",
            claim_tier="core_generalization",
        )
    )

    temporal_double = temporal_test_all[
        ~temporal_test_all["protein_id"].isin(temporal_train_proteins)
        & ~temporal_test_all["reaction_id"].isin(temporal_train_reactions)
        & ~temporal_test_all["protein_id"].isin(base_proteins)
        & ~temporal_test_all["reaction_id"].isin(base_reactions)
    ].copy()
    temporal_double = temporal_double[
        [pair not in base_pairs for pair in zip(temporal_double["protein_id"], temporal_double["reaction_id"])]
    ]
    manifests.append(
        write_cell(
            output,
            name=f"temporal_post{args.temporal_cutoff_year}_double_cold",
            train=temporal_train,
            test=temporal_double,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            expected={
                "protein_unseen": True,
                "reaction_unseen": True,
                "base_protein_unseen": True,
                "base_reaction_unseen": True,
            },
            source_protocol=f"UniProt creation date <= {args.temporal_cutoff_year} train, later protein + unseen reaction test",
            claim_tier="core_generalization",
        )
    )

    # A deterministic reaction holdout gives a complementary, broad reaction-cold
    # regime with proteins retained when possible. It is weaker than similarity-cold
    # and therefore labelled secondary rather than core.
    held_reactions = {
        reaction_id
        for reaction_id in pairs["reaction_id"].unique()
        if stable_bucket(reaction_id, modulo=10) == 0 and reaction_id not in base_reactions
    }
    reaction_train = pairs[~pairs["reaction_id"].isin(held_reactions)].copy()
    reaction_train_proteins = set(reaction_train["protein_id"])
    reaction_test = pairs[
        pairs["reaction_id"].isin(held_reactions)
        & pairs["protein_id"].isin(reaction_train_proteins)
    ].copy()
    reaction_test = reaction_test[
        [pair not in base_pairs for pair in zip(reaction_test["protein_id"], reaction_test["reaction_id"])]
    ]
    manifests.append(
        write_cell(
            output,
            name="broad_reaction_hash_cold_protein_seen",
            train=reaction_train,
            test=reaction_test,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            expected={"reaction_unseen": True, "base_reaction_unseen": True},
            source_protocol="Deterministic 10% Rhea-ID holdout; keep test proteins seen in training",
            claim_tier="secondary_generalization",
        )
    )

    # Random-pair holdout is deliberately only a sanity/edge-completion regime.
    pair_holdout = pairs[
        [stable_bucket(p, r, modulo=10) == 0 for p, r in zip(pairs["protein_id"], pairs["reaction_id"])]
    ].copy()
    pair_train = pairs.drop(pair_holdout.index).copy()
    train_p = set(pair_train["protein_id"])
    train_r = set(pair_train["reaction_id"])
    pair_test = pair_holdout[
        pair_holdout["protein_id"].isin(train_p)
        & pair_holdout["reaction_id"].isin(train_r)
    ].copy()
    pair_test = pair_test[
        [pair not in base_pairs for pair in zip(pair_test["protein_id"], pair_test["reaction_id"])]
    ]
    manifests.append(
        write_cell(
            output,
            name="broad_pair_hash_holdout_both_seen",
            train=pair_train,
            test=pair_test,
            base_proteins=base_proteins,
            base_reactions=base_reactions,
            base_pairs=base_pairs,
            expected={},
            source_protocol="Deterministic 10% exact-pair holdout with both entities retained in train",
            claim_tier="sanity_generalization",
        )
    )

    summary = {
        "version": "broad-rhea-fair-benchmarks-v1",
        "pair_source": str(args.raw.resolve()),
        "candidate_universe": str(args.universe.resolve()),
        "rollback_base_training": str(args.base_train.resolve()),
        "pair_count": int(len(pairs)),
        "protein_count": int(pairs["protein_id"].nunique()),
        "reaction_count": int(pairs["reaction_id"].nunique()),
        "cells": manifests,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
