from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from projects.active.bridge.runtime.cli import load_esmc_model_cached
from projects.active.bridge.runtime.protein_embeddings import (
    VALID_AA,
    batched_mean_embeddings,
    build_length_batches,
    clean_sequence,
    collate,
    mean_embedding,
    save_vector,
    write_status,
)
from projects.active.bridge.model.assets import ROOT

INPUT = ROOT / "results/bridge_layered_v4_cage_gate_bridge/external_proteins.csv"
OUT = ROOT / "results/bridge_layered_v4_cage_gate_bridge/external_protein_esmc"
MEMBERSHIP = ROOT / "results/bridge_layered_v4_r2e_cage/pair_membership.csv.gz"
GENERAL_SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"


def build_external_candidates() -> tuple[pd.DataFrame, dict[str, int]]:
    membership = pd.read_csv(
        MEMBERSHIP,
        dtype=str,
        usecols=["route", "logical_candidate_id", "sequence"],
    ).fillna("")
    membership = (
        membership[membership.route.eq("enzymecage")]
        .drop_duplicates("logical_candidate_id", keep="first")
        .copy()
    )
    general = pd.read_csv(GENERAL_SEQUENCES, sep="\t", dtype=str).fillna("")
    registered_ids = set(general.protein_id.astype(str))
    registered_sequences = {
        clean_sequence(value)
        for value in general.sequence.astype(str)
        if clean_sequence(value)
    }
    membership["clean_sequence"] = membership.sequence.map(clean_sequence)
    direct = membership.logical_candidate_id.astype(str).isin(registered_ids)
    exact_sequence = membership.clean_sequence.isin(registered_sequences)
    external = membership[~direct & ~exact_sequence].copy()
    candidates = pd.DataFrame(
        {
            "Entry": external.logical_candidate_id.astype(str),
            "Sequence": external.sequence.astype(str),
        }
    ).drop_duplicates("Entry", keep="first").sort_values("Entry").reset_index(drop=True)
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(INPUT, index=False)
    audit = {
        "native_gate_unique_proteins": int(len(membership)),
        "registered_direct": int(direct.sum()),
        "exact_sequence_alias": int((~direct & exact_sequence).sum()),
        "open_world_external": int(len(candidates)),
    }
    return candidates, audit


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vector_dir = OUT / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    candidates, input_audit = build_external_candidates()
    candidates["Sequence"] = candidates.Sequence.map(clean_sequence)

    failures: list[dict[str, object]] = []
    completed = 0
    reused = 0
    pending_short: list[tuple[str, str]] = []
    pending_long: list[tuple[str, str]] = []

    for row in candidates.itertuples(index=False):
        entry = str(row.Entry)
        sequence = str(row.Sequence)
        vector_path = vector_dir / f"{entry}.npy"
        if vector_path.exists():
            reused += 1
            continue
        if not sequence or not VALID_AA.match(sequence):
            failures.append(
                {
                    "Entry": entry,
                    "length": len(sequence),
                    "error": "invalid_sequence",
                }
            )
            continue
        if len(sequence) <= 1000:
            pending_short.append((entry, sequence))
        else:
            pending_long.append((entry, sequence))

    model = None
    if pending_short or pending_long:
        model = load_esmc_model_cached("esmc_600m", "cuda")

    progress = tqdm(
        total=len(candidates),
        initial=reused + len(failures),
    )

    def checkpoint() -> None:
        write_status(
            OUT,
            len(candidates),
            completed,
            reused,
            failures,
        )

    def process_batch(batch: list[tuple[str, str]]) -> None:
        nonlocal completed
        if not batch:
            return
        try:
            assert model is not None
            vectors = batched_mean_embeddings(
                model,
                [sequence for _, sequence in batch],
                "cuda",
            )
            if len(vectors) != len(batch):
                raise ValueError("ESM-C batch output size mismatch")
            for (entry, _), vector in zip(batch, vectors, strict=True):
                save_vector(vector_dir, entry, vector)
                completed += 1
                progress.update(1)
        except Exception as exc:
            if len(batch) > 1:
                midpoint = len(batch) // 2
                process_batch(batch[:midpoint])
                process_batch(batch[midpoint:])
            else:
                entry, sequence = batch[0]
                failures.append(
                    {
                        "Entry": entry,
                        "length": len(sequence),
                        "error": repr(exc),
                    }
                )
                progress.update(1)
        if (completed + reused + len(failures)) % 25 == 0:
            checkpoint()

    for batch in build_length_batches(
        pending_short,
        max_batch_tokens=2048,
        max_batch_size=16,
    ):
        process_batch(batch)

    for entry, sequence in pending_long:
        try:
            assert model is not None
            vector = mean_embedding(
                model,
                sequence,
                max_residues=1000,
                overlap=100,
                device="cuda",
            )
            save_vector(vector_dir, entry, vector)
            completed += 1
        except Exception as exc:
            failures.append(
                {
                    "Entry": entry,
                    "length": len(sequence),
                    "error": repr(exc),
                }
            )
        progress.update(1)
        if (completed + reused + len(failures)) % 25 == 0:
            checkpoint()

    progress.close()
    checkpoint()
    n_collated, missing = collate(OUT, candidates)
    summary = {
        "schema": "bridge-layered-v4-cage-gate-external-broad",
        "status": "completed",
        "input_candidates": int(len(candidates)),
        "input_audit": input_audit,
        "reused": int(reused),
        "completed": int(completed),
        "failures": int(len(failures)),
        "collated": int(n_collated),
        "missing_after_collate": int(len(missing)),
        "loader": "repository local-cache ESM-C loader",
        "model": "esmc_600m",
        "max_residues": 1000,
        "overlap": 100,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
