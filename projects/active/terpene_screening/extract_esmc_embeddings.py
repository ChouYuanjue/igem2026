from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "data/terpene_embeddings/esmc600m_mean"
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def chunks(sequence: str, max_residues: int, overlap: int) -> list[str]:
    if len(sequence) <= max_residues:
        return [sequence]
    step = max_residues - overlap
    return [sequence[start : start + max_residues] for start in range(0, len(sequence), step) if sequence[start:]]


def mean_embedding(model, sequence: str, max_residues: int, overlap: int, device: str) -> np.ndarray:
    from esm.sdk.api import ESMProtein, LogitsConfig

    weighted = None
    total_weight = 0
    for fragment in chunks(sequence, max_residues=max_residues, overlap=overlap):
        protein = ESMProtein(sequence=fragment)
        with torch.no_grad():
            encoded = model.encode(protein)
            output = model.logits(encoded, LogitsConfig(sequence=True, return_embeddings=True))
        embeddings = output.embeddings[0]
        if embeddings.shape[0] == len(fragment) + 2:
            embeddings = embeddings[1:-1]
        vector = embeddings.float().mean(dim=0)
        weight = len(fragment)
        weighted = vector * weight if weighted is None else weighted + vector * weight
        total_weight += weight
        del encoded, output, embeddings, vector
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    if weighted is None or total_weight == 0:
        raise ValueError("No embedding fragments were produced.")
    return (weighted / total_weight).cpu().numpy().astype(np.float32)


def batched_mean_embeddings(model, sequences: list[str], device: str) -> np.ndarray:
    if not sequences:
        return np.empty((0, 0), dtype=np.float32)
    tokens = model._tokenize(sequences)
    pad_token_id = model.tokenizer.pad_token_id
    bos_token_id = model.tokenizer.bos_token_id
    eos_token_id = model.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("ESM-C tokenizer has no pad token")
    model_device = next(model.parameters()).device
    autocast_context = (
        torch.autocast(device_type=model_device.type, dtype=torch.bfloat16)
        if model_device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad(), autocast_context:
        embeddings = model.embed(tokens)
        embeddings, _, _ = model.transformer(
            embeddings,
            sequence_id=tokens.eq(pad_token_id),
        )
    residue_mask = tokens.ne(pad_token_id)
    if bos_token_id is not None:
        residue_mask &= tokens.ne(bos_token_id)
    if eos_token_id is not None:
        residue_mask &= tokens.ne(eos_token_id)
    denominator = residue_mask.sum(dim=1, keepdim=True).clamp_min(1)
    pooled = (
        embeddings.float() * residue_mask.unsqueeze(-1)
    ).sum(dim=1) / denominator
    result = pooled.cpu().numpy().astype(np.float32)
    del tokens, embeddings, pooled, residue_mask, denominator
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def build_length_batches(
    items: list[tuple[str, str]],
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[tuple[str, str]]]:
    ordered = sorted(items, key=lambda item: (len(item[1]), item[0]))
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_max_length = 0
    for item in ordered:
        proposed_max = max(current_max_length, len(item[1]) + 2)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > max_batch_size
            or proposed_max * proposed_size > max_batch_tokens
        ):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(item)
        current_max_length = max(current_max_length, len(item[1]) + 2)
    if current:
        batches.append(current)
    return batches


def save_vector(vector_dir: Path, entry: str, vector: np.ndarray) -> None:
    vector_path = vector_dir / f"{entry}.npy"
    temporary = vector_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, vector.astype(np.float16))
    temporary.replace(vector_path)


def write_status(
    output_dir: Path,
    input_sequences: int,
    completed: int,
    reused: int,
    failures: list[dict[str, object]],
) -> None:
    pd.DataFrame(failures, columns=["Entry", "length", "error"]).to_csv(
        output_dir / "failures.csv", index=False
    )
    status = {
        "input_sequences": input_sequences,
        "newly_completed": completed,
        "reused": reused,
        "failed": len(failures),
    }
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )


def collate(output_dir: Path, candidates: pd.DataFrame) -> tuple[int, list[str]]:
    entries: list[str] = []
    vectors: list[np.ndarray] = []
    missing: list[str] = []
    vector_dir = output_dir / "vectors"
    for entry in candidates["Entry"].astype(str):
        path = vector_dir / f"{entry}.npy"
        if not path.exists():
            missing.append(entry)
            continue
        vector = np.load(path)
        if vector.ndim != 1:
            raise ValueError(f"Expected one-dimensional embedding for {entry}, got {vector.shape}")
        entries.append(entry)
        vectors.append(vector.astype(np.float32, copy=False))
    if vectors:
        matrix = np.stack(vectors)
        np.save(output_dir / "embeddings.npy", matrix)
        pd.DataFrame({"row": range(len(entries)), "Entry": entries}).to_csv(output_dir / "entries.csv", index=False)
    return len(entries), missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract resumable ESM-C 600M mean embeddings for TPS candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--input-sep", default="\\t")
    parser.add_argument("--entry-column", default="Entry")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--disable-batching", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.overlap < 0 or args.overlap >= args.max_residues:
        raise ValueError("overlap must be non-negative and smaller than max-residues")
    if args.max_batch_tokens <= 0 or args.max_batch_size <= 0:
        raise ValueError("batch token and size limits must be positive")

    candidates = pd.read_csv(args.input, sep=args.input_sep, dtype=str).fillna("")
    missing_columns = sorted(
        {args.entry_column, args.sequence_column} - set(candidates.columns)
    )
    if missing_columns:
        raise ValueError(f"Embedding input is missing columns: {missing_columns}")
    candidates = candidates[[args.entry_column, args.sequence_column]].rename(
        columns={args.entry_column: "Entry", args.sequence_column: "Sequence"}
    )
    candidates = candidates.drop_duplicates("Entry", keep="first").sort_values("Entry").reset_index(drop=True)
    candidates["Sequence"] = candidates["Sequence"].map(clean_sequence)
    output_dir = args.output_dir.resolve()
    vector_dir = output_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, object]] = []
    completed = 0
    reused = 0
    pending_short: list[tuple[str, str]] = []
    pending_long: list[tuple[str, str]] = []
    for row in candidates.itertuples(index=False):
        entry = str(row.Entry)
        sequence = str(row.Sequence)
        vector_path = vector_dir / f"{entry}.npy"
        if vector_path.exists() and not args.force:
            reused += 1
            continue
        if not sequence or not VALID_AA.match(sequence):
            failures.append(
                {"Entry": entry, "length": len(sequence), "error": "invalid_sequence"}
            )
            continue
        if len(sequence) <= args.max_residues and not args.disable_batching:
            pending_short.append((entry, sequence))
        else:
            pending_long.append((entry, sequence))

    model = None
    if pending_short or pending_long:
        from esm.models.esmc import ESMC

        model = ESMC.from_pretrained(args.model).eval().to(args.device)

    progress = tqdm(total=len(candidates), initial=reused + len(failures))

    def process_batch(batch: list[tuple[str, str]]) -> None:
        nonlocal completed
        if not batch:
            return
        try:
            assert model is not None
            vectors = batched_mean_embeddings(
                model, [sequence for _, sequence in batch], args.device
            )
            if len(vectors) != len(batch):
                raise ValueError("ESM-C batch output size mismatch")
            for (entry, _), vector in zip(batch, vectors):
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
                    {"Entry": entry, "length": len(sequence), "error": repr(exc)}
                )
                progress.update(1)
        if (completed + reused + len(failures)) % 25 == 0:
            write_status(output_dir, len(candidates), completed, reused, failures)

    for batch in build_length_batches(
        pending_short, args.max_batch_tokens, args.max_batch_size
    ):
        process_batch(batch)

    for entry, sequence in pending_long:
        try:
            assert model is not None
            vector = mean_embedding(
                model, sequence, args.max_residues, args.overlap, args.device
            )
            save_vector(vector_dir, entry, vector)
            completed += 1
        except Exception as exc:
            failures.append(
                {"Entry": entry, "length": len(sequence), "error": repr(exc)}
            )
        progress.update(1)
        if (completed + reused + len(failures)) % 25 == 0:
            write_status(output_dir, len(candidates), completed, reused, failures)
    progress.close()
    write_status(output_dir, len(candidates), completed, reused, failures)
    n_collated, missing = collate(output_dir, candidates)
    summary = {
        "input": str(args.input.resolve()),
        "output_dir": str(output_dir),
        "model": args.model,
        "device": args.device,
        "input_sequences": int(len(candidates)),
        "input_entry_column": args.entry_column,
        "input_sequence_column": args.sequence_column,
        "batching_enabled": not args.disable_batching,
        "max_batch_tokens": args.max_batch_tokens,
        "max_batch_size": args.max_batch_size,
        "newly_completed": completed,
        "reused": reused,
        "failed": len(failures),
        "collated_embeddings": n_collated,
        "missing_after_collation": missing,
        "embedding_matrix": str(output_dir / "embeddings.npy"),
        "entries": str(output_dir / "entries.csv"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
