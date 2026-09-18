from __future__ import annotations

import argparse
import contextlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
MOTIF_NAMES = ("ddxxd", "nse_dte", "dxdd", "qw")
DDXXD = re.compile(r"DD..D")
NSE = re.compile(r"[ND]D..[ST]...E")
DTE = re.compile(r"DTE")
DXDD = re.compile(r"D.DD")
QW = re.compile(r"QW")
VALID_AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def motif_centers(pattern: re.Pattern[str], sequence: str) -> list[int]:
    return [int((match.start() + match.end() - 1) // 2) for match in pattern.finditer(sequence)]


def select_motif_contexts(sequence: str) -> dict[str, dict[str, object]]:
    dd_positions = motif_centers(DDXXD, sequence)
    nse_positions = sorted(set(motif_centers(NSE, sequence) + motif_centers(DTE, sequence)))
    dxdd_positions = motif_centers(DXDD, sequence)
    qw_positions = motif_centers(QW, sequence)

    selected_dd: list[int] = []
    selected_nse: list[int] = []
    pair_distance = -1
    pairs = [
        (dd, nse)
        for dd in dd_positions
        for nse in nse_positions
        if nse > dd and 20 <= nse - dd <= 350
    ]
    if pairs:
        dd, nse = min(pairs, key=lambda pair: (abs((pair[1] - pair[0]) - 140), -pair[1]))
        selected_dd = [dd]
        selected_nse = [nse]
        pair_distance = nse - dd
    else:
        if dd_positions:
            selected_dd = [max(dd_positions)]
        if nse_positions:
            selected_nse = [max(nse_positions)]
    return {
        "ddxxd": {
            "all_positions": dd_positions,
            "selected_positions": selected_dd,
        },
        "nse_dte": {
            "all_positions": nse_positions,
            "selected_positions": selected_nse,
        },
        "dxdd": {
            "all_positions": dxdd_positions,
            "selected_positions": [min(dxdd_positions)] if dxdd_positions else [],
        },
        "qw": {
            "all_positions": qw_positions,
            "selected_positions": qw_positions,
        },
        "classI_pair": {
            "pair_distance": pair_distance,
            "present": pair_distance >= 0,
        },
    }


def pool_windows(
    residue_embeddings: torch.Tensor,
    positions: list[int],
    window: int,
) -> torch.Tensor:
    if not positions:
        return torch.zeros(
            residue_embeddings.shape[-1],
            dtype=torch.float32,
            device=residue_embeddings.device,
        )
    vectors: list[torch.Tensor] = []
    length = residue_embeddings.shape[0]
    for center in positions:
        start = max(0, int(center) - window)
        end = min(length, int(center) + window + 1)
        vectors.append(residue_embeddings[start:end].float().mean(dim=0))
    return torch.stack(vectors).mean(dim=0)


def descriptor_vector(
    contexts: dict[str, dict[str, object]],
    sequence_length: int,
) -> np.ndarray:
    values: list[float] = []
    denominator = max(sequence_length - 1, 1)
    for name in MOTIF_NAMES:
        all_positions = list(contexts[name]["all_positions"])
        selected_positions = list(contexts[name]["selected_positions"])
        values.extend(
            [
                float(bool(all_positions)),
                min(len(all_positions), 4) / 4.0,
                (
                    float(np.mean(selected_positions)) / denominator
                    if selected_positions
                    else 0.0
                ),
            ]
        )
    pair = contexts["classI_pair"]
    pair_distance = int(pair["pair_distance"])
    values.extend(
        [
            float(bool(pair["present"])),
            float(pair_distance) / max(sequence_length, 1) if pair_distance >= 0 else 0.0,
        ]
    )
    return np.asarray(values, dtype=np.float32)


def build_length_batches(
    rows: list[tuple[str, str]],
    max_batch_tokens: int,
    max_batch_size: int,
) -> list[list[tuple[str, str]]]:
    ordered = sorted(rows, key=lambda item: (len(item[1]), item[0]))
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    maximum = 0
    for item in ordered:
        proposed_maximum = max(maximum, len(item[1]) + 2)
        proposed_size = len(current) + 1
        if current and (
            proposed_size > max_batch_size
            or proposed_maximum * proposed_size > max_batch_tokens
        ):
            batches.append(current)
            current = []
            maximum = 0
        current.append(item)
        maximum = max(maximum, len(item[1]) + 2)
    if current:
        batches.append(current)
    return batches


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    return vector / norm if float(norm) > 0 else vector


def extract_batch(
    model,
    batch: list[tuple[str, str]],
    global_vectors: dict[str, np.ndarray],
    window: int,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    sequences = [sequence for _, sequence in batch]
    tokens = model._tokenize(sequences)
    pad_token_id = model.tokenizer.pad_token_id
    bos_token_id = model.tokenizer.bos_token_id
    eos_token_id = model.tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("ESM-C tokenizer does not expose a pad token")
    device = next(model.parameters()).device
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda"
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

    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for index, (entry, sequence) in enumerate(batch):
        residues = embeddings[index, residue_mask[index]]
        if len(residues) != len(sequence):
            raise ValueError(
                f"Token/residue mismatch for {entry}: {len(residues)} != {len(sequence)}"
            )
        contexts = select_motif_contexts(sequence)
        blocks = [
            np.asarray(global_vectors[entry], dtype=np.float32),
        ]
        for motif_name in MOTIF_NAMES:
            vector = pool_windows(
                residues,
                list(contexts[motif_name]["selected_positions"]),
                window,
            )
            blocks.append(normalize_vector(vector).cpu().numpy().astype(np.float32))
        descriptors = descriptor_vector(contexts, len(sequence))
        blocks.append(descriptors)
        rows.append(np.concatenate(blocks).astype(np.float32))
        audit_row: dict[str, object] = {
            "Entry": entry,
            "sequence_length": len(sequence),
            "classI_pair_present": bool(contexts["classI_pair"]["present"]),
            "classI_pair_distance": int(contexts["classI_pair"]["pair_distance"]),
        }
        for motif_name in MOTIF_NAMES:
            all_positions = list(contexts[motif_name]["all_positions"])
            selected_positions = list(contexts[motif_name]["selected_positions"])
            audit_row[f"{motif_name}_count"] = len(all_positions)
            audit_row[f"{motif_name}_positions"] = ";".join(map(str, all_positions))
            audit_row[f"{motif_name}_selected"] = ";".join(map(str, selected_positions))
        audit.append(audit_row)
    del tokens, embeddings, residue_mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract motif-aware residue-context ESM-C features for TPS retrieval."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sep", default="\t")
    parser.add_argument("--entry-column", default="Entry")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--global-embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="esmc_600m")
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.window < 0:
        raise ValueError("window must be non-negative")
    candidates = pd.read_csv(args.input, sep=args.input_sep, dtype=str).fillna("")
    missing = {args.entry_column, args.sequence_column} - set(candidates.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    candidates = candidates[[args.entry_column, args.sequence_column]].rename(
        columns={args.entry_column: "Entry", args.sequence_column: "Sequence"}
    )
    candidates = candidates.drop_duplicates("Entry", keep="first").sort_values("Entry")
    candidates["Sequence"] = candidates["Sequence"].map(clean_sequence)

    global_dir = args.global_embedding_dir.resolve()
    global_entries = pd.read_csv(global_dir / "entries.csv", dtype=str).fillna("")
    global_entries["row"] = pd.to_numeric(global_entries["row"]).astype(int)
    global_matrix = np.load(global_dir / "embeddings.npy").astype(np.float32)
    global_vectors = {
        str(row.Entry): global_matrix[int(row.row)]
        for row in global_entries.itertuples(index=False)
    }
    valid_rows: list[tuple[str, str]] = []
    skipped: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        entry = str(row.Entry)
        sequence = str(row.Sequence)
        if entry not in global_vectors:
            skipped.append({"Entry": entry, "reason": "missing_global_embedding"})
        elif not sequence or not VALID_AA.fullmatch(sequence):
            skipped.append({"Entry": entry, "reason": "invalid_sequence"})
        else:
            valid_rows.append((entry, sequence))
    if not valid_rows:
        raise ValueError("No valid sequences with global embeddings")

    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(args.model).eval().to(args.device)
    feature_by_entry: dict[str, np.ndarray] = {}
    audits: list[dict[str, object]] = []
    batches = build_length_batches(
        valid_rows,
        args.max_batch_tokens,
        args.max_batch_size,
    )
    for batch in tqdm(batches, desc="motif-context batches"):
        vectors, batch_audit = extract_batch(
            model,
            batch,
            global_vectors,
            args.window,
        )
        for (entry, _), vector in zip(batch, vectors):
            feature_by_entry[entry] = vector
        audits.extend(batch_audit)

    entries = [entry for entry, _ in valid_rows if entry in feature_by_entry]
    matrix = np.stack([feature_by_entry[entry] for entry in entries]).astype(np.float32)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", matrix)
    pd.DataFrame({"row": range(len(entries)), "Entry": entries}).to_csv(
        output_dir / "entries.csv", index=False
    )
    audit = pd.DataFrame(audits).sort_values("Entry")
    audit.to_csv(output_dir / "motif_context_audit.csv", index=False)
    pd.DataFrame(skipped).to_csv(output_dir / "skipped.csv", index=False)
    descriptor_dimension = 3 * len(MOTIF_NAMES) + 2
    summary = {
        "input": str(args.input.resolve()),
        "global_embedding_dir": str(global_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "device": args.device,
        "window": args.window,
        "motif_names": MOTIF_NAMES,
        "global_dimension": int(global_matrix.shape[1]),
        "motif_context_dimension": int(len(MOTIF_NAMES) * global_matrix.shape[1]),
        "descriptor_dimension": descriptor_dimension,
        "output_dimension": int(matrix.shape[1]),
        "input_sequences": int(len(candidates)),
        "embedded_sequences": int(len(entries)),
        "skipped_sequences": int(len(skipped)),
        "motif_presence": {
            name: int((audit[f"{name}_count"] > 0).sum()) for name in MOTIF_NAMES
        },
        "classI_pair_present": int(audit["classI_pair_present"].astype(bool).sum()),
        "outputs": {
            "embeddings": str(output_dir / "embeddings.npy"),
            "entries": str(output_dir / "entries.csv"),
            "motif_context_audit": str(output_dir / "motif_context_audit.csv"),
            "skipped": str(output_dir / "skipped.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
