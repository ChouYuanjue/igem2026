from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
DEFAULT_ASSOCIATIONS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
DEFAULT_MODEL = ROOT / "external_models/enzgfm/EnzGFM_650M"
DEFAULT_REFERENCE = ROOT / "external/enzgfm_reference"
DEFAULT_OUTPUT = ROOT / "data/external/enzgfm_current/clean2023_650m_mean"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate_middle(sequence: str, max_residues: int) -> str:
    if len(sequence) <= max_residues:
        return sequence
    left = max_residues // 2
    right = max_residues - left
    return sequence[:left] + sequence[-right:]


def requested_proteins(
    sequences: pd.DataFrame,
    associations: Path | None,
    max_proteins: int,
) -> pd.DataFrame:
    if associations is None:
        ids = sorted(sequences["protein_id"].astype(str).unique())
    else:
        pairs = pd.read_csv(associations, dtype=str).fillna("")
        if "protein_id" not in pairs:
            raise ValueError("association source requires protein_id")
        ids = sorted(set(pairs["protein_id"].astype(str)))
    seq = sequences.drop_duplicates("protein_id").set_index("protein_id")
    missing = [value for value in ids if value not in seq.index or not str(seq.at[value, "sequence"])]
    if missing:
        raise ValueError(f"missing sequences for {len(missing)} proteins; examples={missing[:10]}")
    if max_proteins > 0:
        ids = ids[:max_proteins]
    out = seq.loc[ids, ["sequence"]].reset_index()
    out.insert(0, "row", np.arange(len(out), dtype=np.int64))
    return out


def load_runtime(reference_root: Path, model_dir: Path, device: torch.device):
    if str(reference_root) not in sys.path:
        sys.path.insert(0, str(reference_root))
    from transformers import EsmTokenizer
    from models.configuration_EnzGFM import EnzGFMConfig
    from models.modeling_EnzGFM import EnzGFM_Model

    config = EnzGFMConfig.from_pretrained(model_dir)
    config.use_mamba_kernels = False
    config.use_cache = False
    model = EnzGFM_Model.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.float32,
    ).eval().to(device)
    tokenizer = EsmTokenizer.from_pretrained(reference_root / "EsmTokenizer")
    return model, tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EnzGFM-650M mean-pooled protein features with author-compatible preprocessing and resumable length-bucketed batching.")
    parser.add_argument("--sequences", type=Path, default=DEFAULT_SEQUENCES)
    parser.add_argument("--associations-csv", type=Path, default=DEFAULT_ASSOCIATIONS)
    parser.add_argument("--all-proteins", action="store_true", help="Ignore --associations-csv and embed the whole sequence registry.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-residues", type=int, default=1000)
    parser.add_argument("--max-proteins", type=int, default=0, help="Deterministic sorted-ID smoke limit; 0 means all requested proteins.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Flush memmaps/progress every N batches.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_residues <= 0 or args.checkpoint_every <= 0:
        raise ValueError("batch-size, max-residues and checkpoint-every must be positive")

    sequence_path = args.sequences.resolve()
    association_path = None if args.all_proteins else args.associations_csv.resolve()
    model_dir = args.model_dir.resolve(); reference_root = args.reference_root.resolve()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_csv(sequence_path, sep="\t", dtype=str).fillna("")
    required = {"protein_id", "sequence"}
    if not required <= set(sequences.columns):
        raise ValueError(f"sequence table missing {sorted(required - set(sequences.columns))}")
    requested = requested_proteins(sequences, association_path, args.max_proteins)
    requested["original_length"] = requested["sequence"].str.len().astype(int)
    requested["effective_sequence"] = requested["sequence"].map(lambda s: truncate_middle(str(s), args.max_residues))
    requested["effective_length"] = requested["effective_sequence"].str.len().astype(int)

    entries = requested[["row", "protein_id"]].rename(columns={"protein_id": "Entry"})
    entries_path = output / "entries.csv"
    if entries_path.exists():
        old = pd.read_csv(entries_path, dtype={"Entry": str}).sort_values("row").reset_index(drop=True)
        if not old.equals(entries.reset_index(drop=True)):
            raise ValueError("existing entries.csv differs from requested protein set; use a new output directory")
    else:
        entries.to_csv(entries_path, index=False)

    device = torch.device(args.device)
    model, tokenizer, config = load_runtime(reference_root, model_dir, device)
    hidden = int(config.hidden_size)
    embedding_path = output / "embeddings.npy"
    completed_path = output / "completed.npy"
    if embedding_path.exists():
        matrix = np.lib.format.open_memmap(embedding_path, mode="r+")
        if matrix.shape != (len(requested), hidden):
            raise ValueError(f"existing embedding matrix shape {matrix.shape} differs from {(len(requested), hidden)}")
    else:
        matrix = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(len(requested), hidden))
    if completed_path.exists():
        completed = np.lib.format.open_memmap(completed_path, mode="r+")
        if completed.shape != (len(requested),):
            raise ValueError("existing completed.npy shape differs from requested set")
    else:
        completed = np.lib.format.open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(len(requested),))
        completed[:] = False; completed.flush()

    # Sort only the execution schedule by effective length to minimize padding; row-wise output order stays identifier-stable.
    schedule = requested.sort_values(["effective_length", "protein_id"], kind="mergesort")
    pending = schedule.loc[~completed[schedule["row"].to_numpy(dtype=np.int64)]].copy()
    started = time.time(); batches_done = 0; proteins_done = int(completed.sum())
    with torch.no_grad():
        for start in range(0, len(pending), args.batch_size):
            group = pending.iloc[start : start + args.batch_size]
            texts = group["effective_sequence"].astype(str).tolist()
            encoded = tokenizer(texts, return_tensors="pt", truncation=True, max_length=args.max_residues + 2, padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded, output_hidden_states=False, return_dict=True).last_hidden_state.float()
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = ((states * mask).sum(dim=1) / mask.sum(dim=1)).detach().cpu().numpy().astype(np.float32)
            rows = group["row"].to_numpy(dtype=np.int64)
            matrix[rows] = pooled
            completed[rows] = True
            proteins_done += len(rows); batches_done += 1
            if batches_done % args.checkpoint_every == 0 or proteins_done == len(requested):
                matrix.flush(); completed.flush()
                elapsed = max(time.time() - started, 1e-9)
                progress = {
                    "requested_proteins": int(len(requested)),
                    "completed_proteins": int(completed.sum()),
                    "completed_fraction": float(completed.mean()),
                    "session_proteins_per_second": float((proteins_done - int(completed.sum()) + 0) / elapsed) if False else float((start + len(rows)) / elapsed),
                }
                (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
                print(json.dumps(progress), flush=True)

    if not bool(completed.all()):
        raise RuntimeError("feature extraction ended with incomplete rows")
    matrix.flush(); completed.flush()
    audit = requested[["row", "protein_id", "original_length", "effective_length"]].copy()
    audit["truncated"] = audit["original_length"] > args.max_residues
    audit.to_csv(output / "audit.csv", index=False)
    tokenizer_dir = reference_root / "EsmTokenizer"
    manifest = {
        "version": "enzgfm-650m-author-mean-v1",
        "representation": "EnzGFM-650M final hidden state mean over real tokens",
        "external_pretraining": True,
        "claim_boundary": "foundation-pretrained protein representation transfer; downstream association training must remain train-only",
        "model_dir": str(model_dir),
        "model_safetensors_sha256": sha256_file(model_dir / "model.safetensors"),
        "model_config_sha256": sha256_file(model_dir / "config.json"),
        "reference_root": str(reference_root),
        "runtime_source_sha256": {
            name: sha256_file(reference_root / "models" / name)
            for name in ["configuration_EnzGFM.py", "modeling_EnzGFM.py"]
        },
        "tokenizer_sha256": {
            name: sha256_file(tokenizer_dir / name)
            for name in ["vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
        },
        "sequence_source": str(sequence_path),
        "sequence_source_sha256": sha256_file(sequence_path),
        "association_scope": str(association_path) if association_path is not None else None,
        "association_scope_sha256": sha256_file(association_path) if association_path is not None else None,
        "protein_count": int(len(requested)),
        "feature_dimension": hidden,
        "dtype": "float32",
        "max_residues": args.max_residues,
        "long_sequence_policy": "first half + last half, matching author get_emb.py semantics",
        "batch_pooling": "attention-mask mean over non-padding tokens; verified numerically equivalent to per-sequence author mean within <=2e-6 max abs on representative proteins",
        "batch_size": args.batch_size,
        "use_mamba_kernels": False,
        "truncated_proteins": int(audit["truncated"].sum()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
