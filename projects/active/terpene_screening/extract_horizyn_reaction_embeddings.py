from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
HORIZYN_ROOT = ROOT / "external/horizyn"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HORIZYN_ROOT) not in sys.path:
    sys.path.insert(0, str(HORIZYN_ROOT))

from projects.active.terpene_screening.train_horizyn_reaction_adapter_double_cold import (  # noqa: E402
    build_horizyn_fingerprints,
    encode_horizyn_reactions,
)

DEFAULT_CONFIG = HORIZYN_ROOT / "configs/sota.yaml"
DEFAULT_CHECKPOINT = HORIZYN_ROOT / "checkpoints/horizyn_v1_0_dev.ckpt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract official Horizyn reaction embeddings for a reaction table."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sep", default=",")
    parser.add_argument("--id-column", required=True)
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, sep=args.input_sep, dtype=str).fillna("")
    missing = {args.id_column, args.smiles_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    reactions = frame[[args.id_column, args.smiles_column]].rename(
        columns={args.id_column: "reaction_id", args.smiles_column: "reaction_smiles"}
    )
    reactions = reactions.drop_duplicates("reaction_id", keep="first").reset_index(drop=True)
    if reactions["reaction_id"].eq("").any():
        raise ValueError("Reaction IDs must be non-empty")

    output_dir = args.output_dir.resolve()
    fingerprint_cache = output_dir / "fingerprint_cache"
    valid_mask = reactions["reaction_smiles"].astype(str).str.contains(">>", regex=False)
    valid_reactions = reactions[valid_mask].reset_index(drop=True)
    if valid_reactions.empty:
        raise ValueError("No parseable reaction SMILES are available for Horizyn encoding")
    fingerprints, valid_audit = build_horizyn_fingerprints(
        valid_reactions,
        args.config.resolve(),
        fingerprint_cache,
    )
    valid_embeddings = encode_horizyn_reactions(
        fingerprints,
        args.checkpoint.resolve(),
        torch.device(args.device),
        args.batch_size,
    )
    valid_success = valid_audit["success"].astype(str).str.lower().eq("true").to_numpy()
    valid_embeddings[~valid_success] = 0.0
    embeddings = np.zeros((len(reactions), valid_embeddings.shape[1]), dtype=np.float32)
    valid_rows = np.flatnonzero(valid_mask.to_numpy())
    embeddings[valid_rows] = valid_embeddings
    audit = pd.DataFrame(
        {
            "reaction_id": reactions["reaction_id"].astype(str),
            "success": False,
            "fingerprint_mode": "missing_reaction_zero",
            "error": "missing_or_unparseable_reaction_smiles",
        }
    )
    valid_audit_by_id = valid_audit.set_index("reaction_id")
    for row_index in valid_rows:
        reaction_id = str(reactions.iloc[row_index]["reaction_id"])
        audit.loc[row_index, ["success", "fingerprint_mode", "error"]] = valid_audit_by_id.loc[
            reaction_id, ["success", "fingerprint_mode", "error"]
        ].to_list()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings.astype(np.float32))
    pd.DataFrame(
        {"row": range(len(reactions)), "reaction_id": reactions["reaction_id"]}
    ).to_csv(output_dir / "entries.csv", index=False)
    audit.to_csv(output_dir / "fingerprint_audit.csv", index=False)
    reactions.to_csv(output_dir / "reaction_table.csv", index=False)
    summary = {
        "input": str(args.input.resolve()),
        "id_column": args.id_column,
        "smiles_column": args.smiles_column,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "device": args.device,
        "n_reactions": int(len(reactions)),
        "embedding_dimension": int(embeddings.shape[1]),
        "fingerprint_success": int(
            audit["success"].astype(str).str.lower().eq("true").sum()
        ),
        "fingerprint_modes": audit["fingerprint_mode"].value_counts().to_dict(),
        "outputs": {
            "embeddings": str(output_dir / "embeddings.npy"),
            "entries": str(output_dir / "entries.csv"),
            "fingerprint_audit": str(output_dir / "fingerprint_audit.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
