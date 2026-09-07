from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
DEFAULT_ENTRIES = ROOT / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1/entries.csv"
DEFAULT_RUNTIME = ROOT / "external_runtime/rxnmapper"
DEFAULT_OUTPUT = ROOT / "data/external/rxnmapper_current/general_merged_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapper(runtime: Path, *, batch_size: int, allow_cuda: bool):
    if not allow_cuda:
        # This must happen before importing rxnmapper/torch. The mapping model is
        # preprocessing only and should not contend with the retrieval GPU job.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from rxnmapper import BatchedMapper  # type: ignore

    return BatchedMapper(batch_size=batch_size, canonicalize=False)


def model_asset_hashes(runtime: Path) -> dict[str, str]:
    model_dir = runtime / "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k"
    names = [
        "config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.txt",
    ]
    return {name: sha256_file(model_dir / name) for name in names if (model_dir / name).is_file()}


def write_checkpoint(frame: pd.DataFrame, output: Path) -> None:
    ordered = frame.sort_values("row", kind="mergesort").reset_index(drop=True)
    temp = output / "mapped_reactions.csv.tmp"
    ordered.to_csv(temp, index=False)
    temp.replace(output / "mapped_reactions.csv")
    success = ordered["success"].astype(bool) if len(ordered) else pd.Series(dtype=bool)
    progress = {
        "requested_reactions": int(ordered["requested_total"].max()) if len(ordered) else 0,
        "processed_reactions": int(len(ordered)),
        "successful_mappings": int(success.sum()) if len(ordered) else 0,
        "failed_mappings": int((~success).sum()) if len(ordered) else 0,
    }
    (output / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a resumable atom-mapped reaction registry with isolated RXNMapper. "
            "All failures and confidence values are retained; no confidence filtering is applied."
        )
    )
    parser.add_argument("--reactions", type=Path, default=DEFAULT_REACTIONS)
    parser.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Flush after this many mapping batches.")
    parser.add_argument("--max-reactions", type=int, default=0, help="Sorted registry prefix for smoke testing; 0 means all registered reactions.")
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true", help="Allow RXNMapper to see CUDA; default is CPU-only to avoid retrieval-GPU contention.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.checkpoint_every <= 0 or args.max_reactions < 0:
        raise ValueError("batch-size/checkpoint-every must be positive and max-reactions non-negative")

    reactions_path = args.reactions.resolve(); entries_path = args.entries.resolve(); runtime = args.runtime.resolve()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    reactions = pd.read_csv(reactions_path, dtype=str).fillna("")
    entries = pd.read_csv(entries_path, dtype=str).fillna("")
    required_reactions = {"reaction_id", "reaction_smiles"}
    if not required_reactions <= set(reactions):
        raise ValueError(f"reaction table missing {sorted(required_reactions - set(reactions))}")
    if not {"row", "reaction_id"} <= set(entries):
        raise ValueError("entries requires row and reaction_id")
    entries["row"] = pd.to_numeric(entries["row"], errors="raise").astype(int)
    entries = entries.sort_values("row", kind="mergesort").reset_index(drop=True)
    if entries["reaction_id"].duplicated().any() or reactions["reaction_id"].duplicated().any():
        raise ValueError("reaction IDs must be unique")
    source = entries.merge(reactions[["reaction_id", "reaction_smiles", "source_layer"]], on="reaction_id", how="left", validate="one_to_one")
    if source["reaction_smiles"].isna().any() or source["reaction_smiles"].eq("").any():
        missing = source.loc[source["reaction_smiles"].isna() | source["reaction_smiles"].eq(""), "reaction_id"].tolist()
        raise ValueError(f"missing reaction SMILES for {len(missing)} registered reactions; examples={missing[:10]}")
    if args.max_reactions > 0:
        source = source.iloc[: args.max_reactions].copy()
    requested_total = len(source)

    current_path = output / "mapped_reactions.csv"
    columns = [
        "row", "reaction_id", "original_smiles", "source_layer", "mapped_rxn",
        "confidence", "success", "mapping_status", "requested_total",
    ]
    if current_path.is_file():
        current = pd.read_csv(current_path, dtype={"reaction_id": str, "success": str}).fillna("")
        current["row"] = pd.to_numeric(current["row"], errors="raise").astype(int)
        current["confidence"] = pd.to_numeric(current["confidence"], errors="coerce")
        current["success"] = current["success"].astype(str).str.lower().eq("true")
        # Protect against resuming into a different requested registry.
        expected = source[["row", "reaction_id"]]
        overlap = current[["row", "reaction_id"]].merge(expected, on=["row", "reaction_id"], how="inner")
        if len(overlap) != len(current):
            raise ValueError("existing mapping checkpoint does not match requested entries; use another output directory")
    else:
        current = pd.DataFrame(columns=columns)

    processed = set(current["reaction_id"].astype(str))
    if args.retry_failures and len(current):
        failed = set(current.loc[~current["success"], "reaction_id"].astype(str))
        current = current[~current["reaction_id"].isin(failed)].copy()
        processed -= failed
    pending = source[~source["reaction_id"].isin(processed)].copy()
    mapper = load_mapper(runtime, batch_size=args.batch_size, allow_cuda=args.allow_cuda)
    started = time.time(); new_rows: list[dict[str, object]] = []; batches = 0
    for start in range(0, len(pending), args.batch_size):
        local = pending.iloc[start : start + args.batch_size]
        infos = list(mapper.map_reactions_with_info(local["reaction_smiles"].astype(str).tolist()))
        if len(infos) != len(local):
            raise RuntimeError("RXNMapper returned a different number of rows than requested")
        for record, info in zip(local.itertuples(index=False), infos, strict=True):
            success = bool(info and str(info.get("mapped_rxn", "")) not in {"", ">>"})
            confidence = float(info.get("confidence")) if info and info.get("confidence") is not None else float("nan")
            new_rows.append(
                {
                    "row": int(record.row),
                    "reaction_id": str(record.reaction_id),
                    "original_smiles": str(record.reaction_smiles),
                    "source_layer": str(record.source_layer),
                    "mapped_rxn": str(info.get("mapped_rxn", "")) if info else "",
                    "confidence": confidence,
                    "success": success,
                    "mapping_status": "mapped" if success else "rxnmapper_failed",
                    "requested_total": requested_total,
                }
            )
        batches += 1
        if batches % args.checkpoint_every == 0 or start + len(local) >= len(pending):
            if new_rows:
                current = pd.concat([current, pd.DataFrame(new_rows)], ignore_index=True)
                new_rows.clear()
            write_checkpoint(current, output)
            elapsed = max(time.time() - started, 1e-9)
            session_done = start + len(local)
            print(
                json.dumps(
                    {
                        "requested_reactions": requested_total,
                        "processed_reactions": len(current),
                        "successful_mappings": int(current["success"].sum()),
                        "session_reactions_per_second": float(session_done / elapsed),
                    }
                ),
                flush=True,
            )

    if len(current) != requested_total:
        raise RuntimeError(f"mapping ended incomplete: {len(current)}/{requested_total}")
    current = current.sort_values("row", kind="mergesort").reset_index(drop=True)
    if current["reaction_id"].tolist() != source["reaction_id"].astype(str).tolist():
        raise RuntimeError("mapping output order differs from registered reaction order")
    write_checkpoint(current, output)
    confidence = pd.to_numeric(current["confidence"], errors="coerce")
    success = current["success"].astype(bool)
    metadata = runtime / "rxnmapper-0.4.3.dist-info/METADATA"
    manifest = {
        "version": "rxnmapper-general-merged-v1",
        "purpose": "atom mapping required by the official CLIPZyme reaction-graph encoder; preprocessing only",
        "reaction_table": str(reactions_path),
        "reaction_table_sha256": sha256_file(reactions_path),
        "registered_entries": str(entries_path),
        "registered_entries_sha256": sha256_file(entries_path),
        "reaction_count": requested_total,
        "successful_mappings": int(success.sum()),
        "failed_mappings": int((~success).sum()),
        "success_fraction": float(success.mean()),
        "confidence_mean_success": float(confidence[success].mean()) if success.any() else None,
        "confidence_median_success": float(confidence[success].median()) if success.any() else None,
        "confidence_lt_0p5_success": int(((confidence < 0.5) & success).sum()),
        "confidence_lt_0p2_success": int(((confidence < 0.2) & success).sum()),
        "confidence_policy": "retain every successful mapping regardless of confidence; preserve failures explicitly; stratify later evaluations rather than filtering after seeing scores",
        "rxnmapper_runtime": str(runtime),
        "rxnmapper_metadata_sha256": sha256_file(metadata) if metadata.is_file() else None,
        "rxnmapper_model_assets_sha256": model_asset_hashes(runtime),
        "rxnmapper_config": {
            "version": "0.4.3",
            "license": "MIT",
            "model": "albert_heads_8_uspto_all_1310k",
            "canonicalize": False,
            "batch_size": args.batch_size,
            "cuda_visible": bool(args.allow_cuda),
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
