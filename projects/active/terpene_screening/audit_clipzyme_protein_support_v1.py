from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import normalize_reaction_bag, normalize_sequence, sha256_file

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REACTZYME_TEST = ROOT / "data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt"
DEFAULT_REACTZYME_REACTION_SUPPORT = ROOT / "results/clipzyme_reactzyme_direction_support_v1/eligible_reaction_queries.csv"
DEFAULT_CLIP_SEQUENCES = ROOT / "external_models/clipzyme_audit/clipzyme_data/uniprot2sequence.p"
DEFAULT_SCREENING_SET = ROOT / "external_models/clipzyme_audit/clipzyme_data/clipzyme_screening_set.p"
DEFAULT_PROTEIN_SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
DEFAULT_BENCHMARK_ROOT = ROOT / "results/broad_rhea_fair_benchmarks_v1"
DEFAULT_FALLBACK_REACTION_SUPPORT = ROOT / "results/clipzyme_directed_fallback_support_v1"
DEFAULT_OUTPUT = ROOT / "results/clipzyme_protein_support_v1"
FALLBACK_CELLS = (
    "reactzyme_reaction_projected_double_cold",
    "temporal_post2020_double_cold",
    "broad_reaction_hash_cold_protein_seen",
)


def load_screening_uniprots(path: Path) -> tuple[set[str] | None, dict[str, object]]:
    if not path.is_file():
        return None, {"present": False, "path": str(path)}
    payload = pickle.load(path.open("rb"))
    if not isinstance(payload, dict) or "uniprots" not in payload:
        raise ValueError("CLIPZyme screening set must be a dict containing uniprots")
    uniprots = [str(value) for value in payload["uniprots"]]
    if len(uniprots) != len(set(uniprots)):
        raise ValueError("CLIPZyme screening set contains duplicate UniProt IDs")
    meta: dict[str, object] = {"present": True, "path": str(path), "uniprot_count": len(uniprots)}
    if "hiddens" in payload:
        hiddens = payload["hiddens"]
        shape = list(np.shape(hiddens))
        meta["hidden_shape"] = shape
        if shape and shape[0] != len(uniprots):
            raise ValueError("CLIPZyme screening hiddens/uniprots row mismatch")
    return set(uniprots), meta


def load_clip_sequences(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    payload = pickle.load(path.open("rb"))
    if not isinstance(payload, dict):
        raise ValueError("CLIPZyme uniprot2sequence asset must be a dict")
    id_to_seq = {str(k): normalize_sequence(str(v)) for k, v in payload.items() if normalize_sequence(str(v))}
    seq_to_ids: dict[str, list[str]] = defaultdict(list)
    for protein_id, seq in id_to_seq.items():
        seq_to_ids[seq].append(protein_id)
    for ids in seq_to_ids.values():
        ids.sort()
    return id_to_seq, dict(seq_to_ids)


def load_reactzyme_rows(path: Path) -> pd.DataFrame:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = []
    for source_key, value in payload.items():
        if len(value) < 2:
            raise ValueError(f"unexpected ReactZyme tuple at {source_key}")
        rows.append({
            "source_key": str(source_key),
            "reaction_bag": normalize_reaction_bag(value[0]),
            "sequence": normalize_sequence(value[1]),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != 14692 or frame["reaction_bag"].nunique() != 386:
        raise AssertionError("official ReactZyme reaction_smi test support drift")
    return frame


def reactzyme_audit(
    test: pd.DataFrame,
    *,
    seq_to_clip_ids: dict[str, list[str]],
    screening_uniprots: set[str] | None,
    eligible_reaction_path: Path,
) -> tuple[dict[str, object], pd.DataFrame]:
    eligible_reactions: set[str] | None = None
    if eligible_reaction_path.is_file():
        eligible = pd.read_csv(eligible_reaction_path, dtype=str).fillna("")
        if "reaction_bag" not in eligible:
            raise ValueError("ReactZyme eligible reaction audit lacks reaction_bag")
        eligible_reactions = set(eligible["reaction_bag"].astype(str))
    local = test.copy()
    local["reaction_side_eligible"] = True if eligible_reactions is None else local["reaction_bag"].isin(eligible_reactions)
    local["author_exact_sequence_ids"] = local["sequence"].map(lambda seq: seq_to_clip_ids.get(seq, []))
    local["author_exact_sequence_match"] = local["author_exact_sequence_ids"].map(bool)
    # source_key is accepted only when it is itself an exact author screening identifier. We never
    # substitute another UniProt ID merely because its amino-acid sequence is identical: CLIPZyme's
    # released protein embedding also depends on the author-associated structure input.
    if screening_uniprots is None:
        local["source_key_in_screening"] = False
        local["executable_precomputed_protein_support"] = False
    else:
        local["source_key_in_screening"] = local["source_key"].isin(screening_uniprots)
        local["executable_precomputed_protein_support"] = local["source_key_in_screening"]
    unique_seq = local.drop_duplicates("sequence")
    executable = local[local["reaction_side_eligible"] & local["executable_precomputed_protein_support"]]
    summary = {
        "source_rows": int(len(local)),
        "unique_sequences": int(local["sequence"].nunique()),
        "author_exact_sequence_match_unique": int(unique_seq["author_exact_sequence_match"].sum()),
        "screening_asset_present": screening_uniprots is not None,
        "source_key_in_screening_unique_rows": int(local.drop_duplicates("source_key")["source_key_in_screening"].sum()),
        "reaction_side_filter_present": eligible_reactions is not None,
        "reaction_side_eligible_unique_reactions": int(local.loc[local["reaction_side_eligible"], "reaction_bag"].nunique()),
        "common_executable_positive_rows": int(len(executable)),
        "common_executable_r2e_queries": int(executable["reaction_bag"].nunique()),
        "common_executable_e2r_queries_by_source_key": int(executable["source_key"].nunique()),
        "native_protein_inference_still_allowed_for_unsupported": True,
        "exact_sequence_alias_is_executable_support": False,
    }
    return summary, local


def fallback_audit(
    *,
    benchmark_root: Path,
    protein_sequences_path: Path,
    seq_to_clip_ids: dict[str, list[str]],
    screening_uniprots: set[str] | None,
    reaction_support_root: Path,
) -> tuple[list[dict[str, object]], dict[str, pd.DataFrame]]:
    proteins = pd.read_csv(protein_sequences_path, sep="\t", dtype=str).fillna("")
    if not {"protein_id", "sequence"} <= set(proteins):
        raise ValueError("general protein sequence table lacks protein_id/sequence")
    seq_lookup = dict(zip(proteins["protein_id"].astype(str), proteins["sequence"].map(normalize_sequence)))
    summaries: list[dict[str, object]] = []
    details: dict[str, pd.DataFrame] = {}
    for priority, cell in enumerate(FALLBACK_CELLS, start=1):
        pair_path = benchmark_root / cell / "test_pairs.csv"
        reaction_detail_path = reaction_support_root / f"{priority:02d}_{cell}_reaction_support.csv"
        if not pair_path.is_file():
            summaries.append({"priority": priority, "cell": cell, "status": "missing_frozen_cell"})
            continue
        pairs = pd.read_csv(pair_path, dtype=str).fillna("")
        if not {"protein_id", "reaction_id"} <= set(pairs):
            raise ValueError(f"{pair_path} lacks protein_id/reaction_id")
        reaction_eligible: set[str] | None = None
        if reaction_detail_path.is_file():
            r = pd.read_csv(reaction_detail_path, dtype=str).fillna("")
            if not {"reaction_id", "native_reaction_supported"} <= set(r):
                raise ValueError(f"{reaction_detail_path} lacks reaction support columns")
            enabled = r["native_reaction_supported"].astype(str).str.lower().eq("true")
            reaction_eligible = set(r.loc[enabled, "reaction_id"].astype(str))
        local = pairs[["protein_id", "reaction_id"]].drop_duplicates().copy()
        local["sequence"] = local["protein_id"].map(lambda pid: normalize_sequence(seq_lookup.get(str(pid), "")))
        local["author_exact_sequence_ids"] = local["sequence"].map(lambda seq: seq_to_clip_ids.get(seq, []) if seq else [])
        local["author_exact_sequence_match"] = local["author_exact_sequence_ids"].map(bool)
        if screening_uniprots is None:
            local["protein_id_in_screening"] = False
            local["executable_precomputed_protein_support"] = False
        else:
            local["protein_id_in_screening"] = local["protein_id"].isin(screening_uniprots)
            local["executable_precomputed_protein_support"] = local["protein_id_in_screening"]
        local["reaction_side_eligible"] = True if reaction_eligible is None else local["reaction_id"].isin(reaction_eligible)
        local["common_executable_positive"] = local["reaction_side_eligible"] & local["executable_precomputed_protein_support"]
        details[cell] = local
        unique_proteins = local.drop_duplicates("protein_id")
        common = local[local["common_executable_positive"]]
        summaries.append({
            "priority": priority,
            "cell": cell,
            "status": "protein_support_audited" if screening_uniprots is not None else "screening_asset_missing_registry_context_only",
            "unique_proteins": int(local["protein_id"].nunique()),
            "author_exact_sequence_match_unique": int(unique_proteins["author_exact_sequence_match"].sum()),
            "protein_id_in_screening_unique": int(unique_proteins["protein_id_in_screening"].sum()),
            "reaction_side_filter_present": reaction_eligible is not None,
            "common_executable_positive_rows": int(len(common)),
            "common_executable_r2e_queries": int(common["reaction_id"].nunique()),
            "common_executable_e2r_queries": int(common["protein_id"].nunique()),
            "exact_sequence_alias_is_executable_support": False,
        })
    return summaries, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance-blind official CLIPZyme protein-support audit for ReactZyme and frozen directed fallback cells.")
    parser.add_argument("--reactzyme-test", type=Path, default=DEFAULT_REACTZYME_TEST)
    parser.add_argument("--reactzyme-reaction-support", type=Path, default=DEFAULT_REACTZYME_REACTION_SUPPORT)
    parser.add_argument("--clip-sequences", type=Path, default=DEFAULT_CLIP_SEQUENCES)
    parser.add_argument("--screening-set", type=Path, default=DEFAULT_SCREENING_SET)
    parser.add_argument("--protein-sequences", type=Path, default=DEFAULT_PROTEIN_SEQUENCES)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--fallback-reaction-support", type=Path, default=DEFAULT_FALLBACK_REACTION_SUPPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clip_sequences_path = args.clip_sequences.resolve()
    id_to_clip_seq, seq_to_clip_ids = load_clip_sequences(clip_sequences_path)
    screening_path = args.screening_set.resolve()
    screening_uniprots, screening_meta = load_screening_uniprots(screening_path)

    reactzyme_test = load_reactzyme_rows(args.reactzyme_test.resolve())
    reactzyme_summary, reactzyme_detail = reactzyme_audit(
        reactzyme_test,
        seq_to_clip_ids=seq_to_clip_ids,
        screening_uniprots=screening_uniprots,
        eligible_reaction_path=args.reactzyme_reaction_support.resolve(),
    )
    fallback_summaries, fallback_details = fallback_audit(
        benchmark_root=args.benchmark_root.resolve(),
        protein_sequences_path=args.protein_sequences.resolve(),
        seq_to_clip_ids=seq_to_clip_ids,
        screening_uniprots=screening_uniprots,
        reaction_support_root=args.fallback_reaction_support.resolve(),
    )

    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    reactzyme_detail.to_csv(out / "reactzyme_protein_support.csv", index=False)
    for cell, detail in fallback_details.items():
        detail.to_csv(out / f"{cell}_protein_support.csv", index=False)
    pd.DataFrame(fallback_summaries).to_csv(out / "fallback_cell_summary.csv", index=False)

    payload = {
        "status": "performance_blind_protein_support_audit_no_model_scores",
        "screening_asset": screening_meta,
        "author_sequence_registry": {
            "path": str(clip_sequences_path),
            "protein_ids": len(id_to_clip_seq),
            "unique_sequences": len(seq_to_clip_ids),
            "sha256": sha256_file(clip_sequences_path),
        },
        "reactzyme": reactzyme_summary,
        "fallback_cells": fallback_summaries,
        "selection_uses_model_scores": False,
        "selection_uses_test_performance": False,
        "identity_boundary": (
            "Only the exact protein identifier in the released CLIPZyme screening set counts as executable precomputed support. "
            "An identical amino-acid sequence under another UniProt ID is reported as context only and is never silently used as a "
            "replacement embedding because CLIPZyme's protein representation also depends on author-associated structure input. "
            "Unsupported proteins may instead be evaluated only through the unchanged official author-native protein inference path."
        ),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
