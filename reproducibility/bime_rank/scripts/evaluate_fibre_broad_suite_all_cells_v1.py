from __future__ import annotations

import argparse
import json
import pickle as pkl
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.model.index import FibreCandidateIndex

BENCH_ROOT = ROOT / "results/broad_rhea_fair_benchmarks_v1"
CAGE_2023 = ROOT / "data/external/enzymecage_current/rhea_2023_compact.csv.gz"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
CAGE_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
CAGE_ESM_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
CAGE_ESM_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
DEFAULT_OUTPUT = ROOT / "results/fibre_broad_suite_all_cells_v1"
REPORT_K = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def aliases(frame: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rec in frame.to_dict("records"):
        pid = str(rec["protein_id"]).strip()
        vals = [pid, str(rec.get("canonical_accession", "")).strip()]
        vals.extend(str(rec.get("aliases", "")).split(";"))
        out[pid] = list(dict.fromkeys(v.strip() for v in vals if v.strip()))
    return out


def load_cells() -> dict[str, pd.DataFrame]:
    cells: dict[str, pd.DataFrame] = {}
    for path in sorted(BENCH_ROOT.iterdir()):
        test = path / "test_pairs.csv"
        if not test.exists():
            continue
        frame = pd.read_csv(test, dtype=str).fillna("")
        frame = frame[["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)
        cells[path.name] = frame
    return cells


def cage_assets(alias_map: dict[str, list[str]], sequence_map: dict[str, str]):
    cage = pd.read_csv(CAGE_2023, dtype=str).fillna("")
    cage_p = set(cage["UniprotID"].astype(str))
    cage_r = set(cage["reaction_id"].astype(str))
    cage_pair = set(zip(cage["UniprotID"].astype(str), cage["reaction_id"].astype(str)))

    gvp = torch.load(CAGE_GVP, map_location="cpu", weights_only=False)
    esm = torch.load(CAGE_ESM_NODE, map_location="cpu", weights_only=False)
    valid_structure = {
        uid for uid in set(gvp) & set(esm)
        if int(gvp[uid][0].shape[0]) == int(esm[uid].shape[0])
    }
    mean = pkl.load(open(CAGE_ESM_MEAN, "rb"))
    canonical_to_cage: dict[str, str] = {}
    for pid, vals in alias_map.items():
        seq = sequence_map.get(pid, "")
        if not seq or seq not in mean:
            continue
        for alias in vals:
            if alias in valid_structure:
                canonical_to_cage[pid] = alias
                break

    def protein_seen(pid: str) -> bool:
        return any(alias in cage_p for alias in alias_map.get(pid, [pid]))

    def pair_seen(pid: str, rid: str) -> bool:
        return any((alias, rid) in cage_pair for alias in alias_map.get(pid, [pid]))

    return cage_p, cage_r, cage_pair, canonical_to_cage, protein_seen, pair_seen, {
        "cage_2023_proteins": len(cage_p),
        "cage_2023_reactions": len(cage_r),
        "cage_2023_pairs": len(cage_pair),
        "cage_valid_structure_uids": len(valid_structure),
        "cage_protein_level_sequences": len(mean),
    }


def summarize_cell(frame: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {
        "query_instances": int(len(frame)),
        "positive_pairs": int(frame["positive_count"].sum()),
        "mean_positive_count": float(frame["positive_count"].mean()),
        "cage_2023_protein_seen_positive_fraction": float(
            frame["cage_protein_seen_positive_count"].sum() / max(frame["positive_count"].sum(), 1)
        ),
        "cage_2023_exact_pair_seen_positive_fraction": float(
            frame["cage_pair_seen_positive_count"].sum() / max(frame["positive_count"].sum(), 1)
        ),
        "cage_2023_reaction_seen_query_fraction": float(frame["cage_reaction_seen"].mean()),
        "joint_clean_query_count": int(frame["joint_clean_query"].sum()),
        "joint_clean_query_fraction": float(frame["joint_clean_query"].mean()),
        "cage_structure_supported_positive_fraction": float(
            frame["cage_structure_supported_positive_count"].sum() / max(frame["positive_count"].sum(), 1)
        ),
        "cage_structure_supported_query_fraction": float(
            frame["cage_structure_supported_positive_count"].gt(0).mean()
        ),
        "mean_cage_supported_candidates_in_broad_top5000": float(
            frame["cage_supported_top5000_count"].mean()
        ),
        "query_fraction_with_cage_supported_positive_in_broad_top5000": float(
            frame["cage_supported_positive_top5000_count"].gt(0).mean()
        ),
    }
    for k in REPORT_K:
        out[f"broad_hit_at_{k}"] = float(frame[f"broad_hit_at_{k}"].mean())
        out[f"broad_macro_positive_recall_at_{k}"] = float(
            frame[f"broad_positive_recall_at_{k}"].mean()
        )
        out[f"broad_micro_positive_recall_at_{k}"] = float(
            frame[f"broad_positive_hits_at_{k}"].sum() / max(frame["positive_count"].sum(), 1)
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    args = parser.parse_args()

    cells = load_cells()
    meta = pd.read_csv(META, dtype=str).fillna("")
    alias_map = aliases(meta)
    seq_frame = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    sequence_map = dict(zip(seq_frame["protein_id"].astype(str), seq_frame["sequence"].astype(str)))
    _, cage_r, _, canonical_to_cage, protein_seen, pair_seen, cage_summary = cage_assets(
        alias_map, sequence_map
    )

    positives: dict[str, dict[str, set[str]]] = defaultdict(dict)
    all_queries: set[str] = set()
    for cell, frame in cells.items():
        for rid, group in frame.groupby("reaction_id"):
            positives[str(rid)][cell] = set(group["protein_id"].astype(str))
            all_queries.add(str(rid))
    queries = sorted(all_queries)

    index = FibreCandidateIndex(device=args.device)
    missing_reactions = [rid for rid in queries if rid not in index.reaction_index]
    if missing_reactions:
        raise RuntimeError(f"{len(missing_reactions)} benchmark reactions missing from current Broad index")

    pids = index.protein_ids
    pid_to_col = index.protein_index
    protein_z = index.protein_embeddings
    cage_supported_mask = torch.as_tensor(
        [pid in canonical_to_cage for pid in pids],
        dtype=torch.bool,
        device=index.device,
    )

    records: list[dict[str, object]] = []
    batch_size = int(args.batch_size)
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rrows = torch.as_tensor(
            [index.reaction_index[rid] for rid in batch],
            dtype=torch.long,
            device=index.device,
        )
        rz = index.reaction_embeddings.index_select(0, rrows)
        with torch.no_grad():
            scores = rz @ protein_z.T
            top_values, top_indices = torch.topk(scores, k=max(REPORT_K), dim=1, largest=True, sorted=True)
        top_indices_cpu = top_indices.cpu().numpy()

        for bi, rid in enumerate(batch):
            top_ids = [pids[int(j)] for j in top_indices_cpu[bi]]
            cage_supported_top5000 = sum(pid in canonical_to_cage for pid in top_ids)
            top_rank = {pid: rank for rank, pid in enumerate(top_ids, 1)}
            for cell, pos in positives[rid].items():
                ranks = sorted(top_rank[pid] for pid in pos if pid in top_rank)
                protein_seen_count = sum(protein_seen(pid) for pid in pos)
                pair_seen_count = sum(pair_seen(pid, rid) for pid in pos)
                structure_supported_count = sum(pid in canonical_to_cage for pid in pos)
                structure_supported_top5000_count = sum(
                    pid in canonical_to_cage and pid in top_rank for pid in pos
                )
                row: dict[str, object] = {
                    "cell": cell,
                    "reaction_id": rid,
                    "positive_count": len(pos),
                    "cage_reaction_seen": int(rid in cage_r),
                    "cage_protein_seen_positive_count": int(protein_seen_count),
                    "cage_pair_seen_positive_count": int(pair_seen_count),
                    "joint_clean_query": int(
                        rid not in cage_r and protein_seen_count == 0 and pair_seen_count == 0
                    ),
                    "cage_structure_supported_positive_count": int(structure_supported_count),
                    "cage_supported_top5000_count": int(cage_supported_top5000),
                    "cage_supported_positive_top5000_count": int(
                        structure_supported_top5000_count
                    ),
                    "broad_best_positive_rank_top5000": int(min(ranks)) if ranks else 0,
                }
                for k in REPORT_K:
                    hits = sum(rank <= k for rank in ranks)
                    row[f"broad_hit_at_{k}"] = int(hits > 0)
                    row[f"broad_positive_hits_at_{k}"] = int(hits)
                    row[f"broad_positive_recall_at_{k}"] = float(hits / len(pos))
                records.append(row)
        print(f"queries={min(start+batch_size,len(queries))}/{len(queries)}", flush=True)

    query_metrics = pd.DataFrame(records).sort_values(["cell", "reaction_id"]).reset_index(drop=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    query_metrics.to_csv(output / "query_metrics.csv", index=False)

    cell_summaries = {
        cell: summarize_cell(query_metrics[query_metrics["cell"].eq(cell)].copy())
        for cell in sorted(cells)
    }
    overall = summarize_cell(query_metrics.copy())
    pair_union = pd.concat(
        [frame.assign(cell=cell) for cell, frame in cells.items()],
        ignore_index=True,
    )
    suite = {
        "schema": "fibre-broad-suite-all-cells-v1",
        "status": "completed",
        "current_broad_candidate_universe": len(pids),
        "cells": len(cells),
        "cell_query_instances": int(sum(frame["reaction_id"].nunique() for frame in cells.values())),
        "cell_positive_pairs": int(sum(len(frame) for frame in cells.values())),
        "unique_reaction_queries": int(pair_union["reaction_id"].nunique()),
        "unique_positive_pairs": int(
            len(pair_union[["protein_id", "reaction_id"]].drop_duplicates())
        ),
        "unique_positive_proteins": int(pair_union["protein_id"].nunique()),
        "cage_assets": cage_summary,
        "metrics_semantics": {
            "broad": "current FibreCandidateIndex frozen Broad Core, full 185918-protein candidate universe, no target-specific fitting",
            "cage_2023_protein_seen": "generous association-graph reach upper bound: positive enzyme exists anywhere in EnzymeCAGE 2023 RHEA association database",
            "cage_2023_exact_pair_seen": "positive enzyme-reaction association itself is present in EnzymeCAGE 2023 database",
            "cage_structure_supported": "positive protein has complete GVP + ESM-node + protein-level ESM assets required for EnzymeCAGE neural inference",
            "joint_clean": "query reaction unseen by CAGE 2023 and every positive protein/pair unseen by CAGE 2023",
        },
        "overall_cell_weighted": overall,
        "per_cell": cell_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(suite, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
