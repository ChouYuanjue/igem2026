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
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_broad_shared_pool_v1 import (
    aliases,
    cage_supported_aliases,
)

BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
OUT = ROOT / "results/fibre_vs_enzymecage_layered_full7_v1"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
REACTION_ASSET_ROOT = (
    ROOT / "results/enzymecage_reaction_family_response_v1/full_outer_assets"
)
CAGE_CKPT_DIR = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
CAGE_RESULT_DIR = OUT / "cage_top1000_inference"
TOPK = 1000


def load_cells():
    cells = {}
    for d in sorted(BENCH.iterdir()):
        p = d / "test_pairs.csv"
        if not p.exists():
            continue
        frame = pd.read_csv(p, dtype=str).fillna("")
        cells[d.name] = {
            str(q): set(g["protein_id"].astype(str))
            for q, g in frame.groupby("reaction_id", sort=False)
        }
    return cells


def load_cage_alias(broad_seq):
    meta = pd.read_csv(META, dtype=str).fillna("")
    return cage_supported_aliases(aliases(meta), broad_seq)


def prepare(device: str):
    cells = load_cells()
    by_query = defaultdict(list)
    for cell, qmap in cells.items():
        for q, pos in qmap.items():
            by_query[q].append((cell, pos))

    seq_frame = pd.read_csv(SEQUENCES, sep="	", dtype=str).fillna("")
    broad_seq = dict(
        zip(
            seq_frame["protein_id"].astype(str),
            seq_frame["sequence"].astype(str),
        )
    )
    cage_alias, support = load_cage_alias(broad_seq)
    valid_reactions = set(
        pd.read_csv(
            REACTION_ASSET_ROOT / "probe_strict.csv",
            usecols=["reaction_id"],
            dtype=str,
        )["reaction_id"].astype(str)
    )
    rx = pd.read_csv(REACTIONS, dtype=str).fillna("")
    reaction_smiles = dict(
        zip(rx["reaction_id"].astype(str), rx["reaction_smiles"].astype(str))
    )

    index = FibreCandidateIndex(device=device)
    queries = sorted(by_query)
    recall_rows = []
    scoreable_rows = []
    union: dict[tuple[str, str], dict[str, str]] = {}

    for start in range(0, len(queries), 128):
        batch_q = queries[start : start + 128]
        qrows = torch.as_tensor(
            [index.reaction_index[q] for q in batch_q],
            dtype=torch.long,
            device=index.device,
        )
        with torch.no_grad():
            score = (
                index.reaction_embeddings.index_select(0, qrows)
                @ index.protein_embeddings.T
            ).float()
            values, rows = torch.topk(
                score, k=TOPK, dim=1, largest=True, sorted=True
            )
        values = values.cpu().numpy()
        rows = rows.cpu().numpy()

        for i, q in enumerate(batch_q):
            pids = [index.protein_ids[int(r)] for r in rows[i]]
            scoreable = [
                (pid, cage_alias[pid])
                for pid in pids
                if pid in cage_alias
            ]
            reaction_ok = q in valid_reactions
            any_scoreable_positive = False
            for cell, positives in by_query[q]:
                hits = [pid for pid in pids if pid in positives]
                scoreable_hits = [
                    (pid, uid)
                    for pid, uid in scoreable
                    if pid in positives
                ] if reaction_ok else []
                if scoreable_hits:
                    any_scoreable_positive = True
                    scoreable_rows.extend(
                        {
                            "cell": cell,
                            "reaction_id": q,
                            "protein_id": pid,
                            "candidate_uid": uid,
                            "label": int(pid in positives),
                        }
                        for pid, uid in scoreable
                    )
                recall_rows.append(
                    {
                        "cell": cell,
                        "reaction_id": q,
                        "positive_count": len(positives),
                        "broad_top1000_positive_count": len(hits),
                        "broad_top1000_query_hit": int(bool(hits)),
                        "broad_top1000_positive_recall": (
                            len(hits) / len(positives) if positives else 0.0
                        ),
                        "broad_top1000_cage_scoreable_count": (
                            len(scoreable) if reaction_ok else 0
                        ),
                        "broad_top1000_cage_scoreable_positive_count": len(
                            scoreable_hits
                        ),
                        "broad_top1000_cage_scoreable_query_hit": int(
                            bool(scoreable_hits)
                        ),
                        "reaction_scoreable_by_cage": int(reaction_ok),
                    }
                )

            if any_scoreable_positive and reaction_ok:
                for pid, uid in scoreable:
                    union.setdefault(
                        (q, uid),
                        {
                            "reaction_id": q,
                            "CANO_RXN_SMILES": reaction_smiles[q],
                            "UniprotID": uid,
                            "sequence": broad_seq[pid],
                            "Label": "0",
                        },
                    )
        print(
            f"top1000_queries={min(start+128,len(queries))}/{len(queries)}",
            flush=True,
        )

    recall = pd.DataFrame(recall_rows)
    recall.to_csv(OUT / "broad_top1000_recall_query_instances.csv", index=False)
    pd.DataFrame(scoreable_rows).to_csv(
        OUT / "broad_top1000_cage_scoreable_candidates.csv.gz", index=False
    )
    pairs = pd.DataFrame(union.values())
    pairs.to_csv(OUT / "cage_top1000_score_union_pairs.csv", index=False)

    config = yaml.safe_load(
        (REACTION_ASSET_ROOT / "probe.yaml").read_text()
    )
    config["data_path"] = str(
        (OUT / "cage_top1000_score_union_pairs.csv").resolve()
    )
    config["batch_size"] = 256
    config["ckpt_dir"] = str(CAGE_CKPT_DIR.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str(CAGE_RESULT_DIR.resolve())
    config_path = OUT / "cage_top1000_generic_infer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def sm(g):
        return {
            "instances": int(len(g)),
            "query_hit": float(g["broad_top1000_query_hit"].mean()),
            "macro_positive_recall": float(
                g["broad_top1000_positive_recall"].mean()
            ),
            "cage_scoreable_query_hit": float(
                g["broad_top1000_cage_scoreable_query_hit"].mean()
            ),
        }

    summary = {
        "schema": "fibre-broad-top1000-cage-prep-v1",
        "status": "prepared",
        "unique_queries": len(queries),
        "cell_query_instances": len(recall),
        "overall": sm(recall),
        "per_cell": {
            cell: sm(g) for cell, g in recall.groupby("cell", sort=True)
        },
        "cage_native_support": support,
        "cage_scoring_pairs": int(len(pairs)),
        "cage_scoring_queries": int(
            pairs["reaction_id"].nunique() if len(pairs) else 0
        ),
        "inference_config": str(config_path.relative_to(ROOT)),
        "protocol": (
            "Broad Core Top-1000 candidates exactly as the deployed general "
            "candidate pool; generic CAGE ranks only its native-scoreable subset; "
            "Broad scores are not provided to CAGE."
        ),
    }
    (OUT / "broad_top1000_prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    prepare(args.device)


if __name__ == "__main__":
    main()
