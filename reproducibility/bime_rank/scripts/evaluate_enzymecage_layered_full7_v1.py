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

from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_broad_shared_pool_v1 import (
    aliases,
    cage_supported_aliases,
)

BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1"
GATE_ROOT = ROOT / "results/enzymecage_original_gate_full7_v1"
OUT = ROOT / "results/fibre_vs_enzymecage_layered_full7_v1"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
AUTHOR_UID_SEQ = (
    ROOT
    / "data/external/enzymecage_current/orphan335_author_assets_v1/"
    "uid_sequences.tsv.gz"
)
CAGE_GVP = (
    ROOT
    / "data/external/enzymecage_current/cage_official_features/"
    "gvp_feature/gvp_protein_feature.pt"
)
CAGE_ESM_NODE = (
    ROOT
    / "data/external/enzymecage_current/cage_official_features/"
    "esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
)
CAGE_ESM_MEAN = (
    ROOT
    / "data/external/enzymecage_current/cage_official_features/"
    "esmc600m_max_support_v1/protein_level/seq2feature.pkl"
)
REACTION_ASSET_ROOT = (
    ROOT
    / "results/enzymecage_reaction_family_response_v1/full_outer_assets"
)
CAGE_CKPT_DIR = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42"
CAGE_RESULT_DIR = OUT / "cage_inference"


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


def positive_aliases(
    cells: dict[str, dict[str, set[str]]],
    alias_map: dict[str, list[str]],
):
    out = {}
    for cell, qmap in cells.items():
        for q, positives in qmap.items():
            a = set()
            for pid in positives:
                a.update(alias_map.get(pid, [pid]))
            out[(cell, q)] = a
    return out


def load_native_cage_support():
    gvp = torch.load(CAGE_GVP, map_location="cpu", weights_only=False)
    node = torch.load(CAGE_ESM_NODE, map_location="cpu", weights_only=False)
    valid = {
        str(uid)
        for uid in set(gvp) & set(node)
        if int(gvp[uid][0].shape[0]) == int(node[uid].shape[0])
    }
    mean = pkl.load(open(CAGE_ESM_MEAN, "rb"))
    return valid, mean


def prepare(device: str):
    OUT.mkdir(parents=True, exist_ok=True)
    gate_q = pd.read_csv(
        GATE_ROOT / "query_gate.csv",
        dtype={"cell": str, "reaction_id": str},
    ).fillna("")
    gate = pd.read_csv(
        GATE_ROOT / "gate_candidates.csv.gz",
        dtype={"cell": str, "reaction_id": str, "candidate_uid": str},
    ).fillna("")
    cells = load_cells()

    expected_instances = sum(len(x) for x in cells.values())
    if len(gate_q) != expected_instances:
        raise RuntimeError(
            f"gate query-instance drift: {len(gate_q)} != {expected_instances}"
        )

    meta = pd.read_csv(META, dtype=str).fillna("")
    alias_map = aliases(meta)
    pos_alias = positive_aliases(cells, alias_map)
    broad_sequences = pd.read_csv(SEQUENCES, sep="	", dtype=str).fillna("")
    broad_seq = dict(
        zip(
            broad_sequences["protein_id"].astype(str),
            broad_sequences["sequence"].astype(str),
        )
    )
    author_seq_frame = pd.read_csv(AUTHOR_UID_SEQ, sep="	", dtype=str).fillna("")
    author_seq = dict(
        zip(
            author_seq_frame["UniprotID"].astype(str),
            author_seq_frame["sequence"].astype(str),
        )
    )
    reaction_frame = pd.read_csv(REACTIONS, dtype=str).fillna("")
    reaction_smiles = dict(
        zip(
            reaction_frame["reaction_id"].astype(str),
            reaction_frame["reaction_smiles"].astype(str),
        )
    )

    native_valid, mean_feature = load_native_cage_support()
    cage_alias, support_audit = cage_supported_aliases(alias_map, broad_seq)
    valid_reactions = set(
        pd.read_csv(
            REACTION_ASSET_ROOT / "probe_strict.csv",
            usecols=["reaction_id"],
            dtype=str,
        )["reaction_id"].astype(str)
    )

    # Build a single max-K Broad pool per query, then slice it separately for
    # each cell according to that cell's original CAGE gate candidate count.
    index = FibreCandidateIndex(device=device)
    max_k_by_query = (
        gate_q.groupby("reaction_id", sort=False)["candidate_count"]
        .apply(lambda s: int(pd.to_numeric(s).max()))
        .to_dict()
    )
    broad_by_query = {}
    query_ids = sorted(max_k_by_query)
    for i, q in enumerate(query_ids, 1):
        k = int(max_k_by_query[q])
        if k <= 0:
            broad_by_query[q] = (np.empty(0, dtype=np.int64), np.empty(0))
            continue
        qr = index.reaction_index.get(q)
        if qr is None:
            broad_by_query[q] = (np.empty(0, dtype=np.int64), np.empty(0))
            continue
        with torch.no_grad():
            score = (
                index.reaction_embeddings[qr] @ index.protein_embeddings.T
            ).float()
            values, rows = torch.topk(
                score, k=min(k, len(index.protein_ids)), largest=True, sorted=True
            )
        broad_by_query[q] = (
            rows.cpu().numpy().astype(np.int64),
            values.cpu().numpy().astype(np.float32),
        )
        if i % 250 == 0:
            print(f"broad_candidates={i}/{len(query_ids)}", flush=True)

    gate_groups = {
        (str(cell), str(q)): g
        for (cell, q), g in gate.groupby(["cell", "reaction_id"], sort=False)
    }

    broad_rows = []
    recall_rows = []
    score_pairs: dict[tuple[str, str], dict[str, str]] = {}
    original_scoreable_rows = []
    broad_scoreable_rows = []

    for rec in gate_q.itertuples(index=False):
        cell = str(rec.cell)
        q = str(rec.reaction_id)
        k = int(rec.candidate_count)
        positives = cells[cell][q]
        aliases_for_pos = pos_alias[(cell, q)]
        reaction_ok = q in valid_reactions

        og = gate_groups.get((cell, q))
        original_uids = [] if og is None else list(
            dict.fromkeys(og["candidate_uid"].astype(str).tolist())
        )
        original_uid_set = set(original_uids)
        original_hit_pids = [
            pid for pid in positives
            if original_uid_set & set(alias_map.get(pid, [pid]))
        ]

        rows, values = broad_by_query[q]
        rows = rows[:k]
        values = values[:k]
        broad_ids = [index.protein_ids[int(r)] for r in rows]
        broad_hit_ids = [p for p in broad_ids if p in positives]

        original_scoreable = []
        original_scoreable_positive = []
        original_scoreable_hit_pids = []
        if reaction_ok:
            for uid in original_uids:
                seq = author_seq.get(uid, "")
                if (
                    uid in native_valid
                    and seq
                    and seq in mean_feature
                ):
                    original_scoreable.append(uid)
                    if uid in aliases_for_pos:
                        original_scoreable_positive.append(uid)
            original_scoreable_set = set(original_scoreable)
            original_scoreable_hit_pids = [
                pid for pid in positives
                if original_scoreable_set & set(alias_map.get(pid, [pid]))
            ]
            if original_scoreable_positive:
                for uid in original_scoreable:
                    key = (q, uid)
                    score_pairs.setdefault(
                        key,
                        {
                            "reaction_id": q,
                            "CANO_RXN_SMILES": reaction_smiles[q],
                            "UniprotID": uid,
                            "sequence": author_seq[uid],
                            "Label": "0",
                        },
                    )

        broad_scoreable = []
        broad_scoreable_positive = []
        if reaction_ok:
            for rank, (pid, score) in enumerate(zip(broad_ids, values), 1):
                uid = cage_alias.get(pid)
                broad_rows.append(
                    {
                        "cell": cell,
                        "reaction_id": q,
                        "protein_id": pid,
                        "broad_rank": rank,
                        "broad_score": float(score),
                        "cage_uid": uid or "",
                        "cage_scoreable": int(bool(uid)),
                        "label": int(pid in positives),
                    }
                )
                if uid:
                    broad_scoreable.append((pid, uid))
                    if pid in positives:
                        broad_scoreable_positive.append((pid, uid))
            if broad_scoreable_positive:
                for pid, uid in broad_scoreable:
                    key = (q, uid)
                    score_pairs.setdefault(
                        key,
                        {
                            "reaction_id": q,
                            "CANO_RXN_SMILES": reaction_smiles[q],
                            "UniprotID": uid,
                            "sequence": broad_seq[pid],
                            "Label": "0",
                        },
                    )
        else:
            for rank, (pid, score) in enumerate(zip(broad_ids, values), 1):
                broad_rows.append(
                    {
                        "cell": cell,
                        "reaction_id": q,
                        "protein_id": pid,
                        "broad_rank": rank,
                        "broad_score": float(score),
                        "cage_uid": "",
                        "cage_scoreable": 0,
                        "label": int(pid in positives),
                    }
                )

        recall_rows.append(
            {
                "cell": cell,
                "reaction_id": q,
                "positive_count": len(positives),
                "cage_gate_status": str(rec.status),
                "candidate_budget": k,
                "cage_gate_positive_count": len(original_hit_pids),
                "cage_gate_query_hit": int(bool(original_hit_pids)),
                "cage_gate_positive_recall": (
                    len(original_hit_pids) / len(positives) if positives else 0.0
                ),
                "cage_gate_scoreable_count": len(original_scoreable),
                "cage_gate_scoreable_positive_count": len(
                    original_scoreable_hit_pids
                ),
                "cage_gate_scoreable_query_hit": int(
                    bool(original_scoreable_hit_pids)
                ),
                "broad_gate_positive_count": len(broad_hit_ids),
                "broad_gate_query_hit": int(bool(broad_hit_ids)),
                "broad_gate_positive_recall": (
                    len(broad_hit_ids) / len(positives) if positives else 0.0
                ),
                "broad_gate_scoreable_count": len(broad_scoreable),
                "broad_gate_scoreable_positive_count": len(
                    broad_scoreable_positive
                ),
                "broad_gate_scoreable_query_hit": int(
                    bool(broad_scoreable_positive)
                ),
                "reaction_scoreable_by_cage": int(reaction_ok),
            }
        )

        if original_scoreable_positive:
            original_scoreable_rows.extend(
                {
                    "cell": cell,
                    "reaction_id": q,
                    "candidate_uid": uid,
                    "label": int(uid in aliases_for_pos),
                }
                for uid in original_scoreable
            )
        if broad_scoreable_positive:
            broad_scoreable_rows.extend(
                {
                    "cell": cell,
                    "reaction_id": q,
                    "protein_id": pid,
                    "candidate_uid": uid,
                    "label": int(pid in positives),
                }
                for pid, uid in broad_scoreable
            )

    recall = pd.DataFrame(recall_rows)
    recall.to_csv(OUT / "gate_recall_query_instances.csv", index=False)
    pd.DataFrame(broad_rows).to_csv(
        OUT / "broad_matched_budget_candidates.csv.gz", index=False
    )
    pd.DataFrame(original_scoreable_rows).to_csv(
        OUT / "original_gate_scoreable_candidates.csv.gz", index=False
    )
    pd.DataFrame(broad_scoreable_rows).to_csv(
        OUT / "broad_gate_scoreable_candidates.csv.gz", index=False
    )

    pair_frame = pd.DataFrame(score_pairs.values())
    pair_frame.to_csv(OUT / "cage_score_union_pairs.csv", index=False)

    config = yaml.safe_load(
        (REACTION_ASSET_ROOT / "probe.yaml").read_text()
    )
    config["data_path"] = str((OUT / "cage_score_union_pairs.csv").resolve())
    config["batch_size"] = 256
    config["ckpt_dir"] = str(CAGE_CKPT_DIR.resolve())
    config["model_list"] = ["epoch_19.pth"]
    config["result_dir"] = str(CAGE_RESULT_DIR.resolve())
    config_path = OUT / "cage_generic_infer.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def rec_summary(frame):
        return {
            "instances": int(len(frame)),
            "cage_gate_query_hit": float(frame["cage_gate_query_hit"].mean()),
            "cage_gate_macro_positive_recall": float(
                frame["cage_gate_positive_recall"].mean()
            ),
            "cage_gate_scoreable_query_hit": float(
                frame["cage_gate_scoreable_query_hit"].mean()
            ),
            "broad_gate_query_hit": float(frame["broad_gate_query_hit"].mean()),
            "broad_gate_macro_positive_recall": float(
                frame["broad_gate_positive_recall"].mean()
            ),
            "broad_gate_scoreable_query_hit": float(
                frame["broad_gate_scoreable_query_hit"].mean()
            ),
            "mean_candidate_budget": float(frame["candidate_budget"].mean()),
            "median_candidate_budget": float(frame["candidate_budget"].median()),
        }

    summary = {
        "schema": "fibre-vs-enzymecage-layered-full7-prepare-v1",
        "status": "prepared",
        "unique_queries": int(gate_q["reaction_id"].nunique()),
        "cell_query_instances": int(len(recall)),
        "overall_recall": rec_summary(recall),
        "per_cell_recall": {
            cell: rec_summary(g)
            for cell, g in recall.groupby("cell", sort=True)
        },
        "cage_native_support": {
            **support_audit,
            "author_uid_sequences": int(len(author_seq)),
            "reaction_feature_supported_queries": int(len(valid_reactions)),
        },
        "cage_union_scoring_pairs": int(len(pair_frame)),
        "cage_union_scoring_queries": int(
            pair_frame["reaction_id"].nunique() if len(pair_frame) else 0
        ),
        "inference_config": str(config_path.relative_to(ROOT)),
        "protocol": {
            "original_cage": (
                "published CAGE Top-10 similar-reaction gate per cell, followed "
                "by generic-pretrain EnzymeCAGE ranking on native-scoreable candidates"
            ),
            "broad_gate_cage": (
                "for each cell-query, Broad Core returns exactly the same number "
                "of unique candidates as that query's original CAGE gate; CAGE "
                "then independently ranks only its native-scoreable subset"
            ),
            "broad_score_used_by_cage_ranker": False,
            "seven_cell_labels_used_for_training": False,
            "no_parameter_fitting": True,
        },
    }
    (OUT / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))



def rank_metrics(rows: list[tuple[str, float, int]]) -> dict[str, float]:
    if not rows:
        return {
            "best_positive_rank": 0,
            "rr": 0.0,
            "hit10": 0.0,
            "hit100": 0.0,
        }
    ordered = sorted(rows, key=lambda x: (-float(x[1]), str(x[0])))
    positives = [i for i, (_, _, y) in enumerate(ordered, 1) if int(y) == 1]
    best = min(positives) if positives else 0
    return {
        "best_positive_rank": int(best),
        "rr": 0.0 if best == 0 else 1.0 / best,
        "hit10": float(best > 0 and best <= 10),
        "hit100": float(best > 0 and best <= 100),
    }


def summarize_block(frame: pd.DataFrame) -> dict[str, object]:
    n = len(frame)
    if n == 0:
        return {"instances": 0}
    out = {"instances": int(n), "unique_queries": int(frame["reaction_id"].nunique())}
    for prefix in ("original_cage", "broad_gate_cage", "full_system"):
        out[prefix] = {
            "mrr": float(frame[f"{prefix}_rr"].mean()),
            "hit10": float(frame[f"{prefix}_hit10"].mean()),
            "hit100": float(frame[f"{prefix}_hit100"].mean()),
            "query_positive_coverage": float(
                frame[f"{prefix}_best_positive_rank"].gt(0).mean()
            ),
        }
    out["gate_recall"] = {
        "original_cage_query_hit": float(frame["cage_gate_query_hit"].mean()),
        "original_cage_macro_positive_recall": float(
            frame["cage_gate_positive_recall"].mean()
        ),
        "broad_matched_query_hit": float(frame["broad_gate_query_hit"].mean()),
        "broad_matched_macro_positive_recall": float(
            frame["broad_gate_positive_recall"].mean()
        ),
        "original_cage_scoreable_query_hit": float(
            frame["cage_gate_scoreable_query_hit"].mean()
        ),
        "broad_matched_scoreable_query_hit": float(
            frame["broad_gate_scoreable_query_hit"].mean()
        ),
    }
    return out


def add_cage_novelty_flags(frame: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(META, dtype=str).fillna("")
    alias_map = aliases(meta)
    cage = pd.read_csv(
        ROOT / "data/external/enzymecage_current/rhea_2023_compact.csv.gz",
        usecols=["UniprotID", "reaction_id"],
        dtype=str,
    ).fillna("")
    cage_uids = set(cage["UniprotID"].astype(str))
    cage_pairs = set(
        zip(cage["reaction_id"].astype(str), cage["UniprotID"].astype(str))
    )
    cells = load_cells()
    flags = []
    for cell, qmap in cells.items():
        for q, positives in qmap.items():
            protein_seen = [
                any(a in cage_uids for a in alias_map.get(pid, [pid]))
                for pid in positives
            ]
            pair_seen = [
                any((q, a) in cage_pairs for a in alias_map.get(pid, [pid]))
                for pid in positives
            ]
            flags.append(
                {
                    "cell": cell,
                    "reaction_id": q,
                    "all_positive_proteins_unseen_by_cage": not any(protein_seen),
                    "any_positive_protein_unseen_by_cage": not all(protein_seen),
                    "all_positive_pairs_unseen_by_cage": not any(pair_seen),
                    "any_positive_pair_unseen_by_cage": not all(pair_seen),
                }
            )
    flags = pd.DataFrame(flags)
    return frame.merge(flags, on=["cell", "reaction_id"], how="left", validate="one_to_one")


def summarize():
    recall = pd.read_csv(
        OUT / "gate_recall_query_instances.csv",
        dtype={"cell": str, "reaction_id": str},
    ).fillna("")
    inference_path = CAGE_RESULT_DIR / "cage_score_union_pairs_epoch_19.csv"
    inference = pd.read_csv(
        inference_path,
        dtype={"reaction_id": str, "UniprotID": str},
    ).fillna("")
    inference["pred_logit"] = pd.to_numeric(inference["pred_logit"])
    score = {
        (str(q), str(uid)): float(v)
        for q, uid, v in inference[
            ["reaction_id", "UniprotID", "pred_logit"]
        ].itertuples(index=False)
    }

    original = pd.read_csv(
        OUT / "original_gate_scoreable_candidates.csv.gz",
        dtype={"cell": str, "reaction_id": str, "candidate_uid": str},
    ).fillna("")
    broad = pd.read_csv(
        OUT / "broad_gate_scoreable_candidates.csv.gz",
        dtype={
            "cell": str,
            "reaction_id": str,
            "protein_id": str,
            "candidate_uid": str,
        },
    ).fillna("")

    original_metrics = {}
    for (cell, q), g in original.groupby(["cell", "reaction_id"], sort=False):
        rows = []
        for uid, label in g[["candidate_uid", "label"]].itertuples(index=False):
            v = score.get((str(q), str(uid)))
            if v is not None:
                rows.append((str(uid), v, int(label)))
        original_metrics[(str(cell), str(q))] = rank_metrics(rows)

    broad_metrics = {}
    for (cell, q), g in broad.groupby(["cell", "reaction_id"], sort=False):
        rows = []
        for pid, uid, label in g[
            ["protein_id", "candidate_uid", "label"]
        ].itertuples(index=False):
            v = score.get((str(q), str(uid)))
            if v is not None:
                rows.append((str(pid), v, int(label)))
        broad_metrics[(str(cell), str(q))] = rank_metrics(rows)

    final = pd.read_csv(
        ROOT / "results/fibre_final_integrated_experts_full_outer/query_metrics.csv",
        dtype={"cell": str, "query_id": str},
    ).fillna("")
    final_key = {
        (str(c), str(q)): int(r)
        for c, q, r in final[
            ["cell", "query_id", "final_best_rank"]
        ].itertuples(index=False)
    }

    records = []
    zero = rank_metrics([])
    for r in recall.to_dict("records"):
        key = (str(r["cell"]), str(r["reaction_id"]))
        om = original_metrics.get(key, zero)
        bm = broad_metrics.get(key, zero)
        fr = int(final_key[key])
        fm = {
            "best_positive_rank": fr,
            "rr": 1.0 / fr,
            "hit10": float(fr <= 10),
            "hit100": float(fr <= 100),
        }
        row = dict(r)
        for prefix, metrics in (
            ("original_cage", om),
            ("broad_gate_cage", bm),
            ("full_system", fm),
        ):
            for name, value in metrics.items():
                row[f"{prefix}_{name}"] = value
        records.append(row)

    frame = pd.DataFrame(records)
    frame = add_cage_novelty_flags(frame)
    frame.to_csv(OUT / "layered_query_instances.csv", index=False)

    subsets = {
        "all_seven_cells": np.ones(len(frame), dtype=bool),
        "all_positive_proteins_unseen_by_cage": frame[
            "all_positive_proteins_unseen_by_cage"
        ].astype(bool).to_numpy(),
        "all_positive_pairs_unseen_by_cage": frame[
            "all_positive_pairs_unseen_by_cage"
        ].astype(bool).to_numpy(),
        "any_positive_protein_unseen_by_cage": frame[
            "any_positive_protein_unseen_by_cage"
        ].astype(bool).to_numpy(),
        "any_positive_pair_unseen_by_cage": frame[
            "any_positive_pair_unseen_by_cage"
        ].astype(bool).to_numpy(),
    }
    summary = {
        "schema": "fibre-vs-enzymecage-layered-full7-v1",
        "status": "completed",
        "headline_protocol": (
            "Original EnzymeCAGE versus matched-budget Broad Core candidate "
            "generation followed by the same generic EnzymeCAGE ranker, then "
            "the full integrated FIBRE system on the fixed seven-cell test."
        ),
        "overall": summarize_block(frame),
        "per_cell": {
            cell: summarize_block(g)
            for cell, g in frame.groupby("cell", sort=True)
        },
        "cross_cell_novelty_subsets": {
            name: summarize_block(frame[mask].copy())
            for name, mask in subsets.items()
            if name != "all_seven_cells"
        },
        "coverage_boundary": {
            "cage_retrieval_unavailable_unique_queries": 3,
            "cage_native_structure_uids": 8503,
            "candidate_generation_recall_reported_before_structure_filter": True,
            "ranking_metrics_include_all_query_instances": True,
            "unretrieved_or_unscoreable_positive_queries_receive_zero_rr_hit": True,
        },
        "inference_pairs": int(len(inference)),
        "generic_cage_checkpoint": (
            "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42/epoch_19.pth"
        ),
        "broad_score_used_by_cage_ranker": False,
        "labels_used_for_fitting": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "summarize"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare(args.device)
    else:
        summarize()


if __name__ == "__main__":
    main()
