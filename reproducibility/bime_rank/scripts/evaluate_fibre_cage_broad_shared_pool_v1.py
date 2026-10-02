from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.model.index import FibreCandidateIndex

BENCH = ROOT / "results/broad_rhea_fair_benchmarks_v1/temporal_post2020_double_cold/test_pairs.csv"
CAGE_2023 = ROOT / "data/external/enzymecage_current/rhea_2023_compact.csv.gz"
BROAD_TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
META = ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv"
SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
CAGE_GVP = ROOT / "data/external/enzymecage_current/cage_official_features/gvp_feature/gvp_protein_feature.pt"
CAGE_ESM_NODE = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/pocket_node_feature/esm_node_feature.pt"
CAGE_ESM_MEAN = ROOT / "data/external/enzymecage_current/cage_official_features/esmc600m_max_support_v1/protein_level/seq2feature.pkl"
CAGE_CKPT = ROOT / "external_repos/EnzymeCAGE/checkpoints/pretrain/seed_42/epoch_19.pth"
DEFAULT_OUTPUT = ROOT / "results/fibre_vs_enzymecage_broad_shared_pool_v1"
TOPK = 5000
REPORT_K = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def aliases(frame: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rec in frame.to_dict("records"):
        canonical = str(rec["protein_id"]).strip()
        values = [canonical, str(rec.get("canonical_accession", "")).strip()]
        values.extend(str(rec.get("aliases", "")).split(";"))
        out[canonical] = list(dict.fromkeys(x.strip() for x in values if x.strip()))
    return out


def joint_clean_queries(
    bench: pd.DataFrame,
    alias_map: dict[str, list[str]],
    cage: pd.DataFrame,
    broad_train: pd.DataFrame,
) -> list[str]:
    cage_p = set(cage["UniprotID"].astype(str))
    cage_r = set(cage["reaction_id"].astype(str))
    cage_pair = set(zip(cage["UniprotID"].astype(str), cage["reaction_id"].astype(str)))
    broad_p = set(broad_train["protein_id"].astype(str))
    broad_r = set(broad_train["reaction_id"].astype(str))
    broad_pair = set(zip(broad_train["protein_id"].astype(str), broad_train["reaction_id"].astype(str)))

    def entity_seen(pid: str, universe: set[str]) -> bool:
        return any(a in universe for a in alias_map.get(pid, [pid]))

    def pair_seen(pid: str, rid: str, universe: set[tuple[str, str]]) -> bool:
        return any((a, rid) in universe for a in alias_map.get(pid, [pid]))

    selected: list[str] = []
    for rid, group in bench.groupby("reaction_id", sort=True):
        proteins = group["protein_id"].astype(str).tolist()
        cage_clean = (
            rid not in cage_r
            and not any(entity_seen(p, cage_p) for p in proteins)
            and not any(pair_seen(p, rid, cage_pair) for p in proteins)
        )
        broad_clean = (
            rid not in broad_r
            and not any(entity_seen(p, broad_p) for p in proteins)
            and not any(pair_seen(p, rid, broad_pair) for p in proteins)
        )
        if cage_clean and broad_clean:
            selected.append(str(rid))
    return selected


def cage_supported_aliases(
    alias_map: dict[str, list[str]],
    sequence_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, int]]:
    gvp = torch.load(CAGE_GVP, map_location="cpu", weights_only=False)
    esm = torch.load(CAGE_ESM_NODE, map_location="cpu", weights_only=False)
    valid = {
        uid
        for uid in set(gvp) & set(esm)
        if int(gvp[uid][0].shape[0]) == int(esm[uid].shape[0])
    }
    mean = pkl.load(open(CAGE_ESM_MEAN, "rb"))
    mapping: dict[str, str] = {}
    for pid, values in alias_map.items():
        sequence = sequence_map.get(pid, "")
        if not sequence or sequence not in mean:
            continue
        for value in values:
            if value in valid:
                mapping[pid] = value
                break
    return mapping, {
        "gvp_uids": len(gvp),
        "esm_node_uids": len(esm),
        "valid_structure_uids": len(valid),
        "protein_level_sequence_features": len(mean),
    }


def ranking_metrics(group: pd.DataFrame, score_col: str) -> dict[str, float]:
    order = group.sort_values(
        [score_col, "protein_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    labels = order["Label"].astype(int).to_numpy()
    ranks = np.flatnonzero(labels == 1) + 1
    positive_count = int(labels.sum())
    best = int(ranks[0]) if len(ranks) else 0
    ap = 0.0
    if positive_count:
        ap = float(
            sum(float(labels[:rank].sum()) / float(rank) for rank in ranks)
            / positive_count
        )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(positive_count, 10)))
    dcg = sum(1.0 / math.log2(int(rank) + 1) for rank in ranks if rank <= 10)
    return {
        "candidate_count": int(len(group)),
        "positive_count": positive_count,
        "best_positive_rank": best,
        "reciprocal_rank": 0.0 if best == 0 else 1.0 / best,
        "average_precision": ap,
        "ndcg_at_10": 0.0 if ideal == 0 else float(dcg / ideal),
        "hit_at_3": float(best > 0 and best <= 3),
        "hit_at_10": float(best > 0 and best <= 10),
        "hit_at_20": float(best > 0 and best <= 20),
    }


def summarize(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame[f"{prefix}_reciprocal_rank"].mean()),
        "map": float(frame[f"{prefix}_average_precision"].mean()),
        "ndcg_at_10": float(frame[f"{prefix}_ndcg_at_10"].mean()),
        "hit_at_3": float(frame[f"{prefix}_hit_at_3"].mean()),
        "hit_at_10": float(frame[f"{prefix}_hit_at_10"].mean()),
        "hit_at_20": float(frame[f"{prefix}_hit_at_20"].mean()),
        "median_best_positive_rank": float(
            frame[f"{prefix}_best_positive_rank"].median()
        ),
    }


def prepare(output: Path, device: str) -> None:
    bench = pd.read_csv(BENCH, dtype=str).fillna("")
    cage = pd.read_csv(CAGE_2023, dtype=str).fillna("")
    broad_train = pd.read_csv(BROAD_TRAIN, dtype=str).fillna("")
    meta = pd.read_csv(META, dtype=str).fillna("")
    seq_frame = pd.read_csv(SEQUENCES, sep="\t", dtype=str).fillna("")
    reaction_frame = pd.read_csv(REACTIONS, dtype=str).fillna("")
    alias_map = aliases(meta)
    sequence_map = dict(zip(seq_frame["protein_id"], seq_frame["sequence"]))
    reaction_smiles = dict(
        zip(reaction_frame["reaction_id"], reaction_frame["reaction_smiles"])
    )

    query_ids = joint_clean_queries(bench, alias_map, cage, broad_train)
    if len(query_ids) != 233:
        raise RuntimeError(f"joint-clean query drift: expected 233, got {len(query_ids)}")
    cage_alias, support = cage_supported_aliases(alias_map, sequence_map)

    index = FibreCandidateIndex(device=device)
    candidate_ids = index.protein_ids
    protein_z = index.protein_embeddings
    rows: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []

    for i, rid in enumerate(query_ids, 1):
        positive_ids = set(
            bench.loc[bench["reaction_id"].eq(rid), "protein_id"].astype(str)
        )
        reaction_row = torch.as_tensor(
            [index.reaction_index[rid]], device=index.device, dtype=torch.long
        )
        reaction_z = index.reaction_embeddings.index_select(0, reaction_row)
        with torch.no_grad():
            scores = (reaction_z @ protein_z.T).flatten()
            values, indices = torch.topk(scores, TOPK, largest=True, sorted=True)
        ids = [candidate_ids[int(x)] for x in indices.cpu().tolist()]
        score_values = values.float().cpu().numpy()
        positive_ranks = [
            rank for rank, pid in enumerate(ids, 1) if pid in positive_ids
        ]

        record: dict[str, object] = {
            "reaction_id": rid,
            "positive_count": len(positive_ids),
            # On this jointly clean cohort every positive enzyme is absent from
            # the EnzymeCAGE 2023 association database, so the original
            # similar-reaction gate cannot retrieve a true positive at any K.
            "cage_2023_gate_query_hit": 0,
            "cage_2023_gate_positive_recall": 0.0,
            "broad_best_positive_rank_top5000": (
                min(positive_ranks) if positive_ranks else 0
            ),
            "broad_reciprocal_rank_top5000": (
                1.0 / min(positive_ranks) if positive_ranks else 0.0
            ),
        }
        for k in REPORT_K:
            hit_count = sum(rank <= k for rank in positive_ranks)
            record[f"broad_hit_at_{k}"] = int(hit_count > 0)
            record[f"broad_positive_hits_at_{k}"] = int(hit_count)
            record[f"broad_positive_recall_at_{k}"] = (
                float(hit_count / len(positive_ids)) if positive_ids else 0.0
            )

        supported_count = 0
        supported_positive_count = 0
        for rank, (pid, score) in enumerate(zip(ids, score_values), 1):
            cage_uid = cage_alias.get(pid)
            if not cage_uid:
                continue
            label = int(pid in positive_ids)
            supported_count += 1
            supported_positive_count += label
            rows.append(
                {
                    "reaction_id": rid,
                    "CANO_RXN_SMILES": reaction_smiles[rid],
                    "protein_id": pid,
                    "UniprotID": cage_uid,
                    "sequence": sequence_map[pid],
                    "Label": label,
                    "broad_score": float(score),
                    "broad_rank": rank,
                }
            )
        record["cage_supported_top5000_count"] = supported_count
        record["cage_supported_positive_count"] = supported_positive_count
        record["cage_supported_positive_recall"] = (
            float(supported_positive_count / len(positive_ids))
            if positive_ids
            else 0.0
        )
        queries.append(record)
        if i % 25 == 0:
            print(f"queries={i}/{len(query_ids)}", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    pair_frame = pd.DataFrame(rows)
    query_frame = pd.DataFrame(queries)
    pair_frame.to_csv(output / "broad_top5000_cage_supported_pairs.csv", index=False)
    query_frame.to_csv(output / "broad_query_metrics_pre_cage.csv", index=False)
    evaluable_queries = set(
        query_frame.loc[
            query_frame["cage_supported_positive_count"].gt(0), "reaction_id"
        ].astype(str)
    )
    pair_frame[pair_frame["reaction_id"].isin(evaluable_queries)].to_csv(
        output / "broad_top5000_cage_supported_evaluable_pairs.csv", index=False
    )
    prep = {
        "schema": "fibre-vs-enzymecage-broad-shared-pool-prep-v1",
        "joint_clean_queries": len(query_ids),
        "positive_rows": int(
            bench[bench["reaction_id"].isin(query_ids)].shape[0]
        ),
        "broad_candidate_universe": len(candidate_ids),
        "broad_topk": TOPK,
        "cage_support": support,
        "cage_supported_rows_in_broad_top5000": int(len(pair_frame)),
        "cage_supported_unique_proteins_in_broad_top5000": int(
            pair_frame["protein_id"].nunique()
        ),
        "cage_supported_evaluable_queries": len(evaluable_queries),
        "mean_cage_supported_candidates_per_query": float(
            query_frame["cage_supported_top5000_count"].mean()
        ),
    }
    (output / "prepare_summary.json").write_text(
        json.dumps(prep, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(prep, indent=2))


def final_summary(output: Path, cage_inference: Path) -> None:
    pre = pd.read_csv(output / "broad_query_metrics_pre_cage.csv")
    inference = pd.read_csv(cage_inference)
    required = {
        "reaction_id",
        "protein_id",
        "UniprotID",
        "Label",
        "broad_score",
        "pred_logit",
    }
    if not required <= set(inference.columns):
        raise ValueError(f"CAGE inference missing columns: {sorted(required-set(inference.columns))}")
    if inference.duplicated(["reaction_id", "protein_id"]).any():
        raise ValueError("duplicate shared-pool pair after CAGE inference")

    evaluable = sorted(
        inference.loc[inference["Label"].astype(int).eq(1), "reaction_id"]
        .astype(str)
        .unique()
        .tolist()
    )
    if len(evaluable) != 49:
        raise RuntimeError(f"evaluable-query drift: expected 49, got {len(evaluable)}")

    records: list[dict[str, object]] = []
    for rid, group in inference.groupby("reaction_id", sort=True):
        if rid not in evaluable:
            continue
        broad = ranking_metrics(group, "broad_score")
        cage = ranking_metrics(group, "pred_logit")
        row: dict[str, object] = {"reaction_id": rid}
        row.update({f"broad_{k}": v for k, v in broad.items()})
        row.update({f"cage_{k}": v for k, v in cage.items()})
        records.append(row)
    rank_frame = pd.DataFrame(records)
    rank_frame.to_csv(output / "shared_support_query_metrics.csv", index=False)

    broad_curve = {}
    for k in REPORT_K:
        broad_curve[str(k)] = {
            "query_hit": float(pre[f"broad_hit_at_{k}"].mean()),
            "macro_positive_recall": float(
                pre[f"broad_positive_recall_at_{k}"].mean()
            ),
            "micro_positive_hits": int(pre[f"broad_positive_hits_at_{k}"].sum()),
        }

    summary = {
        "schema": "fibre-vs-enzymecage-broad-shared-pool-v1",
        "status": "completed",
        "protocol": {
            "benchmark": "temporal_post2020_double_cold",
            "joint_clean_rule": (
                "reaction unseen and every positive protein/pair unseen in both "
                "current Broad clean2023 training and EnzymeCAGE 2023 RHEA snapshot"
            ),
            "joint_clean_queries": int(len(pre)),
            "broad_candidate_universe": 185918,
            "broad_shared_pool_topk": TOPK,
            "cage_original_gate": (
                "author 2023 similar-reaction association retrieval; positive recall "
                "is structurally zero on the joint-clean cohort because every positive "
                "protein is absent from its retrievable 2023 association database"
            ),
            "ranking_comparison": (
                "restricted to the exact CAGE-structural-feature-supported subset of "
                "Broad Top-5000 and to queries containing at least one supported positive"
            ),
            "cage_ranking_score": "generic-pretrain seed42 epoch19 pred_logit",
            "no_model_retraining": True,
            "no_parameter_fitting": True,
        },
        "inputs": {
            "benchmark": str(BENCH.relative_to(ROOT)),
            "benchmark_sha256": sha256_file(BENCH),
            "cage_2023": str(CAGE_2023.relative_to(ROOT)),
            "cage_2023_sha256": sha256_file(CAGE_2023),
            "broad_clean2023": str(BROAD_TRAIN.relative_to(ROOT)),
            "broad_clean2023_sha256": sha256_file(BROAD_TRAIN),
            "cage_inference": str(cage_inference.relative_to(ROOT)),
            "cage_inference_sha256": sha256_file(cage_inference),
            "cage_checkpoint": str(CAGE_CKPT.relative_to(ROOT)),
            "cage_checkpoint_sha256": sha256_file(CAGE_CKPT),
        },
        "candidate_generation": {
            "cage_2023_gate_query_hit": 0.0,
            "cage_2023_gate_macro_positive_recall": 0.0,
            "broad": broad_curve,
        },
        "coverage": {
            "cage_global_valid_structure_uids": 8503,
            "cage_global_structure_fraction_of_broad_universe": float(8503 / 185918),
            "mean_cage_supported_candidates_in_broad_top5000": float(
                pre["cage_supported_top5000_count"].mean()
            ),
            "mean_cage_supported_fraction_of_broad_top5000": float(
                pre["cage_supported_top5000_count"].mean() / TOPK
            ),
            "queries_with_supported_positive_in_broad_top5000": int(
                pre["cage_supported_positive_count"].gt(0).sum()
            ),
            "query_fraction_with_supported_positive_in_broad_top5000": float(
                pre["cage_supported_positive_count"].gt(0).mean()
            ),
        },
        "shared_support_ranking": {
            "queries": int(len(rank_frame)),
            "broad": summarize(rank_frame, "broad"),
            "enzymecage": summarize(rank_frame, "cage"),
        },
        "interpretation": {
            "candidate_generation": (
                "The broad learned retriever reaches post-2020 enzymes outside the "
                "EnzymeCAGE 2023 association graph; this is the primary broad-domain "
                "advantage of the current system."
            ),
            "ranking": (
                "The neural ranking result is intentionally conditioned on CAGE feature "
                "availability and must not be used to hide its much narrower structural "
                "coverage."
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "summarize"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cage-inference", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(output, args.device)
        return
    if args.cage_inference is None:
        raise ValueError("--cage-inference is required for summarize")
    final_summary(output, args.cage_inference.resolve())


if __name__ == "__main__":
    main()
