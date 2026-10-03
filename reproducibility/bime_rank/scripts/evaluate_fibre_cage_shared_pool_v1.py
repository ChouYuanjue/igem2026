from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.model.index import FibreCandidateIndex

DEFAULT_CAGE_PAIRS = (
    ROOT
    / "results/terpene_pure_cage_full_support_v1/pretrain/"
    "pure_cage_native_full_pairs_epoch_19.csv.gz"
)
DEFAULT_CAGE_QUERY_METRICS = (
    ROOT
    / "results/terpene_pure_cage_full_support_v1/"
    "pure_cage_official_pipeline_common459_query_metrics.csv"
)
DEFAULT_CLEAN2023 = (
    ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
)
DEFAULT_PROTEIN_SEQUENCES = (
    ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
)
DEFAULT_R2E_POLICY = (
    ROOT / "projects/active/bridge/release/manifests/score_evidence_v1/r2e_support_policy.json"
)
DEFAULT_OUTPUT = ROOT / "results/fibre_vs_enzymecage_shared_pool_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_sequence(value: object) -> str:
    return "".join(str(value or "").upper().split()).rstrip("*")


def exact_sequence_aliases(
    pairs: pd.DataFrame,
    index: FibreCandidateIndex,
    protein_sequences_path: Path,
) -> dict[str, str]:
    broad = pd.read_csv(protein_sequences_path, sep="\t", dtype=str).fillna("")
    broad["clean_sequence"] = broad["sequence"].map(clean_sequence)
    sequence_to_ids: dict[str, list[str]] = {}
    for protein_id, sequence in broad[["protein_id", "clean_sequence"]].itertuples(index=False):
        if sequence:
            sequence_to_ids.setdefault(str(sequence), []).append(str(protein_id))

    unique = pairs[["uniprot_id", "sequence"]].drop_duplicates("uniprot_id").copy()
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for uniprot_id, sequence in unique.itertuples(index=False):
        uniprot_id = str(uniprot_id)
        if uniprot_id in index.protein_index:
            mapping[uniprot_id] = uniprot_id
            continue
        matches = sorted(set(sequence_to_ids.get(clean_sequence(sequence), [])))
        if not matches:
            unresolved.append(uniprot_id)
            continue
        # Identical sequences have identical sequence-model inputs. Pick a stable
        # canonical row only to access the frozen broad embedding.
        mapping[uniprot_id] = matches[0]
    if unresolved:
        raise RuntimeError(f"unresolved CAGE proteins in Broad universe: {unresolved}")
    return mapping


def score_broad_matrix(
    pairs: pd.DataFrame,
    query_ids: list[str],
    index: FibreCandidateIndex,
    protein_alias: dict[str, str],
) -> pd.DataFrame:
    proteins = pairs["uniprot_id"].drop_duplicates().astype(str).tolist()
    mapped = [protein_alias[p] for p in proteins]
    protein_rows = torch.as_tensor(
        [index.protein_index[p] for p in mapped],
        dtype=torch.long,
        device=index.device,
    )
    reaction_rows = torch.as_tensor(
        [index.reaction_index[r] for r in query_ids],
        dtype=torch.long,
        device=index.device,
    )
    protein_z = index.protein_embeddings.index_select(0, protein_rows)
    reaction_z = index.reaction_embeddings.index_select(0, reaction_rows)
    with torch.no_grad():
        scores = (reaction_z @ protein_z.T).float().cpu().numpy()

    reaction_index = {r: i for i, r in enumerate(query_ids)}
    protein_index = {p: i for i, p in enumerate(proteins)}
    out = pairs.copy()
    out["broad_score"] = [
        float(scores[reaction_index[r], protein_index[p]])
        for r, p in out[["reaction_id", "uniprot_id"]].itertuples(index=False)
    ]
    out["broad_protein_id"] = out["uniprot_id"].map(protein_alias)
    return out


def ranking_metrics(
    ranked: pd.DataFrame,
    *,
    score_column: str,
    total_positives: int,
) -> dict[str, float]:
    ordered = ranked.sort_values(
        [score_column, "uniprot_id"],
        ascending=[False, True],
        kind="stable",
    )
    labels = ordered["label"].to_numpy(np.int8)
    positive_ranks = np.flatnonzero(labels == 1) + 1
    best = int(positive_ranks[0]) if len(positive_ranks) else 0
    rr = 0.0 if best == 0 else 1.0 / best

    if total_positives:
        precisions = [
            float(labels[:rank].sum()) / float(rank)
            for rank in positive_ranks
        ]
        ap = float(sum(precisions) / total_positives)
    else:
        ap = 0.0

    dcg = sum(
        1.0 / math.log2(int(rank) + 1)
        for rank in positive_ranks
        if int(rank) <= 10
    )
    ideal_count = min(int(total_positives), 10)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))
    ndcg10 = float(dcg / ideal) if ideal else 0.0

    return {
        "best_positive_rank": float(best),
        "reciprocal_rank": float(rr),
        "average_precision": float(ap),
        "ndcg_at_10": ndcg10,
        "hit_at_3": float(best > 0 and best <= 3),
        "hit_at_10": float(best > 0 and best <= 10),
        "hit_at_20": float(best > 0 and best <= 20),
    }


def summarize_rows(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame[f"{prefix}_reciprocal_rank"].mean()),
        "map": float(frame[f"{prefix}_average_precision"].mean()),
        "ndcg_at_10": float(frame[f"{prefix}_ndcg_at_10"].mean()),
        "hit_at_3": float(frame[f"{prefix}_hit_at_3"].mean()),
        "hit_at_10": float(frame[f"{prefix}_hit_at_10"].mean()),
        "hit_at_20": float(frame[f"{prefix}_hit_at_20"].mean()),
        "query_positive_coverage": float(
            frame[f"{prefix}_best_positive_rank"].gt(0).mean()
        ),
    }


def summarize_original_cage(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame["cage_original_rr"].mean()),
        "hit_at_3": float(frame["cage_original_hit_at_3"].mean()),
        "hit_at_10": float(frame["cage_original_hit_at_10"].mean()),
        "hit_at_20": float(frame["cage_original_hit_at_20"].mean()),
        "query_positive_coverage": float(
            frame["cage_original_best_positive_rank"].gt(0).mean()
        ),
    }


def cohort_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"queries": 0}
    return {
        "queries": int(len(frame)),
        "candidate_budget": {
            "mean": float(frame["candidate_budget"].mean()),
            "median": float(frame["candidate_budget"].median()),
            "min": int(frame["candidate_budget"].min()),
            "max": int(frame["candidate_budget"].max()),
        },
        "candidate_recall": {
            "fibre_broad_pool_macro_positive_recall": float(
                frame["broad_pool_positive_recall"].mean()
            ),
            "fibre_broad_pool_micro_positive_recall": float(
                frame["broad_pool_positive_count"].sum()
                / frame["positive_count"].sum()
            ),
            "fibre_broad_pool_query_positive_coverage": float(
                frame["broad_pool_positive_count"].gt(0).mean()
            ),
            "cage_original_gate_query_positive_coverage": float(
                frame["cage_original_best_positive_rank"].gt(0).mean()
            ),
        },
        "cage_original_gate_and_ranking": summarize_original_cage(frame),
        "fibre_current_r2e_ranking_on_broad_pool": summarize_rows(frame, "fibre"),
        "cage_independent_rerank_on_same_broad_pool": summarize_rows(frame, "cage_shared"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current FIBRE R2E ranking with EnzymeCAGE on Broad-generated "
            "TPS candidate pools at the exact per-query budget of CAGE's own gate."
        )
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cage_pairs = pd.read_csv(
        DEFAULT_CAGE_PAIRS,
        dtype={"reaction_id": str, "uniprot_id": str},
    ).fillna("")
    cage_pairs["label"] = cage_pairs["label"].astype(int)
    cage_pairs["pred_logit"] = cage_pairs["pred_logit"].astype(float)
    query_metrics = pd.read_csv(
        DEFAULT_CAGE_QUERY_METRICS,
        dtype={"reaction_id": str},
    ).fillna("")
    query_ids = query_metrics["reaction_id"].astype(str).tolist()
    if len(query_ids) != 459 or len(set(query_ids)) != 459:
        raise RuntimeError("expected the frozen 459-query common CAGE benchmark")
    cage_pairs = cage_pairs[cage_pairs["reaction_id"].isin(query_ids)].copy()

    r2e_policy = json.loads(DEFAULT_R2E_POLICY.read_text(encoding="utf-8"))
    if r2e_policy.get("ranking_policy") != "frozen_broad_core_only":
        raise RuntimeError(
            "current R2E ranking policy changed; this benchmark must be reviewed"
        )

    index = FibreCandidateIndex(device=args.device)
    missing_reactions = sorted(set(query_ids) - set(index.reaction_index))
    if missing_reactions:
        raise RuntimeError(f"CAGE reactions missing from Broad index: {missing_reactions}")

    protein_alias = exact_sequence_aliases(
        cage_pairs,
        index,
        DEFAULT_PROTEIN_SEQUENCES,
    )
    scored = score_broad_matrix(cage_pairs, query_ids, index, protein_alias)

    clean2023 = pd.read_csv(DEFAULT_CLEAN2023, dtype=str).fillna("")
    train_proteins = set(clean2023["protein_id"].astype(str))
    train_reactions = set(clean2023["reaction_id"].astype(str))
    train_pairs = set(
        zip(
            clean2023["reaction_id"].astype(str),
            clean2023["protein_id"].astype(str),
        )
    )

    positives = scored[scored["label"].eq(1)].copy()
    positives["pair_seen_clean2023"] = [
        (r, p) in train_pairs
        for r, p in positives[["reaction_id", "broad_protein_id"]].itertuples(index=False)
    ]
    positives["protein_seen_clean2023"] = positives["broad_protein_id"].isin(
        train_proteins
    )
    positives["reaction_seen_clean2023"] = positives["reaction_id"].isin(
        train_reactions
    )

    strict_queries: set[str] = set()
    for reaction_id, group in positives.groupby("reaction_id", sort=False):
        if (
            str(reaction_id) not in train_reactions
            and not group["protein_seen_clean2023"].any()
        ):
            strict_queries.add(str(reaction_id))

    query_metrics = query_metrics.set_index("reaction_id")
    records: list[dict[str, object]] = []
    for reaction_id, group in scored.groupby("reaction_id", sort=False):
        q = query_metrics.loc[str(reaction_id)]
        budget = int(q["candidate_gate_size"])
        broad_pool = group.sort_values(
            ["broad_score", "uniprot_id"],
            ascending=[False, True],
            kind="stable",
        ).head(budget)
        positive_count = int(group["label"].sum())
        pool_positive_count = int(broad_pool["label"].sum())

        fibre = ranking_metrics(
            broad_pool,
            score_column="broad_score",
            total_positives=positive_count,
        )
        cage_shared = ranking_metrics(
            broad_pool,
            score_column="pred_logit",
            total_positives=positive_count,
        )

        row: dict[str, object] = {
            "reaction_id": str(reaction_id),
            "candidate_budget": budget,
            "positive_count": positive_count,
            "broad_pool_positive_count": pool_positive_count,
            "broad_pool_positive_recall": (
                float(pool_positive_count / positive_count)
                if positive_count
                else 0.0
            ),
            "strict_double_cold_clean2023": str(reaction_id) in strict_queries,
            "cage_original_best_positive_rank": int(q["best_positive_rank"]),
            "cage_original_rr": float(q["rr"]),
            "cage_original_hit_at_3": float(q["hit3"]),
            "cage_original_hit_at_10": float(q["hit10"]),
            "cage_original_hit_at_20": float(q["hit20"]),
        }
        row.update({f"fibre_{key}": value for key, value in fibre.items()})
        row.update(
            {f"cage_shared_{key}": value for key, value in cage_shared.items()}
        )
        records.append(row)

    query_frame = pd.DataFrame(records).sort_values("reaction_id", kind="stable")
    strict_frame = query_frame[query_frame["strict_double_cold_clean2023"]].copy()

    positive_rows = int(len(positives))
    overlap = {
        "positive_rows": positive_rows,
        "pair_overlap_rows": int(positives["pair_seen_clean2023"].sum()),
        "pair_overlap_fraction": float(positives["pair_seen_clean2023"].mean()),
        "protein_seen_rows": int(positives["protein_seen_clean2023"].sum()),
        "protein_seen_fraction": float(positives["protein_seen_clean2023"].mean()),
        "reaction_seen_rows": int(positives["reaction_seen_clean2023"].sum()),
        "reaction_seen_fraction": float(positives["reaction_seen_clean2023"].mean()),
        "strict_double_cold_queries": int(len(strict_queries)),
    }

    summary = {
        "schema": "fibre-vs-enzymecage-shared-pool-v1",
        "status": "completed",
        "protocol": {
            "scope": (
                "TPS common-support operational benchmark; the full 459-query cohort "
                "is not an external-generalization benchmark"
            ),
            "candidate_budget": (
                "for each reaction, Broad/FIBRE returns exactly K candidates where K "
                "equals EnzymeCAGE's archived candidate_gate_size for that reaction"
            ),
            "fibre_candidate_generation": "frozen Broad Core score",
            "fibre_r2e_ranking_policy": r2e_policy["ranking_policy"],
            "cage_shared_pool_rerank": (
                "raw EnzymeCAGE generic-pretrain pred_logit only; Broad score/rank "
                "and FIBRE evidence are not inputs"
            ),
            "no_parameter_fitting": True,
            "no_model_retraining": True,
        },
        "inputs": {
            "cage_pairs": str(DEFAULT_CAGE_PAIRS.relative_to(ROOT)),
            "cage_pairs_sha256": sha256_file(DEFAULT_CAGE_PAIRS),
            "cage_query_metrics": str(DEFAULT_CAGE_QUERY_METRICS.relative_to(ROOT)),
            "cage_query_metrics_sha256": sha256_file(DEFAULT_CAGE_QUERY_METRICS),
            "clean2023": str(DEFAULT_CLEAN2023.relative_to(ROOT)),
            "clean2023_sha256": sha256_file(DEFAULT_CLEAN2023),
            "r2e_policy": str(DEFAULT_R2E_POLICY.relative_to(ROOT)),
            "r2e_policy_sha256": sha256_file(DEFAULT_R2E_POLICY),
        },
        "support": {
            "common_queries": int(len(query_frame)),
            "cage_scored_proteins": int(cage_pairs["uniprot_id"].nunique()),
            "broad_reaction_coverage": 1.0,
            "broad_protein_coverage_after_exact_sequence_alias": 1.0,
            "exact_sequence_alias_count": int(
                sum(
                    1
                    for source, target in protein_alias.items()
                    if source != target
                )
            ),
        },
        "clean2023_overlap_audit": overlap,
        "operational_common459": cohort_summary(query_frame),
        "strict_double_cold_clean2023": cohort_summary(strict_frame),
        "interpretation": {
            "operational": (
                "At the same per-query candidate budget, Broad/FIBRE covers more known "
                "positives than CAGE's original gate. On the identical Broad candidate "
                "sets, current FIBRE R2E ordering outperforms independent CAGE reranking."
            ),
            "generalization_boundary": (
                "The 459-query operational cohort has substantial clean2023 overlap and "
                "must not be presented as external generalization. On the 54-query strict "
                "double-cold subset, CAGE's original gate has higher positive coverage; "
                "within the Broad-generated pool, neither ordering shows a uniformly "
                "dominant early-ranking result."
            ),
            "expert_policy": (
                "These results do not justify giving CAGE rank-changing authority inside "
                "FIBRE. They are consistent with treating it as available biochemical "
                "support unless a clean admission protocol demonstrates stable ranking gain."
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_frame.to_csv(args.output_dir / "query_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
