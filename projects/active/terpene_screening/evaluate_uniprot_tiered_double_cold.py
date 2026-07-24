from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.analyze_uniprot_expansion_quality import (  # noqa: E402
    pfam_architecture,
)
from projects.active.terpene_screening.evaluate_uniprot_expanded_double_cold import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_LONG_MODELS,
    DEFAULT_SHORT_MODELS,
    aggregate,
    ensemble_scores,
    load_fold_models,
    normalize_rows,
)
from projects.active.terpene_screening.rank_open_world import load_protein_library  # noqa: E402
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics  # noqa: E402

DEFAULT_UNIPROT = ROOT / "data/terpene_embeddings/uniprot_tps_primary_esmc600m"
DEFAULT_METADATA = (
    ROOT / "data/terpene_uniprot_expansion/uniprot_tps_primary_embedding_candidates.tsv"
)
DEFAULT_MARTS = ROOT / "data/terpene_marts/marts_reaction_pairs.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_uniprot_tiered_double_cold"
DEFAULT_CONTRACTS = (
    ROOT
    / "data/terpene_uniprot_expansion/reaction_architecture_contracts/reaction_architecture_contracts.csv"
)
DEFAULT_BUDGETS = (3, 10, 20)

TIER_GROUPS = {
    "ab": {"A_reviewed", "B_experimental_or_transcript_named"},
    "abc": {
        "A_reviewed",
        "B_experimental_or_transcript_named",
        "C_homology_named",
    },
    "abcd": {
        "A_reviewed",
        "B_experimental_or_transcript_named",
        "C_homology_named",
        "D_named_predicted",
    },
}


def load_uniprot_metadata(path: Path, ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    frame = frame.rename(columns={"accession": "candidate_id"}).drop_duplicates("candidate_id")
    ordered = pd.DataFrame({"candidate_id": ids}).merge(frame, on="candidate_id", how="left")
    if ordered["evidence_quality_tier"].eq("").any():
        missing = ordered.loc[ordered["evidence_quality_tier"].eq(""), "candidate_id"].tolist()
        raise ValueError(f"Missing UniProt metadata: {missing[:20]}")
    return ordered


def reaction_contract_map(
    cache_dir: Path, contracts_path: Path
) -> tuple[dict[str, set[str]], dict[str, str]]:
    reactions = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    contracts = pd.read_csv(contracts_path, dtype=str).fillna("")
    merged = reactions[["reaction_id", "reaction_signature"]].merge(
        contracts[
            [
                "reaction_signature",
                "allowed_candidate_architectures",
                "contract_status",
            ]
        ].drop_duplicates("reaction_signature"),
        on="reaction_signature",
        how="left",
    ).fillna("")
    allowed = {
        str(row.reaction_id): {
            value
            for value in str(row.allowed_candidate_architectures).split(";")
            if value
        }
        for row in merged.itertuples(index=False)
    }
    status = dict(
        zip(
            merged["reaction_id"].astype(str),
            merged["contract_status"].replace("", "contract_missing"),
        )
    )
    return allowed, status


def build_universe_indices(
    canonical_count: int,
    metadata: pd.DataFrame,
    allowed_architectures: set[str],
) -> dict[str, np.ndarray]:
    canonical = np.arange(canonical_count, dtype=np.int64)
    tiers = metadata["evidence_quality_tier"].astype(str)
    architectures = metadata["pfam_architecture"].astype(str)
    contract_compatible = architectures.isin(allowed_architectures).to_numpy()
    universes = {"canonical": canonical}
    for label, accepted in TIER_GROUPS.items():
        tier_mask = tiers.isin(accepted).to_numpy()
        all_local = np.flatnonzero(tier_mask)
        contract_local = np.flatnonzero(tier_mask & contract_compatible)
        universes[f"expanded_{label}"] = np.concatenate(
            [canonical, canonical_count + all_local]
        )
        universes[f"expanded_{label}_contract"] = np.concatenate(
            [canonical, canonical_count + contract_local]
        )
    return universes


def generalized_pairing(
    frame: pd.DataFrame, contract_supported_only: bool = False
) -> pd.DataFrame:
    working = frame[frame["contract_supported"]].copy() if contract_supported_only else frame
    rows = []
    for budget, budget_frame in working.groupby("budget", sort=True):
        canonical = budget_frame[
            budget_frame["candidate_universe"].eq("canonical")
        ].set_index(["split_id", "query_id"])
        for universe, group in budget_frame.groupby("candidate_universe", sort=True):
            if universe == "canonical":
                continue
            candidate = group.set_index(["split_id", "query_id"])
            aligned = canonical[
                ["hit", "best_positive_rank", "reciprocal_rank"]
            ].join(
                candidate[
                    [
                        "hit",
                        "best_positive_rank",
                        "reciprocal_rank",
                        "added_uniprot_count",
                    ]
                ],
                lsuffix="_canonical",
                rsuffix="_candidate",
                how="inner",
            )
            canonical_hits = aligned["hit_canonical"].astype(int)
            candidate_hits = aligned["hit_candidate"].astype(int)
            retained = int(((canonical_hits == 1) & (candidate_hits == 1)).sum())
            lost = int(((canonical_hits == 1) & (candidate_hits == 0)).sum())
            rows.append(
                {
                    "budget": int(budget),
                    "candidate_universe": universe,
                    "scope": (
                        "contract_supported_queries"
                        if contract_supported_only
                        else "all_strict_queries"
                    ),
                    "n_queries": len(aligned),
                    "canonical_hits": int(canonical_hits.sum()),
                    "candidate_hits": int(candidate_hits.sum()),
                    "retained_hits": retained,
                    "hits_lost": lost,
                    "hit_retention_fraction": (
                        retained / canonical_hits.sum() if canonical_hits.sum() else np.nan
                    ),
                    "median_positive_rank_inflation": float(
                        (
                            aligned["best_positive_rank_candidate"]
                            - aligned["best_positive_rank_canonical"]
                        ).median()
                    ),
                    "mean_positive_rank_inflation": float(
                        (
                            aligned["best_positive_rank_candidate"]
                            - aligned["best_positive_rank_canonical"]
                        ).mean()
                    ),
                    "mrr_ratio_to_canonical": float(
                        aligned["reciprocal_rank_candidate"].mean()
                        / aligned["reciprocal_rank_canonical"].mean()
                    ),
                    "median_added_uniprot_count": float(
                        aligned["added_uniprot_count"].median()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict double-cold UniProt evidence-tier and family-compatibility stress test."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--uniprot-protein-dir", type=Path, default=DEFAULT_UNIPROT)
    parser.add_argument("--uniprot-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--short-model-dir", type=Path, default=DEFAULT_SHORT_MODELS)
    parser.add_argument("--long-model-dir", type=Path, default=DEFAULT_LONG_MODELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    canonical_features = normalize_rows(np.load(cache_dir / "protein_features.npy").astype(np.float32))
    reaction_features = np.load(cache_dir / "reaction_features.npy").astype(np.float32)
    proteins = pd.read_csv(cache_dir / "protein_entities.csv", dtype=str).fillna("")
    reactions = pd.read_csv(cache_dir / "reaction_entities.csv", dtype=str).fillna("")
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    pairs["protein_seen"] = pairs["protein_seen"].astype(str).str.lower().eq("true")
    pairs["reaction_seen"] = pairs["reaction_seen"].astype(str).str.lower().eq("true")
    canonical_ids = proteins["protein_id"].astype(str).tolist()
    reaction_ids = reactions["reaction_id"].astype(str).tolist()
    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    allowed_by_reaction, contract_status_by_reaction = reaction_contract_map(
        cache_dir, args.contracts.resolve()
    )

    uniprot_features, uniprot_ids = load_protein_library(args.uniprot_protein_dir.resolve())
    metadata = load_uniprot_metadata(args.uniprot_metadata.resolve(), uniprot_ids)
    metadata["pfam_architecture"] = metadata["pfam_combination"].map(
        pfam_architecture
    )
    all_features = np.concatenate([canonical_features, uniprot_features], axis=0)
    all_ids = canonical_ids + uniprot_ids
    canonical_count = len(canonical_ids)
    records: list[dict[str, object]] = []

    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test = pairs[
                (pairs["protein_fold"] == protein_fold)
                & (pairs["reaction_fold"] == reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            if test.empty:
                continue
            score_cache: dict[str, np.ndarray] = {}
            for budget, model_dir in [
                (3, args.short_model_dir.resolve()),
                (10, args.short_model_dir.resolve()),
                (20, args.long_model_dir.resolve()),
            ]:
                cache_key = str(model_dir)
                if cache_key not in score_cache:
                    models = load_fold_models(model_dir, split_id, device)
                    score_cache[cache_key] = ensemble_scores(
                        models, all_features, reaction_features, device
                    )
                scores = score_cache[cache_key]
                for reaction_id, group in test.groupby("rhea_id", sort=True):
                    positives = set(group["Entry"].astype(str))
                    reaction_row = reaction_to_row[reaction_id]
                    query_scores = scores[reaction_row]
                    allowed_architectures = allowed_by_reaction.get(reaction_id, set())
                    contract_status = contract_status_by_reaction.get(
                        reaction_id, "contract_missing"
                    )
                    universes = build_universe_indices(
                        canonical_count, metadata, allowed_architectures
                    )
                    for universe, indices in universes.items():
                        candidate_ids = [all_ids[int(index)] for index in indices]
                        metrics = rank_metrics(
                            query_scores[indices],
                            candidate_ids,
                            positives,
                            set(),
                            (budget,),
                        )
                        order = np.lexsort((np.asarray(candidate_ids), -query_scores[indices]))
                        added = indices >= canonical_count
                        best_rank = int(metrics["best_positive_rank"])
                        added_above = int(
                            np.sum(added[order[: max(best_rank - 1, 0)]])
                        )
                        records.append(
                            {
                                "split_id": split_id,
                                "query_id": reaction_id,
                                "architecture_contract_status": contract_status,
                                "allowed_candidate_architectures": ";".join(
                                    sorted(allowed_architectures)
                                ),
                                "contract_supported": bool(allowed_architectures),
                                "budget": budget,
                                "candidate_universe": universe,
                                "candidate_count": len(indices),
                                "added_uniprot_count": int(np.sum(added)),
                                "n_positives": len(positives),
                                "hit": int(metrics[f"hit_at_{budget}"]),
                                "positive_recall": float(metrics[f"positive_recall_at_{budget}"]),
                                "best_positive_rank": best_rank,
                                "reciprocal_rank": float(metrics["reciprocal_rank"]),
                                "top1_is_uniprot": bool(added[order[0]]),
                                "uniprot_above_best_positive": added_above,
                            }
                        )

    query_metrics = pd.DataFrame(records)
    metrics = aggregate(query_metrics)
    paired = generalized_pairing(query_metrics)
    supported_paired = generalized_pairing(
        query_metrics, contract_supported_only=True
    )
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    paired.to_csv(output_dir / "paired_retention.csv", index=False)
    supported_paired.to_csv(
        output_dir / "paired_retention_contract_supported_queries.csv", index=False
    )
    tier_counts = {
        label: int(metadata["evidence_quality_tier"].isin(accepted).sum())
        for label, accepted in TIER_GROUPS.items()
    }
    summary = {
        "strict_external_double_cold": True,
        "canonical_candidate_count": canonical_count,
        "uniprot_tier_counts": tier_counts,
        "architecture_compatibility_is_known_positive_contract_based": True,
        "contract_supported_test_queries": int(
            query_metrics.loc[query_metrics["contract_supported"], "query_id"].nunique()
        ),
        "contract_unsupported_test_queries": int(
            query_metrics.loc[~query_metrics["contract_supported"], "query_id"].nunique()
        ),
        "metrics": metrics.to_dict("records"),
        "paired_retention": paired.to_dict("records"),
        "contract_supported_query_retention": supported_paired.to_dict("records"),
        "outputs": {
            "query_metrics": str(output_dir / "query_metrics.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "paired": str(output_dir / "paired_retention.csv"),
            "supported_paired": str(
                output_dir / "paired_retention_contract_supported_queries.csv"
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
