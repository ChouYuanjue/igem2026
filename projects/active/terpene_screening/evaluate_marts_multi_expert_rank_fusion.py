from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_model_rank_fusion_double_cold import (  # noqa: E402
    split_partition,
)

DEFAULT_CACHE = ROOT / "data/terpene_marts_adaptation"
DEFAULT_DUAL = ROOT / "results/terpene_marts_dual_rankings_v1/rankings.csv"
DEFAULT_MULTI = ROOT / "results/terpene_multi_expert_marts_rankings_v1/rankings.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_marts_multi_expert_rank_fusion"
DEFAULT_BUDGETS = (3, 10, 20)


@dataclass(frozen=True)
class FusionSpec:
    name: str
    kind: str
    sources: tuple[str, ...]
    weights: tuple[float, ...] = ()
    constant: float = 0.0
    power: float = 1.0
    rescue_slots: int = 0


def parse_float_tuple(value: str) -> tuple[float, ...]:
    result = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one float")
    return result


def parse_int_tuple(value: str) -> tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result:
        raise ValueError("Expected at least one integer")
    return result


def build_specs(
    dual_sources: list[str],
    multi_source: str,
    weights: tuple[float, ...],
    constants: tuple[float, ...],
    powers: tuple[float, ...],
    rescue_slots: tuple[int, ...],
    include_triples: bool,
) -> list[FusionSpec]:
    specs = [
        FusionSpec(name=f"single__{source}", kind="single", sources=(source,))
        for source in [*dual_sources, multi_source]
    ]
    for primary in dual_sources:
        for slots in rescue_slots:
            specs.append(
                FusionSpec(
                    name=f"rescue__{primary}__{multi_source}__q{slots}",
                    kind="rescue",
                    sources=(primary, multi_source),
                    rescue_slots=slots,
                )
            )
        for weight in weights:
            for constant in constants:
                for power in powers:
                    specs.append(
                        FusionSpec(
                            name=(
                                f"rrf__{primary}_{weight:g}__{multi_source}_{1-weight:g}"
                                f"__c{constant:g}__p{power:g}"
                            ),
                            kind="rrf",
                            sources=(primary, multi_source),
                            weights=(weight, 1.0 - weight),
                            constant=constant,
                            power=power,
                        )
                    )
    if include_triples:
        for index, left in enumerate(dual_sources):
            for right in dual_sources[index + 1 :]:
                for constant in constants:
                    for power in powers:
                        specs.append(
                            FusionSpec(
                                name=(
                                    f"triple__{left}__{right}__{multi_source}"
                                    f"__c{constant:g}__p{power:g}"
                                ),
                                kind="rrf",
                                sources=(left, right, multi_source),
                                weights=(0.4, 0.4, 0.2),
                                constant=constant,
                                power=power,
                            )
                        )
    return specs


def load_rankings(
    dual_path: Path,
    multi_path: Path,
    multi_method: str,
    multi_source: str,
) -> pd.DataFrame:
    dual = pd.read_csv(dual_path, dtype=str).fillna("")
    for column in ("rank", "score", "is_positive"):
        dual[column] = pd.to_numeric(dual[column])
    multi = pd.read_csv(multi_path, dtype=str).fillna("")
    multi = multi[multi["method"].eq(multi_method)].copy()
    if multi.empty:
        raise ValueError(f"No rows found for multi-expert method: {multi_method}")
    multi["source"] = multi_source
    for column in ("rank", "score", "is_positive"):
        multi[column] = pd.to_numeric(multi[column])
    columns = [
        "source",
        "split_id",
        "direction",
        "query_id",
        "rank",
        "candidate_id",
        "score",
        "is_positive",
    ]
    combined = pd.concat([dual[columns], multi[columns]], ignore_index=True)
    keys = ["source", "split_id", "direction", "query_id"]
    if combined.duplicated(keys + ["candidate_id"]).any():
        raise ValueError("A source ranking contains duplicate candidates")
    depths = combined.groupby(keys)["rank"].max()
    if depths.nunique() != 1:
        raise ValueError(f"Ranking sources use inconsistent depths: {depths.value_counts().to_dict()}")
    return combined


def load_queries(cache_dir: Path) -> pd.DataFrame:
    pairs = pd.read_csv(cache_dir / "marts_pair_folds.csv", dtype=str).fillna("")
    pairs["protein_fold"] = pd.to_numeric(pairs["protein_fold"]).astype(int)
    pairs["reaction_fold"] = pd.to_numeric(pairs["reaction_fold"]).astype(int)
    for column in ("protein_seen", "reaction_seen"):
        pairs[column] = pairs[column].astype(str).str.lower().eq("true")
    rows: list[dict[str, object]] = []
    for protein_fold in range(5):
        for reaction_fold in range(5):
            split_id = f"p{protein_fold}_r{reaction_fold}"
            test = pairs[
                pairs["protein_fold"].eq(protein_fold)
                & pairs["reaction_fold"].eq(reaction_fold)
                & (~pairs["protein_seen"])
                & (~pairs["reaction_seen"])
            ]
            for reaction_id, group in test.groupby("rhea_id", sort=True):
                rows.append(
                    {
                        "split_id": split_id,
                        "partition": split_partition(split_id, 4),
                        "direction": "reaction_to_enzyme",
                        "query_id": str(reaction_id),
                        "positives": frozenset(group["Entry"].astype(str)),
                    }
                )
            for protein_id, group in test.groupby("Entry", sort=True):
                rows.append(
                    {
                        "split_id": split_id,
                        "partition": split_partition(split_id, 4),
                        "direction": "enzyme_to_reaction",
                        "query_id": str(protein_id),
                        "positives": frozenset(group["rhea_id"].astype(str)),
                    }
                )
    return pd.DataFrame(rows)


def ranking_maps(rankings: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, list[str]]]:
    result: dict[tuple[str, str, str], dict[str, list[str]]] = {}
    keys = ["split_id", "direction", "query_id"]
    for query_key, query_group in rankings.groupby(keys, sort=False):
        local: dict[str, list[str]] = {}
        for source, source_group in query_group.groupby("source", sort=False):
            local[str(source)] = (
                source_group.sort_values(["rank", "candidate_id"])["candidate_id"]
                .astype(str)
                .tolist()
            )
        result[tuple(map(str, query_key))] = local
    return result


def fuse_ranking(source_rankings: dict[str, list[str]], spec: FusionSpec, budget: int) -> list[str]:
    if spec.kind == "single":
        return source_rankings[spec.sources[0]][:budget]
    if spec.kind == "rescue":
        primary = source_rankings[spec.sources[0]]
        rescue = source_rankings[spec.sources[1]]
        slots = min(spec.rescue_slots, budget)
        selected = list(primary[: budget - slots])
        used = set(selected)
        for candidate in rescue:
            if len(selected) >= budget:
                break
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
        for candidate in primary:
            if len(selected) >= budget:
                break
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
        return selected
    if spec.kind != "rrf":
        raise ValueError(f"Unknown fusion kind: {spec.kind}")
    score: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for source, weight in zip(spec.sources, spec.weights, strict=True):
        for rank, candidate in enumerate(source_rankings[source], start=1):
            score[candidate] = score.get(candidate, 0.0) + weight / (
                (spec.constant + rank) ** spec.power
            )
            best_rank[candidate] = min(best_rank.get(candidate, rank), rank)
    return [
        candidate
        for candidate, _ in sorted(
            score.items(),
            key=lambda item: (-item[1], best_rank[item[0]], item[0]),
        )[:budget]
    ]


def evaluate_specs(
    queries: pd.DataFrame,
    maps: dict[tuple[str, str, str], dict[str, list[str]]],
    specs: list[FusionSpec],
    budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    for spec in specs:
        hits: list[int] = []
        reciprocal: list[float] = []
        valid = True
        local_rows: list[dict[str, object]] = []
        for row in queries.itertuples(index=False):
            key = (str(row.split_id), str(row.direction), str(row.query_id))
            source_rankings = maps.get(key)
            if source_rankings is None or any(source not in source_rankings for source in spec.sources):
                valid = False
                break
            ranking = fuse_ranking(source_rankings, spec, budget)
            positive_ranks = [index + 1 for index, value in enumerate(ranking) if value in row.positives]
            best = min(positive_ranks) if positive_ranks else 0
            hit = int(best > 0)
            rr = 0.0 if best == 0 else 1.0 / best
            hits.append(hit)
            reciprocal.append(rr)
            local_rows.append(
                {
                    "split_id": row.split_id,
                    "partition": row.partition,
                    "direction": row.direction,
                    "query_id": row.query_id,
                    "budget": budget,
                    "method": spec.name,
                    "hit": hit,
                    "reciprocal_rank_within_budget": rr,
                    "best_positive_rank_within_budget": best,
                }
            )
        if not valid:
            continue
        metric_rows.append(
            {
                "budget": budget,
                "method": spec.name,
                "kind": spec.kind,
                "sources": ";".join(spec.sources),
                "hit_probability": float(np.mean(hits)),
                "mean_reciprocal_rank_within_budget": float(np.mean(reciprocal)),
            }
        )
        query_rows.extend(local_rows)
    return pd.DataFrame(metric_rows), pd.DataFrame(query_rows)


def bootstrap_delta(
    paired: pd.DataFrame,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    cells = sorted(paired["split_id"].unique())
    values = {
        cell: paired.loc[paired["split_id"].eq(cell), "difference"].to_numpy(float)
        for cell in cells
    }
    rng = np.random.default_rng(seed)
    boot = np.empty(samples, dtype=float)
    for index in range(samples):
        sampled = rng.choice(cells, size=len(cells), replace=True)
        boot[index] = np.concatenate([values[str(cell)] for cell in sampled]).mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-selected strict MARTS fusion of adapted dual towers and a multi-expert model."
    )
    parser.add_argument("--dual-rankings", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--multi-rankings", type=Path, default=DEFAULT_MULTI)
    parser.add_argument("--multi-method", default="multi_expert_marts_only")
    parser.add_argument("--multi-source", default="multi_expert")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--weights", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--constants", default="0,10,30,60")
    parser.add_argument("--powers", default="0.5,1")
    parser.add_argument("--rescue-slots", default="1,2,3,5,10")
    parser.add_argument("--include-triples", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    budgets = parse_int_tuple(args.budgets)
    weights = parse_float_tuple(args.weights)
    constants = parse_float_tuple(args.constants)
    powers = parse_float_tuple(args.powers)
    rescue_slots = parse_int_tuple(args.rescue_slots)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rankings = load_rankings(
        args.dual_rankings.resolve(),
        args.multi_rankings.resolve(),
        args.multi_method,
        args.multi_source,
    )
    dual_sources = sorted(set(rankings["source"].astype(str)) - {args.multi_source})
    specs = build_specs(
        dual_sources,
        args.multi_source,
        weights,
        constants,
        powers,
        rescue_slots,
        args.include_triples,
    )
    queries = load_queries(args.cache_dir.resolve())
    maps = ranking_maps(rankings)
    sources_per_query = {key: set(value) for key, value in maps.items()}
    expected_sources = set([*dual_sources, args.multi_source])
    incomplete = [key for key, value in sources_per_query.items() if value != expected_sources]
    if incomplete:
        raise ValueError(f"Rankings are incomplete for {len(incomplete)} queries; examples={incomplete[:3]}")

    development_metrics: list[pd.DataFrame] = []
    development_queries: list[pd.DataFrame] = []
    selected_rows: list[dict[str, object]] = []
    frozen_rows: list[pd.DataFrame] = []
    paired_rows: list[dict[str, object]] = []

    for direction in sorted(queries["direction"].unique()):
        development = queries[
            queries["partition"].eq("development_9_cells")
            & queries["direction"].eq(direction)
        ]
        frozen = queries[
            queries["partition"].eq("frozen_16_cells")
            & queries["direction"].eq(direction)
        ]
        for budget in budgets:
            eligible_specs = [
                spec
                for spec in specs
                if spec.kind != "rescue" or spec.rescue_slots <= budget
            ]
            dev_metrics, dev_query = evaluate_specs(development, maps, eligible_specs, budget)
            dev_metrics.insert(0, "direction", direction)
            development_metrics.append(dev_metrics)
            development_queries.append(dev_query)
            selected = dev_metrics.sort_values(
                ["hit_probability", "mean_reciprocal_rank_within_budget", "method"],
                ascending=[False, False, True],
            ).iloc[0]
            best_single = (
                dev_metrics[dev_metrics["kind"].eq("single")]
                .sort_values(
                    ["hit_probability", "mean_reciprocal_rank_within_budget", "method"],
                    ascending=[False, False, True],
                )
                .iloc[0]
            )
            selected_spec = next(spec for spec in eligible_specs if spec.name == selected.method)
            reference_spec = next(spec for spec in eligible_specs if spec.name == best_single.method)
            frozen_selected_metrics, frozen_selected_query = evaluate_specs(
                frozen, maps, [selected_spec], budget
            )
            frozen_reference_metrics, frozen_reference_query = evaluate_specs(
                frozen, maps, [reference_spec], budget
            )
            selected_rows.append(
                {
                    "direction": direction,
                    "budget": budget,
                    "selected_method": selected.method,
                    "selected_kind": selected.kind,
                    "selected_sources": selected.sources,
                    "development_hit_probability": selected.hit_probability,
                    "development_mrr_within_budget": selected.mean_reciprocal_rank_within_budget,
                    "reference_single_method": best_single.method,
                    "reference_development_hit_probability": best_single.hit_probability,
                    "frozen_selected_hit_probability": frozen_selected_metrics.iloc[0].hit_probability,
                    "frozen_reference_hit_probability": frozen_reference_metrics.iloc[0].hit_probability,
                }
            )
            frozen_selected_query = frozen_selected_query.copy()
            frozen_selected_query["evaluation_role"] = "selected"
            frozen_reference_query = frozen_reference_query.copy()
            frozen_reference_query["evaluation_role"] = "reference_single"
            frozen_rows.extend([frozen_selected_query, frozen_reference_query])
            keys = ["split_id", "direction", "query_id", "budget"]
            paired = frozen_selected_query[keys + ["hit"]].rename(
                columns={"hit": "selected_hit"}
            ).merge(
                frozen_reference_query[keys + ["hit"]].rename(
                    columns={"hit": "reference_hit"}
                ),
                on=keys,
                validate="one_to_one",
            )
            paired["difference"] = paired["selected_hit"] - paired["reference_hit"]
            low, high = bootstrap_delta(paired, args.bootstrap_samples, args.seed + budget)
            paired_rows.append(
                {
                    "direction": direction,
                    "budget": budget,
                    "selected_method": selected.method,
                    "reference_single_method": best_single.method,
                    "n_paired_queries": len(paired),
                    "selected_hit_probability": paired["selected_hit"].mean(),
                    "reference_hit_probability": paired["reference_hit"].mean(),
                    "absolute_delta": paired["difference"].mean(),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "selected_only_hits": int(
                        ((paired["selected_hit"] == 1) & (paired["reference_hit"] == 0)).sum()
                    ),
                    "reference_only_hits": int(
                        ((paired["selected_hit"] == 0) & (paired["reference_hit"] == 1)).sum()
                    ),
                }
            )

    development_metric_frame = pd.concat(development_metrics, ignore_index=True)
    development_query_frame = pd.concat(development_queries, ignore_index=True)
    selected_frame = pd.DataFrame(selected_rows)
    frozen_frame = pd.concat(frozen_rows, ignore_index=True)
    paired_frame = pd.DataFrame(paired_rows)
    development_metric_frame.to_csv(output / "development_candidate_metrics.csv", index=False)
    development_query_frame.to_csv(output / "development_query_metrics.csv", index=False)
    selected_frame.to_csv(output / "selected_methods.csv", index=False)
    frozen_frame.to_csv(output / "frozen_query_metrics.csv", index=False)
    paired_frame.to_csv(output / "frozen_paired_comparison.csv", index=False)
    summary = {
        "selection_protocol": "development_9_cells_only_then_locked_frozen_16_cells",
        "dual_sources": dual_sources,
        "multi_source": args.multi_source,
        "multi_method": args.multi_method,
        "budgets": list(budgets),
        "n_specs": len(specs),
        "ranking_depth": int(rankings["rank"].max()),
        "n_queries": len(queries),
        "outputs": {
            "development_candidate_metrics": str(output / "development_candidate_metrics.csv"),
            "development_query_metrics": str(output / "development_query_metrics.csv"),
            "selected_methods": str(output / "selected_methods.csv"),
            "frozen_query_metrics": str(output / "frozen_query_metrics.csv"),
            "frozen_paired_comparison": str(output / "frozen_paired_comparison.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(selected_frame.to_string(index=False))
    print(paired_frame.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
