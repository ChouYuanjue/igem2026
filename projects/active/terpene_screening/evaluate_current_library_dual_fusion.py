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

DEFAULT_PANELS = ROOT / "results/terpene_current_library_expert_v1/panels.csv"
DEFAULT_EXACT_FOLDS = (
    ROOT / "projects/active/terpene_screening/comparison_assets/legacy_exact_reaction_folds.csv"
)
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_OUTPUT = ROOT / "results/terpene_current_library_dual_fusion"
DEFAULT_BUDGETS = (3, 5, 10, 20)
DEFAULT_DUAL_SOURCES = (
    "baseline=results/terpene_fusion_sources_100e_v1/baseline/rankings.csv",
    "multiview=results/terpene_fusion_sources_100e_v1/multiview/rankings.csv",
    "multiview_top10=results/terpene_fusion_sources_100e_v1/multiview_top10/rankings.csv",
    "me8=results/terpene_fusion_sources_100e_v1/me8_multiview/rankings.csv",
)


@dataclass(frozen=True)
class FusionSpec:
    name: str
    kind: str
    old_method: str
    dual_source: str
    old_slots: int = 0
    old_weight: float = 0.5
    constant: float = 0.0
    power: float = 1.0


def parse_source(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError("Each source must use LABEL=RANKINGS_CSV")
    return label, Path(path).resolve()


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
    budget: int,
    old_methods: list[str],
    dual_sources: list[str],
    weights: tuple[float, ...],
    constants: tuple[float, ...],
    powers: tuple[float, ...],
) -> list[FusionSpec]:
    specs: list[FusionSpec] = []
    for old_method in old_methods:
        specs.append(
            FusionSpec(
                name=f"old_only__{old_method}",
                kind="old_only",
                old_method=old_method,
                dual_source="",
                old_slots=budget,
            )
        )
        for dual_source in dual_sources:
            for old_slots in range(0, budget + 1):
                specs.append(
                    FusionSpec(
                        name=(
                            f"quota__{old_method}__{dual_source}"
                            f"__old{old_slots}__dual{budget-old_slots}"
                        ),
                        kind="quota",
                        old_method=old_method,
                        dual_source=dual_source,
                        old_slots=old_slots,
                    )
                )
            for old_weight in weights:
                for constant in constants:
                    for power in powers:
                        specs.append(
                            FusionSpec(
                                name=(
                                    f"rrf__{old_method}_{old_weight:g}__{dual_source}_{1-old_weight:g}"
                                    f"__c{constant:g}__p{power:g}"
                                ),
                                kind="rrf",
                                old_method=old_method,
                                dual_source=dual_source,
                                old_weight=old_weight,
                                constant=constant,
                                power=power,
                            )
                        )
    for dual_source in dual_sources:
        specs.append(
            FusionSpec(
                name=f"dual_only__{dual_source}",
                kind="dual_only",
                old_method="",
                dual_source=dual_source,
            )
        )
    return specs


def load_old_panels(path: Path) -> dict[tuple[int, str, str], list[str]]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["B"] = pd.to_numeric(frame["B"]).astype(int)
    frame["rank"] = pd.to_numeric(frame["rank"]).astype(int)
    result: dict[tuple[int, str, str], list[str]] = {}
    for key, group in frame.groupby(["B", "method", "reaction_id"], sort=False):
        ranking = (
            group.sort_values(["rank", "uniprot_id"])["uniprot_id"].astype(str).tolist()
        )
        if len(ranking) != int(key[0]) or len(ranking) != len(set(ranking)):
            raise ValueError(f"Invalid old panel for {key}")
        result[(int(key[0]), str(key[1]), str(key[2]))] = ranking
    return result


def load_dual_rankings(sources: dict[str, Path]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    expected_queries: set[str] | None = None
    for label, path in sources.items():
        frame = pd.read_csv(path, dtype=str).fillna("")
        frame = frame[frame["protocol"].eq("legacy_exact")].copy()
        if frame.empty:
            raise ValueError(f"No legacy_exact rankings in {path}")
        frame["rank"] = pd.to_numeric(frame["rank"]).astype(int)
        local_queries = set(frame["reaction_id"].astype(str))
        if expected_queries is None:
            expected_queries = local_queries
        elif local_queries != expected_queries:
            raise ValueError(f"Dual source query sets differ for {label}")
        for reaction_id, group in frame.groupby("reaction_id", sort=False):
            ranking = (
                group.sort_values(["rank", "candidate_id"])["candidate_id"].astype(str).tolist()
            )
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"Duplicate candidates in dual ranking {label}/{reaction_id}")
            result[(label, str(reaction_id))] = ranking
    return result


def quota_fusion(old: list[str], dual: list[str], old_slots: int, budget: int) -> list[str]:
    selected = list(old[:old_slots])
    used = set(selected)
    for candidate in dual:
        if len(selected) >= budget:
            break
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
    for candidate in old[old_slots:]:
        if len(selected) >= budget:
            break
        if candidate not in used:
            selected.append(candidate)
            used.add(candidate)
    return selected


def rrf_fusion(
    old: list[str],
    dual: list[str],
    old_weight: float,
    constant: float,
    power: float,
    budget: int,
) -> list[str]:
    score: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in ((old, old_weight), (dual, 1.0 - old_weight)):
        for rank, candidate in enumerate(ranking, start=1):
            score[candidate] = score.get(candidate, 0.0) + weight / (
                (constant + rank) ** power
            )
            best_rank[candidate] = min(best_rank.get(candidate, rank), rank)
    return [
        candidate
        for candidate, _ in sorted(
            score.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0])
        )[:budget]
    ]


def fuse(
    spec: FusionSpec,
    budget: int,
    reaction_id: str,
    old_panels: dict[tuple[int, str, str], list[str]],
    dual_rankings: dict[tuple[str, str], list[str]],
) -> list[str]:
    if spec.kind == "dual_only":
        return dual_rankings[(spec.dual_source, reaction_id)][:budget]
    old = old_panels[(budget, spec.old_method, reaction_id)]
    if spec.kind == "old_only":
        return old[:budget]
    dual = dual_rankings[(spec.dual_source, reaction_id)]
    if spec.kind == "quota":
        return quota_fusion(old, dual, spec.old_slots, budget)
    if spec.kind == "rrf":
        return rrf_fusion(
            old,
            dual,
            spec.old_weight,
            spec.constant,
            spec.power,
            budget,
        )
    raise ValueError(f"Unknown fusion kind: {spec.kind}")


def evaluate_specs(
    reactions: list[str],
    positives: dict[str, set[str]],
    budget: int,
    specs: list[FusionSpec],
    old_panels: dict[tuple[int, str, str], list[str]],
    dual_rankings: dict[tuple[str, str], list[str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec_index, spec in enumerate(specs):
        hits = 0
        expected_hits = 0
        reciprocal = 0.0
        for reaction_id in reactions:
            ranking = fuse(spec, budget, reaction_id, old_panels, dual_rankings)
            if len(ranking) != budget or len(ranking) != len(set(ranking)):
                raise ValueError(f"Invalid fused ranking for {spec.name}/{reaction_id}")
            positive_ranks = [
                rank
                for rank, candidate in enumerate(ranking, start=1)
                if candidate in positives[reaction_id]
            ]
            count = len(positive_ranks)
            hits += int(count > 0)
            expected_hits += count
            reciprocal += 0.0 if not positive_ranks else 1.0 / min(positive_ranks)
        n = len(reactions)
        rows.append(
            {
                "parameter_index": spec_index,
                "method": spec.name,
                "kind": spec.kind,
                "old_method": spec.old_method,
                "dual_source": spec.dual_source,
                "old_slots": spec.old_slots,
                "old_weight": spec.old_weight,
                "constant": spec.constant,
                "power": spec.power,
                "n_reactions": n,
                "hit_probability": hits / n,
                "expected_hits": expected_hits / n,
                "mean_reciprocal_rank_within_budget": reciprocal / n,
            }
        )
    return pd.DataFrame(rows)


def precompute_outcomes(
    reactions: list[str],
    positives: dict[str, set[str]],
    budget: int,
    specs: list[FusionSpec],
    old_panels: dict[tuple[int, str, str], list[str]],
    dual_rankings: dict[tuple[str, str], list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit = np.zeros((len(specs), len(reactions)), dtype=np.uint8)
    counts = np.zeros((len(specs), len(reactions)), dtype=np.int16)
    reciprocal = np.zeros((len(specs), len(reactions)), dtype=np.float32)
    for spec_index, spec in enumerate(specs):
        for reaction_index, reaction_id in enumerate(reactions):
            ranking = fuse(spec, budget, reaction_id, old_panels, dual_rankings)
            if len(ranking) != budget or len(ranking) != len(set(ranking)):
                raise ValueError(f"Invalid fused ranking for {spec.name}/{reaction_id}")
            positive_ranks = [
                rank
                for rank, candidate in enumerate(ranking, start=1)
                if candidate in positives[reaction_id]
            ]
            counts[spec_index, reaction_index] = len(positive_ranks)
            hit[spec_index, reaction_index] = int(bool(positive_ranks))
            reciprocal[spec_index, reaction_index] = (
                0.0 if not positive_ranks else 1.0 / min(positive_ranks)
            )
    return hit, counts, reciprocal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested exact-reaction fusion of the current-library expert and OOF dual towers."
    )
    parser.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--dual-source", action="append", default=[])
    parser.add_argument("--exact-folds", type=Path, default=DEFAULT_EXACT_FOLDS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--weights", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--constants", default="0,10,30,60")
    parser.add_argument("--powers", default="0.5,1")
    args = parser.parse_args()

    budgets = parse_int_tuple(args.budgets)
    weights = parse_float_tuple(args.weights)
    constants = parse_float_tuple(args.constants)
    powers = parse_float_tuple(args.powers)
    source_values = args.dual_source or list(DEFAULT_DUAL_SOURCES)
    dual_sources = dict(parse_source(value) for value in source_values)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    old_panels = load_old_panels(args.panels.resolve())
    dual_rankings = load_dual_rankings(dual_sources)
    positives_frame = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives = {
        str(reaction_id): set(group["Entry"].astype(str))
        for reaction_id, group in positives_frame.groupby("rhea_id", sort=True)
    }
    folds = pd.read_csv(args.exact_folds, dtype=str).fillna("")
    folds["legacy_exact_fold"] = pd.to_numeric(folds["legacy_exact_fold"]).astype(int)
    fold_by_reaction = dict(
        zip(folds["reaction_id"].astype(str), folds["legacy_exact_fold"].astype(int))
    )
    reactions = sorted(set(positives) & set(fold_by_reaction))
    if len(reactions) != 513:
        raise ValueError(f"Expected 513 exact-reaction queries, found {len(reactions)}")

    selection_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []
    reaction_index = {reaction_id: index for index, reaction_id in enumerate(reactions)}
    fold_array = np.asarray([fold_by_reaction[value] for value in reactions], dtype=np.int8)
    reaction_set = set(reactions)
    for budget in budgets:
        old_methods = sorted(
            {
                method
                for local_budget, method, reaction_id in old_panels
                if local_budget == budget and reaction_id in reaction_set
            }
        )
        specs = build_specs(
            budget,
            old_methods,
            list(dual_sources),
            weights,
            constants,
            powers,
        )
        hit_matrix, count_matrix, reciprocal_matrix = precompute_outcomes(
            reactions,
            positives,
            budget,
            specs,
            old_panels,
            dual_rankings,
        )
        method_names = np.asarray([spec.name for spec in specs], dtype=object)
        for target_fold in range(5):
            selection_mask = fold_array != target_fold
            test_indices = np.flatnonzero(fold_array == target_fold)
            hit_probability = hit_matrix[:, selection_mask].mean(axis=1)
            expected_hits = count_matrix[:, selection_mask].mean(axis=1)
            mrr = reciprocal_matrix[:, selection_mask].mean(axis=1)
            order = np.lexsort((method_names, -mrr, -expected_hits, -hit_probability))
            selected_index = int(order[0])
            selected_spec = specs[selected_index]
            selection_rows.append(
                {
                    "budget": budget,
                    "target_fold": target_fold,
                    "parameter_index": selected_index,
                    "method": selected_spec.name,
                    "kind": selected_spec.kind,
                    "old_method": selected_spec.old_method,
                    "dual_source": selected_spec.dual_source,
                    "old_slots": selected_spec.old_slots,
                    "old_weight": selected_spec.old_weight,
                    "constant": selected_spec.constant,
                    "power": selected_spec.power,
                    "n_reactions": int(selection_mask.sum()),
                    "hit_probability": float(hit_probability[selected_index]),
                    "expected_hits": float(expected_hits[selected_index]),
                    "mean_reciprocal_rank_within_budget": float(mrr[selected_index]),
                }
            )
            for local_index in test_indices:
                reaction_id = reactions[int(local_index)]
                ranking = fuse(
                    selected_spec,
                    budget,
                    reaction_id,
                    old_panels,
                    dual_rankings,
                )
                positive_ranks = [
                    rank
                    for rank, candidate in enumerate(ranking, start=1)
                    if candidate in positives[reaction_id]
                ]
                query_rows.append(
                    {
                        "budget": budget,
                        "target_fold": target_fold,
                        "reaction_id": reaction_id,
                        "selected_method": selected_spec.name,
                        "selected_kind": selected_spec.kind,
                        "selected_old_method": selected_spec.old_method,
                        "selected_dual_source": selected_spec.dual_source,
                        "hit": int(hit_matrix[selected_index, local_index]),
                        "hits": int(count_matrix[selected_index, local_index]),
                        "best_positive_rank_within_budget": (
                            min(positive_ranks) if positive_ranks else 0
                        ),
                        "reciprocal_rank_within_budget": float(
                            reciprocal_matrix[selected_index, local_index]
                        ),
                        "ranking": ";".join(ranking),
                    }
                )

    selection = pd.DataFrame(selection_rows)
    query_metrics = pd.DataFrame(query_rows)
    metrics = (
        query_metrics.groupby("budget", as_index=False)
        .agg(
            n_reactions=("reaction_id", "size"),
            hit_probability=("hit", "mean"),
            expected_hits=("hits", "mean"),
            mean_reciprocal_rank_within_budget=("reciprocal_rank_within_budget", "mean"),
        )
    )
    selection.to_csv(output / "nested_selected_parameters.csv", index=False)
    query_metrics.to_csv(output / "nested_query_metrics.csv", index=False)
    metrics.to_csv(output / "nested_metrics.csv", index=False)
    summary = {
        "protocol": "legacy_exact_five_fold_nested_parameter_selection",
        "n_reactions": len(reactions),
        "budgets": list(budgets),
        "dual_sources": {label: str(path) for label, path in dual_sources.items()},
        "outputs": {
            "nested_selected_parameters": str(output / "nested_selected_parameters.csv"),
            "nested_query_metrics": str(output / "nested_query_metrics.csv"),
            "nested_metrics": str(output / "nested_metrics.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(selection[["budget", "target_fold", "method", "hit_probability", "expected_hits"]].to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
