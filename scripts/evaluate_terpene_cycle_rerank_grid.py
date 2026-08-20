from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.evidence import cycle_consistency_score  # noqa: E402
from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    resolve_protein_dir,
    resolve_reaction_path,
)
from projects.active.terpene_screening.rank_open_world import (  # noqa: E402
    DEFAULT_POSITIVES,
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    build_parser,
    execute_ranking,
    load_external_reaction_rows,
    load_protein_library,
)
from projects.active.terpene_screening.rank_registry_batch import (  # noqa: E402
    DEFAULT_MARTS,
    build_known_association_maps,
)


DEFAULT_OUTPUT = ROOT / "results/terpene_cycle_rerank_grid_v2"
DEFAULT_WEIGHTS = (0.0, 0.05, 0.10, 0.15, 0.20)
DEFAULT_GATES = ("all", "applicability_ge_0.60", "applicability_ge_0.80")
DEFAULT_OBJECTIVES = (3, 10, 20)
DEFAULT_BUDGETS = (3, 10, 20)


def _hash_value(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16)


def _partition(query_id: str) -> str:
    return "development" if _hash_value(f"cycle-v2:{query_id}") % 2 == 0 else "confirmation"


def _execute(argv: list[str]) -> pd.DataFrame:
    args = build_parser().parse_args([*argv, "--conformal-mode", "disabled"])
    return execute_ranking(args)


def _candidate_rank(frame: pd.DataFrame, candidate_id: str) -> int | None:
    matched = frame[frame["candidate_id"].astype(str).eq(str(candidate_id))]
    return None if matched.empty else int(matched.iloc[0]["rank"])


def _balanced_partition_select(
    items: list[tuple[str, list[str]]],
    max_items: int,
) -> list[tuple[str, list[str]]]:
    development = [item for item in items if _partition(item[0]) == "development"]
    confirmation = [item for item in items if _partition(item[0]) == "confirmation"]
    target_development = max_items // 2
    target_confirmation = max_items - target_development
    selected = development[:target_development] + confirmation[:target_confirmation]
    selected_ids = {item[0] for item in selected}
    if len(selected) < max_items:
        selected.extend(
            item
            for item in items
            if item[0] not in selected_ids
        )
    return selected[:max_items]


def _panel(
    *,
    max_queries_per_direction: int,
    registered_protein_dir: Path,
    registered_reactions_csv: Path,
    positives: Path,
    marts: Path,
) -> pd.DataFrame:
    _, registered_protein_ids = load_protein_library(
        resolve_protein_dir(registered_protein_dir)
    )
    registered_reactions = load_external_reaction_rows(
        resolve_reaction_path(registered_reactions_csv)
    )
    registered_reaction_ids = set(registered_reactions["reaction_id"].astype(str))
    reactions_by_enzyme, enzymes_by_reaction = build_known_association_maps(
        marts, positives, registered_reaction_ids
    )
    rows: list[dict[str, Any]] = []
    enzyme_candidates = set(registered_protein_ids)
    current_proteins = pd.read_csv(
        ROOT / "data/terpene_embeddings/esmc600m_mean/entries.csv", dtype=str
    ).fillna("")
    for column in ("Entry", "protein_id"):
        if column in current_proteins:
            enzyme_candidates.update(current_proteins[column].astype(str))
            break
    positive_frame = pd.read_csv(positives, sep="\t", dtype=str).fillna("")
    reaction_candidates = set(positive_frame["rhea_id"].astype(str)) | registered_reaction_ids

    eligible_enzymes = [
        (query_id, sorted(set(reactions_by_enzyme.get(query_id, set())) & reaction_candidates))
        for query_id in registered_protein_ids
    ]
    eligible_enzymes = [(query_id, values) for query_id, values in eligible_enzymes if values]
    eligible_enzymes.sort(key=lambda item: (_hash_value(f"e2r:{item[0]}"), item[0]))
    for query_id, values in _balanced_partition_select(
        eligible_enzymes, max_queries_per_direction
    ):
        rows.append(
            {
                "direction": "enzyme_to_reaction",
                "query_id": query_id,
                "positive_ids": ";".join(values),
                "n_positives": len(values),
                "partition": _partition(query_id),
            }
        )

    eligible_reactions = [
        (query_id, sorted(set(enzymes_by_reaction.get(query_id, set())) & enzyme_candidates))
        for query_id in sorted(registered_reaction_ids)
    ]
    eligible_reactions = [(query_id, values) for query_id, values in eligible_reactions if values]
    eligible_reactions.sort(key=lambda item: (_hash_value(f"r2e:{item[0]}"), item[0]))
    for query_id, values in _balanced_partition_select(
        eligible_reactions, max_queries_per_direction
    ):
        rows.append(
            {
                "direction": "reaction_to_enzyme",
                "query_id": query_id,
                "positive_ids": ";".join(values),
                "n_positives": len(values),
                "partition": _partition(query_id),
            }
        )
    return pd.DataFrame(rows)


def _forward_argv(
    direction: str,
    query_id: str,
    objective: int,
    top_k: int,
    device: str,
) -> list[str]:
    common = [
        "--top-k",
        str(top_k),
        "--ranking-objective",
        f"top{objective}",
        "--device",
        device,
    ]
    if direction == "enzyme_to_reaction":
        return ["rank-reactions", "--enzyme-id", query_id, *common]
    return ["rank-enzymes", "--reaction-id", query_id, *common]


def _reverse_argv(
    forward_direction: str,
    candidate_id: str,
    reverse_top_k: int,
    device: str,
) -> list[str]:
    common = [
        "--top-k",
        str(reverse_top_k),
        "--ranking-objective",
        "top20",
        "--device",
        device,
    ]
    if forward_direction == "enzyme_to_reaction":
        return ["rank-enzymes", "--reaction-id", candidate_id, *common]
    return ["rank-reactions", "--enzyme-id", candidate_id, *common]


def collect_cycles(
    panel: pd.DataFrame,
    *,
    objectives: tuple[int, ...],
    forward_top_k: int,
    cycle_top_n: int,
    reverse_top_k: int,
    reciprocal_rank_constant: float,
    device: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reverse_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for query in panel.itertuples(index=False):
        positives = {value for value in str(query.positive_ids).split(";") if value}
        for objective in objectives:
            forward = _execute(
                _forward_argv(
                    str(query.direction),
                    str(query.query_id),
                    objective,
                    forward_top_k,
                    device,
                )
            )
            applicability_score = float(forward.iloc[0]["query_applicability_score"])
            applicability_tier = str(forward.iloc[0]["query_applicability_tier"])
            evaluated = min(cycle_top_n, len(forward))
            for local_index, candidate in forward.iterrows():
                candidate_id = str(candidate["candidate_id"])
                reverse_rank: int | None = None
                reverse_route = ""
                if local_index < evaluated:
                    reverse_key = (str(query.direction), candidate_id)
                    reverse = reverse_cache.get(reverse_key)
                    if reverse is None:
                        reverse = _execute(
                            _reverse_argv(
                                str(query.direction),
                                candidate_id,
                                reverse_top_k,
                                device,
                            )
                        )
                        reverse_cache[reverse_key] = reverse
                    reverse_rank = _candidate_rank(reverse, str(query.query_id))
                    reverse_route = str(reverse.iloc[0].get("route_id", ""))
                forward_rank = int(candidate["rank"])
                rows.append(
                    {
                        "partition": str(query.partition),
                        "direction": str(query.direction),
                        "objective": f"top{objective}",
                        "query_id": str(query.query_id),
                        "positive_ids": str(query.positive_ids),
                        "candidate_id": candidate_id,
                        "is_positive": candidate_id in positives,
                        "forward_rank": forward_rank,
                        "forward_score": float(candidate["score"]),
                        "forward_route_id": str(candidate.get("route_id", "")),
                        "query_applicability_score": applicability_score,
                        "query_applicability_tier": applicability_tier,
                        "cycle_evaluated": bool(local_index < evaluated),
                        "reverse_rank": reverse_rank,
                        "reverse_route_id": reverse_route,
                        "cycle_consistency_score": cycle_consistency_score(
                            forward_rank,
                            reverse_rank,
                            reciprocal_rank_constant=reciprocal_rank_constant,
                        )
                        if local_index < evaluated
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _gate_applies(gate: str, applicability_score: float) -> bool:
    if gate == "all":
        return True
    if gate == "applicability_ge_0.60":
        return applicability_score >= 0.60
    if gate == "applicability_ge_0.80":
        return applicability_score >= 0.80
    raise ValueError(f"Unknown cycle gate: {gate}")


def rerank_query(
    frame: pd.DataFrame,
    *,
    weight: float,
    gate: str,
    reverse_top_k: int,
    reciprocal_rank_constant: float,
) -> pd.DataFrame:
    result = frame.copy()
    applicability = float(result.iloc[0]["query_applicability_score"])
    effective_weight = weight if _gate_applies(gate, applicability) else 0.0
    reverse_rank = pd.to_numeric(result["reverse_rank"], errors="coerce").fillna(
        reverse_top_k + 1
    )
    result["cycle_grid_score"] = (
        (1.0 - effective_weight)
        / (reciprocal_rank_constant + result["forward_rank"].astype(float))
        + effective_weight / (reciprocal_rank_constant + reverse_rank.astype(float))
    )
    result = result.sort_values(
        ["cycle_grid_score", "candidate_id"], ascending=[False, True]
    ).reset_index(drop=True)
    result["reranked_rank"] = np.arange(1, len(result) + 1, dtype=np.int64)
    result["cycle_weight"] = weight
    result["effective_cycle_weight"] = effective_weight
    result["cycle_gate"] = gate
    return result


def _query_metrics(frame: pd.DataFrame, rank_column: str) -> dict[str, Any]:
    positives = frame[frame["is_positive"]]
    best_rank = int(positives[rank_column].min()) if len(positives) else len(frame) + 1
    return {
        "best_positive_rank": best_rank,
        "reciprocal_rank": 1.0 / best_rank if best_rank <= len(frame) else 0.0,
        **{f"hit_at_{budget}": int(best_rank <= budget) for budget in DEFAULT_BUDGETS},
    }


def evaluate_grid(
    cycles: pd.DataFrame,
    *,
    weights: tuple[float, ...],
    gates: tuple[str, ...],
    reverse_top_k: int,
    reciprocal_rank_constant: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows: list[dict[str, Any]] = []
    reranked_rows: list[pd.DataFrame] = []
    keys = ["partition", "direction", "objective", "query_id"]
    for query_key, frame in cycles.groupby(keys, sort=True):
        baseline = _query_metrics(frame, "forward_rank")
        for gate in gates:
            for weight in weights:
                reranked = rerank_query(
                    frame,
                    weight=weight,
                    gate=gate,
                    reverse_top_k=reverse_top_k,
                    reciprocal_rank_constant=reciprocal_rank_constant,
                )
                reranked_rows.append(reranked)
                metrics = _query_metrics(reranked, "reranked_rank")
                query_rows.append(
                    {
                        **dict(zip(keys, query_key)),
                        "cycle_gate": gate,
                        "cycle_weight": weight,
                        "effective_cycle_weight": float(
                            reranked["effective_cycle_weight"].iloc[0]
                        ),
                        "query_applicability_score": float(
                            reranked["query_applicability_score"].iloc[0]
                        ),
                        "baseline_best_positive_rank": baseline[
                            "best_positive_rank"
                        ],
                        "reranked_best_positive_rank": metrics[
                            "best_positive_rank"
                        ],
                        "baseline_reciprocal_rank": baseline["reciprocal_rank"],
                        "reranked_reciprocal_rank": metrics["reciprocal_rank"],
                        **{
                            f"baseline_hit_at_{budget}": baseline[f"hit_at_{budget}"]
                            for budget in DEFAULT_BUDGETS
                        },
                        **{
                            f"reranked_hit_at_{budget}": metrics[f"hit_at_{budget}"]
                            for budget in DEFAULT_BUDGETS
                        },
                    }
                )
    query_metrics = pd.DataFrame(query_rows)
    grid_rows: list[dict[str, Any]] = []
    for key, frame in query_metrics.groupby(
        ["partition", "direction", "objective", "cycle_gate", "cycle_weight"],
        sort=True,
    ):
        base = {
            "partition": key[0],
            "direction": key[1],
            "objective": key[2],
            "cycle_gate": key[3],
            "cycle_weight": key[4],
            "n_queries": len(frame),
            "mean_effective_cycle_weight": float(frame["effective_cycle_weight"].mean()),
            "baseline_mrr": float(frame["baseline_reciprocal_rank"].mean()),
            "reranked_mrr": float(frame["reranked_reciprocal_rank"].mean()),
            "delta_mrr": float(
                (frame["reranked_reciprocal_rank"] - frame["baseline_reciprocal_rank"]).mean()
            ),
        }
        for budget in DEFAULT_BUDGETS:
            baseline_values = frame[f"baseline_hit_at_{budget}"].astype(int)
            reranked_values = frame[f"reranked_hit_at_{budget}"].astype(int)
            base[f"baseline_hit_at_{budget}"] = float(baseline_values.mean())
            base[f"reranked_hit_at_{budget}"] = float(reranked_values.mean())
            base[f"delta_hit_at_{budget}"] = float(
                (reranked_values - baseline_values).mean()
            )
            base[f"new_hits_at_{budget}"] = int(
                ((reranked_values == 1) & (baseline_values == 0)).sum()
            )
            base[f"lost_hits_at_{budget}"] = int(
                ((reranked_values == 0) & (baseline_values == 1)).sum()
            )
        grid_rows.append(base)
    return query_metrics, pd.DataFrame(grid_rows)


def select_and_confirm(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = grid[grid["partition"].eq("development")]
    confirmation = grid[grid["partition"].eq("confirmation")]
    selected_rows: list[dict[str, Any]] = []
    confirmed_rows: list[dict[str, Any]] = []
    for direction in sorted(development["direction"].unique()):
        for objective in sorted(development["objective"].unique()):
            candidates = development[
                development["direction"].eq(direction)
                & development["objective"].eq(objective)
            ]
            if candidates.empty:
                continue
            for budget in DEFAULT_BUDGETS:
                ordered = candidates.sort_values(
                    [
                        f"reranked_hit_at_{budget}",
                        f"lost_hits_at_{budget}",
                        "reranked_mrr",
                        "cycle_weight",
                        "cycle_gate",
                    ],
                    ascending=[False, True, False, True, True],
                )
                selected = ordered.iloc[0].to_dict()
                selected["selection_budget"] = budget
                selected_rows.append(selected)
                matched = confirmation[
                    confirmation["direction"].eq(direction)
                    & confirmation["objective"].eq(objective)
                    & confirmation["cycle_gate"].eq(selected["cycle_gate"])
                    & confirmation["cycle_weight"].eq(selected["cycle_weight"])
                ]
                if matched.empty:
                    continue
                confirmed = matched.iloc[0].to_dict()
                confirmed["selection_budget"] = budget
                delta_hit = float(confirmed[f"delta_hit_at_{budget}"])
                lost_hits = int(confirmed[f"lost_hits_at_{budget}"])
                delta_mrr = float(confirmed["delta_mrr"])
                confirmed["promotion_status"] = (
                    "candidate_for_locked_route_test"
                    if delta_hit > 0 and lost_hits == 0 and delta_mrr >= 0
                    else "evidence_only_no_route_change"
                )
                confirmed_rows.append(confirmed)
    return pd.DataFrame(selected_rows), pd.DataFrame(confirmed_rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate applicability-gated cycle-consistency reranking on registered TPS queries."
    )
    parser.add_argument("--max-queries-per-direction", type=int, default=6)
    parser.add_argument("--forward-top-k", type=int, default=20)
    parser.add_argument("--cycle-top-n", type=int, default=20)
    parser.add_argument("--reverse-top-k", type=int, default=50)
    parser.add_argument("--objectives", default=",".join(str(value) for value in DEFAULT_OBJECTIVES))
    parser.add_argument("--weights", default=",".join(str(value) for value in DEFAULT_WEIGHTS))
    parser.add_argument("--gates", default=",".join(DEFAULT_GATES))
    parser.add_argument("--rrf-constant", type=float, default=10.0)
    parser.add_argument("--registered-protein-dir", type=Path, default=DEFAULT_REGISTERED_PROTEIN_DIR)
    parser.add_argument("--registered-reactions-csv", type=Path, default=DEFAULT_REGISTERED_REACTIONS)
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--marts", type=Path, default=DEFAULT_MARTS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_queries_per_direction <= 1:
        raise ValueError("max-queries-per-direction must exceed one")
    if args.cycle_top_n <= 0 or args.forward_top_k < args.cycle_top_n:
        raise ValueError("cycle-top-n must be positive and no larger than forward-top-k")
    objectives = tuple(sorted({int(value) for value in args.objectives.split(",") if value}))
    weights = tuple(sorted({float(value) for value in args.weights.split(",") if value}))
    gates = tuple(value for value in args.gates.split(",") if value)
    if any(value not in DEFAULT_OBJECTIVES for value in objectives):
        raise ValueError("objectives must be selected from 3,10,20")
    if any(not 0.0 <= value <= 0.25 for value in weights):
        raise ValueError("cycle weights must be within [0, 0.25]")
    for gate in gates:
        _gate_applies(gate, 1.0)

    panel = _panel(
        max_queries_per_direction=args.max_queries_per_direction,
        registered_protein_dir=args.registered_protein_dir.resolve(),
        registered_reactions_csv=args.registered_reactions_csv.resolve(),
        positives=args.positives.resolve(),
        marts=args.marts.resolve(),
    )
    if panel.groupby("direction").size().min() < 2:
        raise ValueError("Insufficient eligible registered queries for cycle grid")
    cycles = collect_cycles(
        panel,
        objectives=objectives,
        forward_top_k=args.forward_top_k,
        cycle_top_n=args.cycle_top_n,
        reverse_top_k=args.reverse_top_k,
        reciprocal_rank_constant=args.rrf_constant,
        device=args.device,
    )
    query_metrics, grid = evaluate_grid(
        cycles,
        weights=weights,
        gates=gates,
        reverse_top_k=args.reverse_top_k,
        reciprocal_rank_constant=args.rrf_constant,
    )
    selected, confirmed = select_and_confirm(grid)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "query_panel.csv", index=False)
    cycles.to_csv(output_dir / "candidate_cycles.csv", index=False)
    query_metrics.to_csv(output_dir / "query_metrics.csv", index=False)
    grid.to_csv(output_dir / "grid_metrics.csv", index=False)
    selected.to_csv(output_dir / "development_selected_configs.csv", index=False)
    confirmed.to_csv(output_dir / "confirmation_metrics.csv", index=False)
    promoted = (
        confirmed["promotion_status"].eq("candidate_for_locked_route_test").sum()
        if len(confirmed)
        else 0
    )
    summary = {
        "status": "completed",
        "evaluation_scope": "registered_known-association_proxy_not_double-cold_confirmation",
        "n_queries": int(len(panel)),
        "queries_by_direction": panel["direction"].value_counts().to_dict(),
        "queries_by_partition": panel["partition"].value_counts().to_dict(),
        "objectives": list(objectives),
        "weights": list(weights),
        "gates": list(gates),
        "cycle_top_n": args.cycle_top_n,
        "reverse_top_k": args.reverse_top_k,
        "promotion_candidates": int(promoted),
        "production_ranking_modified": False,
        "outputs": {
            "panel": str(output_dir / "query_panel.csv"),
            "cycles": str(output_dir / "candidate_cycles.csv"),
            "grid": str(output_dir / "grid_metrics.csv"),
            "selected": str(output_dir / "development_selected_configs.csv"),
            "confirmation": str(output_dir / "confirmation_metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
