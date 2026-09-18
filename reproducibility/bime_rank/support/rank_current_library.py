from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / "results/terpene_current_library_dual_fusion_restricted_v1"
DEFAULT_OUTPUT = ROOT / "results/terpene_current_library_ranking.csv"
SUPPORTED_BUDGETS = (3, 5, 10, 20)


def resolve_budget(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top-k must be positive")
    for budget in SUPPORTED_BUDGETS:
        if top_k <= budget:
            return budget
    raise ValueError(f"current-library expert supports at most Top-{SUPPORTED_BUDGETS[-1]}")


def load_selected_method(results_dir: Path, budget: int) -> str:
    best = pd.read_csv(results_dir / "best_methods.csv", dtype=str).fillna("")
    best["B"] = pd.to_numeric(best["B"]).astype(int)
    row = best[best["scope"].eq("all513") & best["B"].eq(budget)]
    if len(row) != 1:
        raise ValueError(f"Expected one selected all513 method for budget {budget}")
    return str(row.iloc[0]["method"])


def rank_nested_fusion(
    reaction_id: str,
    top_k: int,
    budget: int,
    results_dir: Path,
) -> pd.DataFrame:
    nested = pd.read_csv(
        results_dir / "nested_query_metrics.csv",
        dtype=str,
    ).fillna("")
    nested["budget"] = pd.to_numeric(nested["budget"]).astype(int)
    selected = nested[
        nested["reaction_id"].eq(reaction_id) & nested["budget"].eq(budget)
    ]
    if len(selected) != 1:
        available = set(nested["reaction_id"].astype(str))
        if reaction_id not in available:
            raise ValueError(
                f"Reaction {reaction_id} is not in the 513-reaction current library"
            )
        raise ValueError(
            f"Expected one nested ranking for reaction={reaction_id}, budget={budget}; "
            f"found {len(selected)}"
        )
    row = selected.iloc[0]
    ranking = [value for value in str(row["ranking"]).split(";") if value]
    if len(ranking) < top_k:
        raise ValueError(
            f"Nested current-library ranking has {len(ranking)} candidates, expected {top_k}"
        )
    ranking = ranking[:top_k]
    if len(ranking) != len(set(ranking)):
        raise ValueError("Nested current-library ranking contains duplicate candidates")
    result = pd.DataFrame(
        {
            "rank": range(1, top_k + 1),
            "candidate_id": ranking,
        }
    )
    result.insert(0, "query_id", reaction_id)
    result.insert(1, "direction", "reaction_to_enzyme")
    result.insert(2, "candidate_scope", "current_library_1391")
    result.insert(3, "score_source", "nested_current_library_dual_fusion")
    result.insert(4, "selection_budget", budget)
    result["selected_method"] = str(row["selected_method"])
    result["selected_kind"] = str(row["selected_kind"])
    result["selected_old_method"] = str(row["selected_old_method"])
    result["selected_dual_source"] = str(row["selected_dual_source"])
    result["validation_fold"] = int(row["target_fold"])
    result["is_external_candidate"] = False
    result["reliability_status"] = "nested_exact_reaction_validated"
    return result


def rank_legacy_panel(
    reaction_id: str,
    top_k: int,
    budget: int,
    results_dir: Path,
) -> pd.DataFrame:
    method = load_selected_method(results_dir, budget)
    panels = pd.read_csv(
        results_dir / "panels.csv",
        dtype={"reaction_id": str, "uniprot_id": str, "method": str},
    ).fillna("")
    panels["B"] = pd.to_numeric(panels["B"]).astype(int)
    panels["rank"] = pd.to_numeric(panels["rank"]).astype(int)
    selected = panels[
        panels["reaction_id"].eq(reaction_id)
        & panels["B"].eq(budget)
        & panels["method"].eq(method)
    ].sort_values(["rank", "uniprot_id"])
    if selected.empty:
        available = sorted(panels["reaction_id"].astype(str).unique())
        if reaction_id not in set(available):
            raise ValueError(
                f"Reaction {reaction_id} is not in the 513-reaction current library"
            )
        raise ValueError(
            f"No precomputed panel for reaction={reaction_id}, budget={budget}, method={method}"
        )
    selected = selected.head(top_k).copy()
    if len(selected) != top_k:
        raise ValueError(
            f"Precomputed current-library panel has {len(selected)} rows, expected {top_k}"
        )
    if selected["uniprot_id"].duplicated().any():
        raise ValueError("Current-library panel contains duplicate candidates")
    selected = selected.rename(columns={"uniprot_id": "candidate_id"})
    selected.insert(0, "query_id", reaction_id)
    selected.insert(1, "direction", "reaction_to_enzyme")
    selected.insert(2, "candidate_scope", "current_library_1391")
    selected.insert(3, "score_source", f"legacy_current_library::{method}")
    selected.insert(4, "selection_budget", budget)
    selected["is_external_candidate"] = False
    selected["reliability_status"] = "nested_exact_reaction_validated"
    columns = [
        "query_id",
        "direction",
        "candidate_scope",
        "score_source",
        "selection_budget",
        "rank",
        "candidate_id",
        "base_score",
        "base_rank",
        "cage_available",
        "cage_rank",
        "calibrated_score",
        "residual_gain",
        "is_external_candidate",
        "reliability_status",
    ]
    return selected[columns].reset_index(drop=True)


def rank_current_library(
    reaction_id: str,
    top_k: int,
    results_dir: Path = DEFAULT_RESULTS,
) -> pd.DataFrame:
    budget = resolve_budget(top_k)
    if (results_dir / "nested_query_metrics.csv").exists():
        return rank_nested_fusion(reaction_id, top_k, budget, results_dir)
    return rank_legacy_panel(reaction_id, top_k, budget, results_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank current-library TPS enzyme candidates for a registered RHEA reaction. "
            "This route is restricted to the 1,391 current proteins and does not replace "
            "the open-world candidate route."
        )
    )
    parser.add_argument("--reaction-id", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = rank_current_library(
        args.reaction_id,
        args.top_k,
        args.results_dir.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(
        json.dumps(
            {
                "output": str(output),
                "n_results": len(result),
                "candidate_scope": "current_library_1391",
                "score_source": str(result.iloc[0]["score_source"]),
                "open_world_route_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
