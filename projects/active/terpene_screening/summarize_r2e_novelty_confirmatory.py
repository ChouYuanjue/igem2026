from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_CELLS = (
    "reactzyme_reaction_projected_double_cold",
    "temporal_post2020_double_cold",
    "broad_reaction_hash_cold_protein_seen",
)
METRICS = (
    "mrr", "map", "macro_roc_auc", "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
    "hit_at_20", "hit_at_50", "ndcg_at_10", "top1_percent_ef", "median_best_positive_rank",
)


def load_metric(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]["reaction_to_enzyme"]


def improvement(metric: str, baseline: float, value: float) -> float:
    return baseline - value if metric == "median_best_positive_rank" else value - baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen R2E novelty expert/router confirmatory results.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/broad_rhea_novelty_confirmatory_v1"))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for cell in DEFAULT_CELLS:
        base = load_metric(Path(f"results/broad_rhea_full_candidate_nested_selected_v1/{cell}/{cell}/summary.json"))
        expert = load_metric(Path(f"results/broad_rhea_full_candidate_novelty_expert_frozen_v1/{cell}/{cell}/summary.json"))
        route_payload = json.loads(Path(f"results/broad_rhea_novelty_route_frozen_v1/{cell}/summary.json").read_text(encoding="utf-8"))
        routed = route_payload["summaries"]["routed"]
        for model, metrics in [("backbone", base), ("novelty_expert", expert), ("frozen_gate_route", routed)]:
            row: dict[str, object] = {
                "cell": cell,
                "model": model,
                "route_guard_pass": route_payload["guard"]["pass"] if model == "frozen_gate_route" else None,
                "expert_query_fraction": route_payload["expert_query_fraction"] if model == "frozen_gate_route" else None,
            }
            for metric in METRICS:
                row[metric] = metrics[metric]
                if model != "backbone":
                    row[f"{metric}_improvement"] = improvement(metric, float(base[metric]), float(metrics[metric]))
            rows.append(row)
    frame = pd.DataFrame(rows)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "metrics.csv", index=False)

    expert = frame[frame.model.eq("novelty_expert")]
    routed = frame[frame.model.eq("frozen_gate_route")]
    positive_counts = {
        metric: int((expert[f"{metric}_improvement"] > 0).sum())
        for metric in METRICS
    }
    summary = {
        "protocol": "pre-frozen_novelty_expert_confirmatory_v1",
        "selection_commit": "00ad433",
        "confirmatory_cells": list(DEFAULT_CELLS),
        "expert_positive_cell_count_by_metric": positive_counts,
        "router_guard_pass_count": int(sum(value is True for value in routed["route_guard_pass"].tolist())),
        "router_guard_cell_count": int(len(routed)),
        "interpretation": (
            "The novelty expert generalizes across clean protocols, while the fixed similarity<0.5 hard router "
            "does not satisfy the no-regression guard on every protocol. Do not retune the v1 router on these outer labels."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cols = ["cell", "model", "mrr", "map", "macro_roc_auc", "hit_at_1", "hit_at_10", "hit_at_50", "ndcg_at_10", "median_best_positive_rank", "route_guard_pass"]
    view = frame[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    lines = ["# Frozen R2E novelty confirmatory v1", "", f"Selection freeze: `{summary['selection_commit']}`", "", header, separator, *body, "", summary["interpretation"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(frame[cols].to_string(index=False))


if __name__ == "__main__":
    main()
