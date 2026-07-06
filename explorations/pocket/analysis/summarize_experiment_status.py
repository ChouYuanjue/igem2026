from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: Any) -> str:
    if value is None:
        return "NA"
    text = str(value)
    return text if text else "NA"


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "No rows.\n"
    rows = df[columns].fillna("NA").astype(str).values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _dataset_column(matrix: pd.DataFrame) -> str:
    if "dataset" in matrix.columns:
        return "dataset"
    if "dataset_scale" in matrix.columns:
        return "dataset_scale"
    return "dataset"


def summarize(results_root: Path, matrix_csv: Path, output_path: Path) -> Path:
    matrix = pd.read_csv(matrix_csv) if matrix_csv.exists() else pd.DataFrame()
    env_smoke = results_root / "env_smoke_test.txt"
    comparison_report = results_root / "comparison/comparison_report.md"
    official_eval_runs = [
        results_root / "official_eval_enzyme405/run_summary.json",
        results_root / "official_eval_orphan335/run_summary.json",
    ]

    lines = [
        "# Pocket Experiment Status",
        "",
        "## Environment",
        "",
        f"- smoke_test: {'present' if env_smoke.exists() else 'missing'}",
        f"- smoke_test_path: {env_smoke}",
        "",
        "## Official Full Eval",
        "",
    ]
    for path in official_eval_runs:
        summary = _load_json(path)
        lines.append(f"- {path.parent.name}: status={summary.get('status', 'not_run')}, failed_step={summary.get('failed_step', 'NA')}")
    lines.extend(
        [
            "",
            "Official full eval is not the main result because the official configs reference missing precomputed feature/data paths. The showable results use derived smallsets and real EnzymeCAGE inference.",
            "",
            "## Completed Baselines",
            "",
        ]
    )
    if matrix.empty:
        lines.append("Final result matrix missing.")
    else:
        dataset_col = _dataset_column(matrix)
        completed = matrix[matrix["status"] == "completed"]
        blocked = matrix[matrix["status"].astype(str).str.startswith("blocked")]
        failed = matrix[matrix["status"].astype(str).str.startswith("failed")]
        columns = [dataset_col, "baseline", "status", "top5_success", "top10_success", "n_pairs", "n_pockets"]
        lines.append(_table(completed, columns).rstrip())
        lines.extend(["", "## Blocked Baselines", ""])
        lines.append(_table(blocked, [dataset_col, "baseline", "status", "blocked_reason"]).rstrip())
        lines.extend(["", "## Failed Baselines", ""])
        lines.append(_table(failed, [dataset_col, "baseline", "status", "blocked_reason"]).rstrip())

    lines.extend(
        [
            "",
            "## Key Result Files",
            "",
            f"- best_available_result_matrix_csv: {results_root / 'best_available_result_matrix.csv'}",
            f"- best_available_result_matrix_md: {results_root / 'best_available_result_matrix.md'}",
            f"- best_available_conclusion_md: {results_root / 'best_available_conclusion.md'}",
            f"- comparison_report: {comparison_report}",
            f"- enzymecage_patch: {results_root / 'patches/enzymecage_path_fixes.patch'}",
            "",
            "## Next Actions",
            "",
            "- Install fpocket and rerun geometry baselines.",
            "- Expand smallset size.",
            "- Add catalytic-residue-aware pocket prior.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize final pocket experiment status.")
    parser.add_argument("--results_root", default="results/pocket")
    parser.add_argument("--matrix_csv", default="results/pocket/final_result_matrix.csv")
    parser.add_argument("--output", default="results/pocket/experiment_status.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = summarize(Path(args.results_root), Path(args.matrix_csv), Path(args.output))
    print(output)


if __name__ == "__main__":
    main()
