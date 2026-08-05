from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "reproducibility/golden/terpene_routes_v1"
CASES = {
    "golden_r2e_current_top3": [
        "rank-enzymes", "--reaction-id", "RHEA:54512", "--top-k", "3",
    ],
    "golden_e2r_current_top20": [
        "rank-reactions", "--enzyme-id", "7S5L_A", "--top-k", "20",
    ],
    "golden_r2e_external_top10": [
        "rank-enzymes", "--query-id", "golden_external_rxn",
        "--reaction-smiles", "CCO>>CC=O", "--top-k", "10",
    ],
}


def clean_value(value: object) -> object:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def normalize_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {key: clean_value(value) for key, value in record.items()}
        for record in records
    ]


def normalize(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    return normalize_records(frame[columns].to_dict("records"))


def main() -> int:
    comparisons = []
    with tempfile.TemporaryDirectory(prefix="terpene_golden_") as temp:
        for name, arguments in CASES.items():
            fixture_path = GOLDEN / f"{name}.json"
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            output = Path(temp) / f"{name}.csv"
            command = [
                sys.executable,
                str(ROOT / "projects/active/terpene_screening/rank_open_world.py"),
                *arguments,
                "--output", str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode:
                comparisons.append(
                    {
                        "case": name,
                        "ok": False,
                        "error": completed.stderr[-4000:],
                    }
                )
                continue
            actual = pd.read_csv(output)
            columns = [str(value) for value in fixture["columns"]]
            missing = sorted(set(columns) - set(actual.columns))
            records = normalize(actual, columns) if not missing else []
            expected = normalize_records(fixture["records"])
            comparisons.append(
                {
                    "case": name,
                    "ok": not missing and records == expected,
                    "missing_columns": missing,
                    "expected_rows": len(expected),
                    "actual_rows": len(records),
                    "first_expected": expected[0] if expected else None,
                    "first_actual": records[0] if records else None,
                }
            )
    failures = [value for value in comparisons if not value["ok"]]
    report = {"status": "passed" if not failures else "failed", "comparisons": comparisons, "failures": failures}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
