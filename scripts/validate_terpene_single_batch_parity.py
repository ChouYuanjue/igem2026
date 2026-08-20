from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.rank_open_world import (
    DEFAULT_REGISTERED_PROTEIN_DIR,
    DEFAULT_REGISTERED_REACTIONS,
    load_external_reaction_rows,
    load_protein_library,
)


def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{completed.stderr[-4000:]}")


def compare(single: pd.DataFrame, batch: pd.DataFrame, direction: str, objective: str) -> dict[str, object]:
    batch = batch[(batch["direction"] == direction) & (batch["ranking_objective"] == objective)].sort_values("rank")
    single = single.sort_values("rank")
    single_ids = single["candidate_id"].astype(str).tolist()
    batch_ids = batch["candidate_id"].astype(str).tolist()
    route_equal = str(single.iloc[0]["route_id"]) == str(batch.iloc[0]["route_id"])
    source_equal = str(single.iloc[0]["score_source"]) == str(batch.iloc[0]["score_source"])
    return {
        "direction": direction,
        "objective": objective,
        "candidate_ids_equal": single_ids == batch_ids,
        "route_equal": route_equal,
        "score_source_equal": source_equal,
        "single_ids": single_ids,
        "batch_ids": batch_ids,
        "ok": single_ids == batch_ids and route_equal and source_equal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exact single-query versus vectorized-batch route parity.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/terpene_single_batch_parity.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    _, registered_ids = load_protein_library(DEFAULT_REGISTERED_PROTEIN_DIR)
    reactions = load_external_reaction_rows(DEFAULT_REGISTERED_REACTIONS)
    enzyme_id = registered_ids[0]
    reaction_id = str(reactions.iloc[0]["reaction_id"])
    comparisons: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="terpene_parity_") as temp:
        temp_path = Path(temp)
        batch_dir = temp_path / "batch"
        run([
            sys.executable,
            str(ROOT / "projects/active/terpene_screening/rank_registry_batch.py"),
            "--direction", "both",
            "--objectives", "3,10,20",
            "--max-queries", "1",
            "--include-known-associations",
            "--device", args.device,
            "--output-dir", str(batch_dir),
        ])
        batch_e2r = pd.read_csv(batch_dir / "enzyme_to_reaction_rankings.csv")
        batch_r2e = pd.read_csv(batch_dir / "reaction_to_enzyme_rankings.csv")
        for top_k in [3, 10, 20]:
            e2r_path = temp_path / f"single_e2r_{top_k}.csv"
            run([
                sys.executable,
                str(ROOT / "projects/active/terpene_screening/rank_open_world.py"),
                "rank-reactions", "--enzyme-id", enzyme_id,
                "--top-k", str(top_k), "--device", args.device,
                "--output", str(e2r_path),
            ])
            comparisons.append(compare(pd.read_csv(e2r_path), batch_e2r, "enzyme_to_reaction", f"top{top_k}"))
            r2e_path = temp_path / f"single_r2e_{top_k}.csv"
            run([
                sys.executable,
                str(ROOT / "projects/active/terpene_screening/rank_open_world.py"),
                "rank-enzymes", "--reaction-id", reaction_id,
                "--top-k", str(top_k), "--device", args.device,
                "--output", str(r2e_path),
            ])
            comparisons.append(compare(pd.read_csv(r2e_path), batch_r2e, "reaction_to_enzyme", f"top{top_k}"))
    failures = [value for value in comparisons if not value["ok"]]
    report = {
        "status": "passed" if not failures else "failed",
        "enzyme_id": enzyme_id,
        "reaction_id": reaction_id,
        "comparisons": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
