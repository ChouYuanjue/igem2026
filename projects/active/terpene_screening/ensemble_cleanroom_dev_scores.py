from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.train_cleanroom_rhea_retriever import evaluate_dev  # noqa: E402

KEYS = ["reaction_id", "protein_id"]


def ensemble_frames(paths: list[Path]) -> pd.DataFrame:
    if len(paths) < 2:
        raise ValueError("at least two member score files are required")
    merged: pd.DataFrame | None = None
    for index, path in enumerate(paths):
        frame = pd.read_csv(path, dtype={"reaction_id": str, "protein_id": str})
        required = set(KEYS) | {"label", "score"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame[KEYS + ["label", "score"]].drop_duplicates(KEYS)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{path} contains duplicate query-candidate pairs")
        frame = frame.rename(columns={"score": f"member_{index}_score", "label": f"label_{index}"})
        merged = frame if merged is None else merged.merge(frame, on=KEYS, how="outer", validate="one_to_one")
    assert merged is not None
    member_score_cols = [f"member_{i}_score" for i in range(len(paths))]
    label_cols = [f"label_{i}" for i in range(len(paths))]
    if merged[member_score_cols].isna().any().any():
        raise ValueError("ensemble members do not score identical query-candidate reservoirs")
    if merged[label_cols].isna().any().any():
        raise ValueError("ensemble members have incomplete labels")
    first = merged[label_cols[0]].astype(int)
    for column in label_cols[1:]:
        if not first.equals(merged[column].astype(int)):
            raise ValueError("ensemble members disagree on labels")
    merged["label"] = first
    merged["score"] = merged[member_score_cols].mean(axis=1)
    merged["score_std"] = merged[member_score_cols].std(axis=1, ddof=0)
    return merged[KEYS + ["label", "score", "score_std"] + member_score_cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean-score ensemble of cleanroom internal-dev runs.")
    parser.add_argument("--member", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    members = [path.resolve() for path in args.member]
    ensemble = ensemble_frames(members)
    metrics, query = evaluate_dev(ensemble)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    ensemble.to_csv(output / "dev_pair_scores.csv", index=False)
    query.to_csv(output / "dev_query_metrics.csv", index=False)
    summary = {
        "method": "cleanroom_mean_cosine_score_ensemble",
        "member_count": len(members),
        "members": [str(path) for path in members],
        "target_benchmark_labels_used": False,
        "dev_metrics": metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
