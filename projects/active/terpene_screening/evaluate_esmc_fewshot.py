from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_OUTPUT = ROOT / "results/terpene_esmc_fewshot"
DEFAULT_M_VALUES = (1, 2, 3, 5)
DEFAULT_BUDGETS = (5, 10, 20)
DEFAULT_REPEATS = 20
DEFAULT_SEED = 20260707


def stable_trial_seed(base_seed: int, reaction_id: str, m: int, rep: int) -> int:
    payload = f"{base_seed}|{reaction_id}|{m}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def evaluate_ranking(
    records: list[dict[str, object]],
    reaction_id: str,
    m: int,
    rep: int,
    method: str,
    ordered_entries: list[str],
    seeds: set[str],
    hidden: set[str],
    budgets: tuple[int, ...],
) -> None:
    ranking = [entry for entry in ordered_entries if entry not in seeds]
    for budget in budgets:
        panel = ranking[:budget]
        hits = sum(entry in hidden for entry in panel)
        records.append(
            {
                "reaction_id": reaction_id,
                "m": m,
                "rep": rep,
                "method": method,
                "B": budget,
                "n_hidden": len(hidden),
                "hits": hits,
                "hit": hits > 0,
                "precision": hits / budget,
                "hidden_recall": hits / len(hidden) if hidden else 0.0,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ESM-C few-shot retrieval for terpene synthases.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--m-values", default=",".join(str(value) for value in DEFAULT_M_VALUES))
    parser.add_argument("--budgets", default=",".join(str(value) for value in DEFAULT_BUDGETS))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    m_values = tuple(int(value) for value in args.m_values.split(",") if value)
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = pd.read_csv(args.embedding_dir / "entries.csv", dtype={"Entry": str}).sort_values("row")
    matrix = np.load(args.embedding_dir / "embeddings.npy").astype(np.float32)
    if len(entries) != len(matrix):
        raise ValueError(f"Embedding rows ({len(matrix)}) do not match entries ({len(entries)}).")
    matrix = normalize_rows(matrix)
    entry_list = entries["Entry"].astype(str).tolist()
    entry_to_row = {entry: index for index, entry in enumerate(entry_list)}

    positives = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives_by_reaction = (
        positives.groupby("rhea_id")["Entry"]
        .apply(lambda values: sorted(set(values.astype(str)) & set(entry_to_row)))
        .to_dict()
    )

    records: list[dict[str, object]] = []
    eligible_counts: dict[int, int] = {}
    for m in m_values:
        eligible = sorted(reaction_id for reaction_id, ids in positives_by_reaction.items() if len(ids) >= m + 1)
        eligible_counts[m] = len(eligible)
        for reaction_id in eligible:
            known = positives_by_reaction[reaction_id]
            for rep in range(args.repeats):
                rng = random.Random(stable_trial_seed(args.seed, reaction_id, m, rep))
                seeds = tuple(sorted(rng.sample(known, m)))
                seed_set = set(seeds)
                hidden = set(known) - seed_set
                seed_rows = np.array([entry_to_row[entry] for entry in seeds], dtype=np.int64)
                similarities = matrix @ matrix[seed_rows].T
                max_score = similarities.max(axis=1)
                centroid = normalize_rows(matrix[seed_rows].mean(axis=0, keepdims=True))[0]
                centroid_score = matrix @ centroid
                max_order = np.argsort(-max_score, kind="stable")
                centroid_order = np.argsort(-centroid_score, kind="stable")
                max_entries = [entry_list[index] for index in max_order]
                centroid_entries = [entry_list[index] for index in centroid_order]
                evaluate_ranking(records, reaction_id, m, rep, "esmc_max_cosine", max_entries, seed_set, hidden, budgets)
                evaluate_ranking(records, reaction_id, m, rep, "esmc_centroid_cosine", centroid_entries, seed_set, hidden, budgets)

    long = pd.DataFrame(records)
    long.to_csv(output_dir / "metrics_long.csv", index=False)
    aggregate = (
        long.groupby(["m", "method", "B"])
        .agg(
            n_trials=("hit", "size"),
            hit_probability=("hit", "mean"),
            expected_hits=("hits", "mean"),
            precision=("precision", "mean"),
            hidden_recall=("hidden_recall", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "positives": str(args.positives.resolve()),
        "embedding_dir": str(args.embedding_dir.resolve()),
        "n_embeddings": int(len(entry_list)),
        "embedding_dim": int(matrix.shape[1]),
        "eligible_counts": eligible_counts,
        "m_values": m_values,
        "budgets": budgets,
        "repeats": args.repeats,
        "outputs": {
            "metrics_long": str(output_dir / "metrics_long.csv"),
            "metrics": str(output_dir / "metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(aggregate.sort_values(["m", "B", "hit_probability"], ascending=[True, True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
