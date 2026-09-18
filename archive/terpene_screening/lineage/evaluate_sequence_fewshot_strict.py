from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_CANDIDATES = ROOT / "data/terpene/all_seq_terpene_synthase.tsv"
DEFAULT_EMBEDDINGS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_CLUSTERS = ROOT / "data/terpene_sequence_clusters/clusters_id50.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_sequence_fewshot_strict"
DEFAULT_M_VALUES = (1, 2, 3, 5)
DEFAULT_BUDGETS = (5, 10, 20)
DEFAULT_REPEATS = 20
DEFAULT_SEED = 20260707


def stable_trial_seed(base_seed: int, scope: str, reaction_id: str, m: int, rep: int) -> int:
    payload = f"{base_seed}|{scope}|{reaction_id}|{m}|{rep}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def kmers(sequence: str, k: int = 3) -> set[str]:
    sequence = re.sub(r"[^A-Z]", "", str(sequence).upper())
    return {sequence[index : index + k] for index in range(len(sequence) - k + 1)} if len(sequence) >= k else set()


def build_kmer_index(sequences: dict[str, str]) -> tuple[dict[str, set[str]], dict[str, int], dict[str, set[str]]]:
    by_entry = {entry: kmers(sequence) for entry, sequence in sequences.items()}
    sizes = {entry: len(values) for entry, values in by_entry.items()}
    inverted: dict[str, set[str]] = defaultdict(set)
    for entry, values in by_entry.items():
        for value in values:
            inverted[value].add(entry)
    return by_entry, sizes, inverted


def kmer_max_scores(
    seeds: tuple[str, ...],
    by_entry: dict[str, set[str]],
    sizes: dict[str, int],
    inverted: dict[str, set[str]],
) -> dict[str, float]:
    best: dict[str, float] = defaultdict(float)
    for seed in seeds:
        seed_kmers = by_entry.get(seed, set())
        if not seed_kmers:
            continue
        intersections: dict[str, int] = defaultdict(int)
        for value in seed_kmers:
            for candidate in inverted.get(value, ()):
                intersections[candidate] += 1
        for candidate, intersection in intersections.items():
            denominator = len(seed_kmers) + sizes.get(candidate, 0) - intersection
            if denominator:
                score = intersection / denominator
                if score > best[candidate]:
                    best[candidate] = score
    return dict(best)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(matrix, axis=1, keepdims=True)
    denominator[denominator == 0] = 1
    return matrix / denominator


def rank_from_scores(entry_list: list[str], scores: np.ndarray | dict[str, float]) -> list[str]:
    if isinstance(scores, np.ndarray):
        order = np.argsort(-scores, kind="stable")
        return [entry_list[index] for index in order]
    return sorted(entry_list, key=lambda entry: (-scores.get(entry, 0.0), entry))


def append_metrics(
    records: list[dict[str, object]],
    scope: str,
    reaction_id: str,
    m: int,
    rep: int,
    method: str,
    ranking: list[str],
    seeds: set[str],
    hidden: set[str],
    budgets: tuple[int, ...],
) -> None:
    ranking = [entry for entry in ranking if entry not in seeds]
    for budget in budgets:
        panel = ranking[:budget]
        hits = sum(entry in hidden for entry in panel)
        records.append(
            {
                "scope": scope,
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


def sample_trial(
    scope: str,
    positives: list[str],
    clusters: dict[str, str],
    m: int,
    rng: random.Random,
) -> tuple[tuple[str, ...], set[str]] | None:
    if scope == "random_positive":
        if len(positives) < m + 1:
            return None
        seeds = tuple(sorted(rng.sample(positives, m)))
        return seeds, set(positives) - set(seeds)

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for entry in positives:
        by_cluster[clusters.get(entry, entry)].append(entry)
    cluster_ids = sorted(by_cluster)
    if len(cluster_ids) < m + 1:
        return None
    seed_clusters = sorted(rng.sample(cluster_ids, m))
    seeds = tuple(sorted(rng.choice(sorted(by_cluster[cluster_id])) for cluster_id in seed_clusters))
    hidden = {
        entry
        for cluster_id, entries in by_cluster.items()
        if cluster_id not in set(seed_clusters)
        for entry in entries
    }
    return seeds, hidden


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict sequence few-shot evaluation for terpene synthases.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
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
        raise ValueError("Embedding matrix and entry map have different lengths.")
    matrix = normalize_rows(matrix)
    entry_list = entries["Entry"].astype(str).tolist()
    entry_to_row = {entry: index for index, entry in enumerate(entry_list)}
    entry_set = set(entry_list)

    candidate_frame = pd.read_csv(args.candidates, sep="\t", dtype=str).fillna("").drop_duplicates("Entry")
    sequences = dict(zip(candidate_frame["Entry"].astype(str), candidate_frame["Sequence"].astype(str)))
    kmer_by_entry, kmer_sizes, inverted = build_kmer_index(sequences)

    cluster_frame = pd.read_csv(args.clusters, dtype=str)
    cluster_map = dict(zip(cluster_frame["entry"].astype(str), cluster_frame["cluster_id"].astype(str)))

    positive_frame = pd.read_csv(args.positives, sep="\t", dtype=str).fillna("")
    positives_by_reaction = (
        positive_frame.groupby("rhea_id")["Entry"]
        .apply(lambda values: sorted(set(values.astype(str)) & entry_set))
        .to_dict()
    )

    records: list[dict[str, object]] = []
    eligible_counts: dict[str, dict[int, int]] = defaultdict(dict)
    ranking_cache: dict[tuple[str, ...], dict[str, list[str]]] = {}
    for scope in ("random_positive", "protein_cluster_cold"):
        for m in m_values:
            eligible = []
            for reaction_id, positives in positives_by_reaction.items():
                if sample_trial(scope, positives, cluster_map, m, random.Random(0)) is not None:
                    eligible.append(reaction_id)
            eligible_counts[scope][m] = len(eligible)
            for reaction_id in sorted(eligible):
                positives = positives_by_reaction[reaction_id]
                for rep in range(args.repeats):
                    rng = random.Random(stable_trial_seed(args.seed, scope, reaction_id, m, rep))
                    sampled = sample_trial(scope, positives, cluster_map, m, rng)
                    if sampled is None:
                        continue
                    seeds, hidden = sampled
                    if seeds not in ranking_cache:
                        seed_rows = np.array([entry_to_row[entry] for entry in seeds], dtype=np.int64)
                        similarities = matrix @ matrix[seed_rows].T
                        max_scores = similarities.max(axis=1)
                        centroid = normalize_rows(matrix[seed_rows].mean(axis=0, keepdims=True))[0]
                        centroid_scores = matrix @ centroid
                        jaccard_scores = kmer_max_scores(seeds, kmer_by_entry, kmer_sizes, inverted)
                        ranking_cache[seeds] = {
                            "esmc_max_cosine": rank_from_scores(entry_list, max_scores),
                            "esmc_centroid_cosine": rank_from_scores(entry_list, centroid_scores),
                            "kmer3_max_jaccard": rank_from_scores(entry_list, jaccard_scores),
                        }
                    for method, ranking in ranking_cache[seeds].items():
                        append_metrics(
                            records,
                            scope,
                            reaction_id,
                            m,
                            rep,
                            method,
                            ranking,
                            set(seeds),
                            hidden,
                            budgets,
                        )

    long = pd.DataFrame(records)
    long.to_csv(output_dir / "metrics_long.csv", index=False)
    aggregate = (
        long.groupby(["scope", "m", "method", "B"])
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

    best_rows = (
        aggregate.sort_values(
            ["scope", "m", "B", "hit_probability", "expected_hits"],
            ascending=[True, True, True, False, False],
        )
        .groupby(["scope", "m", "B"], as_index=False)
        .head(1)
    )
    best_rows.to_csv(output_dir / "best_methods.csv", index=False)
    summary = {
        "n_candidates": len(entry_list),
        "embedding_dim": int(matrix.shape[1]),
        "protein_clusters": str(args.clusters.resolve()),
        "eligible_counts": {scope: {str(m): count for m, count in values.items()} for scope, values in eligible_counts.items()},
        "m_values": m_values,
        "budgets": budgets,
        "repeats": args.repeats,
        "outputs": {
            "metrics_long": str(output_dir / "metrics_long.csv"),
            "metrics": str(output_dir / "metrics.csv"),
            "best_methods": str(output_dir / "best_methods.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(best_rows.to_string(index=False))


if __name__ == "__main__":
    main()
