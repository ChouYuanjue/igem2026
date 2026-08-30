from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from projects.active.terpene_screening.evaluate_lambdarank_stacking_double_cold import train_ranker
from projects.active.terpene_screening.train_dual_tower_cold import rank_metrics

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = ROOT / "results/terpene_cage_neural_common_reservoir_specialists_v1/pair_scores.csv"
DEFAULT_OUTPUT = ROOT / "results/terpene_common_reservoir_lambdarank"
DEFAULT_BUDGETS = (1, 3, 5, 10, 20)


def stable_rng(seed: int, *parts: object) -> np.random.Generator:
    token = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    local = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")
    return np.random.default_rng(local)


def source_columns(frame: pd.DataFrame) -> list[str]:
    direct = sorted(column for column in frame.columns if column.startswith("direct:"))
    if "pure_cage" not in frame.columns or not direct:
        raise ValueError("pair scores must contain pure_cage and at least one direct:* expert")
    return ["pure_cage", *direct]


def build_features(frame: pd.DataFrame, query_column: str, sources: list[str]) -> tuple[pd.DataFrame, list[str]]:
    features = pd.DataFrame(index=frame.index)
    names: list[str] = []
    for source in sources:
        value = frame[source].astype(float)
        grouped = frame.groupby(query_column)[source]
        percentile = grouped.rank(method="average", pct=True)
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, 1).fillna(1)
        gap = grouped.transform("max") - value
        for suffix, data in [("raw", value), ("pct", percentile), ("z", (value - mean) / std), ("gap_top", gap)]:
            name = f"{source}|{suffix}"
            features[name] = data
            names.append(name)
    expert_sources = [source for source in sources if source.startswith("direct:")]
    expert_pct = features[[f"{source}|pct" for source in expert_sources]]
    expert_z = features[[f"{source}|z" for source in expert_sources]]
    derived = {
        "expert_pct_mean": expert_pct.mean(axis=1),
        "expert_pct_std": expert_pct.std(axis=1),
        "expert_pct_min": expert_pct.min(axis=1),
        "expert_pct_max": expert_pct.max(axis=1),
        "expert_z_mean": expert_z.mean(axis=1),
        "expert_z_std": expert_z.std(axis=1),
        "cage_minus_expert_pct": features["pure_cage|pct"] - expert_pct.mean(axis=1),
    }
    for name, value in derived.items():
        features[name] = value
        names.append(name)
    return features.replace([np.inf, -np.inf], 0).fillna(0), names


def sample_training_rows(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    query_column: str,
    hard_negatives: int,
    random_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int], pd.DataFrame]:
    rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    group_sizes: list[int] = []
    audits: list[dict[str, object]] = []
    pct_columns = [column for column in features.columns if column.startswith("direct:") and column.endswith("|pct")]
    for query_id, group in frame.groupby(query_column, sort=True):
        positive_index = group.index[group["label"].astype(int).eq(1)].to_numpy()
        negative_index = group.index[group["label"].astype(int).eq(0)].to_numpy()
        if not len(positive_index) or not len(negative_index):
            continue
        hardness = features.loc[negative_index, pct_columns].mean(axis=1).to_numpy()
        order = np.lexsort((negative_index, -hardness))
        hard = negative_index[order[: min(hard_negatives, len(order))]]
        remaining = np.setdiff1d(negative_index, hard, assume_unique=False)
        rng = stable_rng(seed, query_id)
        n_random = min(random_negatives, len(remaining))
        random_rows = rng.choice(remaining, size=n_random, replace=False) if n_random else np.empty(0, dtype=negative_index.dtype)
        selected = np.concatenate([positive_index, hard, random_rows])
        y = np.concatenate([np.ones(len(positive_index), dtype=np.float32), np.zeros(len(hard) + len(random_rows), dtype=np.float32)])
        rows.append(features.loc[selected].to_numpy(np.float32))
        labels.append(y)
        group_sizes.append(len(selected))
        audits.append({
            "query_id": str(query_id),
            "positive_count": len(positive_index),
            "hard_negative_count": len(hard),
            "random_negative_count": len(random_rows),
            "group_size": len(selected),
        })
    if not rows:
        raise ValueError("No training ranking groups")
    return np.concatenate(rows), np.concatenate(labels), group_sizes, pd.DataFrame(audits)


def evaluate_scores(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    query_column: str,
    candidate_column: str,
    budgets: tuple[int, ...],
) -> pd.DataFrame:
    local = frame[[query_column, candidate_column, "label"]].copy()
    local["score"] = scores
    records: list[dict[str, object]] = []
    for query_id, group in local.groupby(query_column, sort=True):
        positives = set(group.loc[group["label"].astype(int).eq(1), candidate_column].astype(str))
        if not positives:
            continue
        metrics = rank_metrics(
            group["score"].to_numpy(float),
            group[candidate_column].astype(str).tolist(),
            positives,
            set(),
            budgets,
        )
        records.append({"query_id": str(query_id), **metrics})
    return pd.DataFrame(records)


def aggregate(frame: pd.DataFrame, budgets: tuple[int, ...]) -> dict[str, float]:
    result = {"n_queries": int(len(frame)), "mrr": float(frame["reciprocal_rank"].mean())}
    for budget in budgets:
        result[f"hit_at_{budget}"] = float(frame[f"hit_at_{budget}"].mean())
        result[f"positive_recall_at_{budget}"] = float(frame[f"positive_recall_at_{budget}"].mean())
    return result


def crossfit_direction(
    pair_scores: pd.DataFrame,
    *,
    direction: str,
    query_column: str,
    candidate_column: str,
    budgets: tuple[int, ...],
    folds: int,
    objective: str,
    rounds: int,
    max_depth: int,
    learning_rate: float,
    hard_negatives: int,
    random_negatives: int,
    threads: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = pair_scores.groupby(query_column)["label"].sum()
    query_ids = sorted(eligible[eligible > 0].index.astype(str))
    frame = pair_scores[pair_scores[query_column].astype(str).isin(query_ids)].copy()
    sources = source_columns(frame)
    features, feature_names = build_features(frame, query_column, sources)
    unique_queries = np.asarray(query_ids)
    dummy = np.zeros(len(unique_queries))
    oof = pd.Series(np.nan, index=frame.index, dtype=float)
    audits: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for fold, (train_q_idx, test_q_idx) in enumerate(GroupKFold(n_splits=folds).split(unique_queries, dummy, unique_queries)):
        train_queries = set(unique_queries[train_q_idx]); test_queries = set(unique_queries[test_q_idx])
        train_frame = frame[frame[query_column].astype(str).isin(train_queries)]
        test_frame = frame[frame[query_column].astype(str).isin(test_queries)]
        x_train, y_train, groups, audit = sample_training_rows(
            train_frame,
            features,
            query_column=query_column,
            hard_negatives=hard_negatives,
            random_negatives=random_negatives,
            seed=seed + fold,
        )
        booster = train_ranker(
            x_train, y_train, groups, objective, rounds, max_depth, learning_rate,
            1.0, 5.0, threads, seed + fold,
        )
        oof.loc[test_frame.index] = booster.predict(xgb.DMatrix(features.loc[test_frame.index].to_numpy(np.float32)))
        audit.insert(0, "fold", fold); audit.insert(1, "direction", direction); audits.append(audit)
        gain = booster.get_score(importance_type="gain")
        importances.append(pd.DataFrame({
            "fold": fold,
            "direction": direction,
            "feature": feature_names,
            "gain": [float(gain.get(f"f{i}", 0.0)) for i in range(len(feature_names))],
        }))
    if oof.isna().any():
        raise AssertionError("OOF LambdaRank scores are incomplete")
    query_metrics = evaluate_scores(
        frame, oof.to_numpy(), query_column=query_column, candidate_column=candidate_column, budgets=budgets
    )
    query_metrics.insert(0, "direction", direction)
    return query_metrics, pd.concat(audits, ignore_index=True), pd.concat(importances, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-fitted LambdaRank reranking of CAGE + retrieval expert scores on a common pair reservoir.")
    parser.add_argument("--pair-scores", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objective", choices=["rank:pairwise", "rank:ndcg"], default="rank:ndcg")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--hard-negatives", type=int, default=96)
    parser.add_argument("--random-negatives", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.folds < 2 or args.rounds <= 0:
        raise ValueError("folds must be >=2 and rounds positive")
    budgets = tuple(sorted({int(v) for v in args.budgets.split(",") if v}))
    pairs = pd.read_csv(args.pair_scores)
    outputs=[]; audits=[]; importances=[]
    for direction, qcol, ccol in [
        ("reaction_to_enzyme", "reaction_id", "uniprot_id"),
        ("enzyme_to_reaction", "uniprot_id", "reaction_id"),
    ]:
        q,a,i = crossfit_direction(
            pairs, direction=direction, query_column=qcol, candidate_column=ccol,
            budgets=budgets, folds=args.folds, objective=args.objective, rounds=args.rounds,
            max_depth=args.max_depth, learning_rate=args.learning_rate,
            hard_negatives=args.hard_negatives, random_negatives=args.random_negatives,
            threads=args.threads, seed=args.seed,
        )
        outputs.append(q); audits.append(a); importances.append(i)
    query_metrics=pd.concat(outputs,ignore_index=True)
    summary=pd.DataFrame([
        {"direction":direction, **aggregate(group,budgets)}
        for direction,group in query_metrics.groupby("direction",sort=True)
    ])
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    query_metrics.to_csv(out/"query_metrics.csv",index=False)
    summary.to_csv(out/"metrics.csv",index=False)
    pd.concat(audits,ignore_index=True).to_csv(out/"training_group_audit.csv",index=False)
    pd.concat(importances,ignore_index=True).to_csv(out/"feature_importance.csv",index=False)
    payload={
        "method":"cross_fitted_lambdarank_expert_stacker",
        "reused_trainer":"evaluate_lambdarank_stacking_double_cold.train_ranker",
        "pair_scores":str(args.pair_scores.resolve()),
        "objective":args.objective,"folds":args.folds,"rounds":args.rounds,
        "hard_negatives":args.hard_negatives,"random_negatives":args.random_negatives,
        "pure_cage_in_features":True,
        "test_query_labels_used_for_training":False,
    }
    (out/"summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
