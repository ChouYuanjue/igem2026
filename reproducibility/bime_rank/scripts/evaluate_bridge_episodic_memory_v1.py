from __future__ import annotations

import argparse
import hashlib
import json
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from projects.active.bridge.runtime.memory import EPISODIC_FEATURE_NAMES, build_episodic_memory
from reproducibility.bime_rank.scripts import evaluate_bridge_adaptive_relation_context_v8 as long_eval
from reproducibility.bime_rank.scripts import evaluate_bridge_reciprocal_relation_context_v4 as base_eval

OUT = ROOT / "results/bridge_episodic_memory_v1"
ASSET = OUT / "episodic_memory_gate.validation.pkl"
GATES = np.asarray([0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.0], dtype=np.float64)
SUPPORT_SIZES = (1, 2, 4)


def metric(ranks: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": int(len(r)),
        "mrr": float(np.mean(1.0 / r)),
        "hit10": float(np.mean(r <= 10)),
        "hit100": float(np.mean(r <= 100)),
        "hit1000": float(np.mean(r <= 1000)),
        "median_rank": float(np.median(r)),
    }


def best_rank(score: np.ndarray, targets: np.ndarray, lex: np.ndarray) -> int:
    values = score[targets]
    best = float(values.max())
    tied = targets[values == best]
    row = int(tied[np.argmin(lex[tied])])
    return int(
        np.count_nonzero(score > best)
        + np.count_nonzero((score == best) & (lex < lex[row]))
        + 1
    )


def stable_order(query_id: str, ids: list[str]) -> list[str]:
    return sorted(
        ids,
        key=lambda value: hashlib.sha256(f"{query_id}\0{value}".encode()).hexdigest(),
    )


def support_size(query_id: str, positive_count: int) -> int:
    if positive_count < 2:
        return 0
    digest = int(hashlib.sha256(query_id.encode()).hexdigest()[:8], 16)
    wanted = SUPPORT_SIZES[digest % len(SUPPORT_SIZES)]
    return min(wanted, positive_count - 1)


def relation_weight(
    *,
    direction: str,
    gate: dict,
    base_z: np.ndarray,
    ctx: np.ndarray,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    available: np.ndarray,
    both: np.ndarray,
    query_degree: int,
    coherence: float,
) -> float:
    features = long_eval._features(
        base_z,
        ctx,
        comp_a,
        comp_b,
        available,
        both,
        query_degree,
        coherence,
    )
    asset = gate["directions"][direction]
    x = asset["scaler"].transform(features[None, :])
    probability = float(asset["permission_model"].predict_proba(x)[0, 1])
    strength = float(np.expm1(asset["strength_model"].predict(x)[0]))
    strength = float(np.clip(strength, 0.0, float(max(asset["alpha_support"]))))
    return float(
        np.clip(
            probability ** float(asset["permission_power"]) * strength,
            0.0,
            float(max(asset["alpha_support"])),
        )
    )


def collect_direction(
    direction: str,
    index: FibreCandidateIndex,
    validation: pd.DataFrame,
    train: pd.DataFrame,
    pproto: torch.Tensor,
    pmask: torch.Tensor,
    rproto: torch.Tensor,
    rmask: torch.Tensor,
    long_gate: dict,
    batch_size: int,
) -> pd.DataFrame:
    P, R = index.protein_embeddings, index.reaction_embeddings
    if direction == "r2e":
        groups = validation.groupby("reaction_id").protein_id.apply(lambda s: sorted(set(map(str, s)))).to_dict()
        qindex, cindex = index.reaction_index, index.protein_index
        candidate_ids, candidates, query_matrix = index.protein_ids, P, R
        known = train.groupby("reaction_id").protein_id.apply(lambda s: set(map(str, s))).to_dict()
        seen_mask = pmask.detach().cpu().numpy().astype(bool)
    else:
        groups = validation.groupby("protein_id").reaction_id.apply(lambda s: sorted(set(map(str, s)))).to_dict()
        qindex, cindex = index.protein_index, index.reaction_index
        candidate_ids, candidates, query_matrix = index.reaction_ids, R, P
        known = train.groupby("protein_id").reaction_id.apply(lambda s: set(map(str, s))).to_dict()
        seen_mask = rmask.detach().cpu().numpy().astype(bool)

    query_ids = [
        q for q in sorted(groups)
        if q in qindex and sum(x in cindex for x in groups[q]) >= 2
    ]
    lex = np.empty(len(candidate_ids), dtype=np.int64)
    lexical_order = np.argsort(np.asarray(candidate_ids, dtype=object), kind="stable")
    lex[lexical_order] = np.arange(len(lexical_order))

    records: list[dict[str, object]] = []
    for start in range(0, len(query_ids), batch_size):
        qs = query_ids[start : start + batch_size]
        qi = torch.as_tensor([qindex[q] for q in qs], dtype=torch.long, device=index.device)
        with torch.no_grad():
            if direction == "r2e":
                qemb = R.index_select(0, qi)
                raw_base = qemb @ P.T
                raw_a = rproto.index_select(0, qi) @ P.T
                av_a = rmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ pproto.T
                av_b = pmask[None, :].expand_as(raw_b)
            else:
                qemb = P.index_select(0, qi)
                raw_base = qemb @ R.T
                raw_a = pproto.index_select(0, qi) @ R.T
                av_a = pmask.index_select(0, qi)[:, None].expand_as(raw_a)
                raw_b = qemb @ rproto.T
                av_b = rmask[None, :].expand_as(raw_b)
            bz = base_eval.z_masked(raw_base, torch.ones_like(raw_base, dtype=torch.bool))
            za = base_eval.z_masked(raw_a, av_a)
            zb = base_eval.z_masked(raw_b, av_b)
            den = av_a.float() + av_b.float()
            ctx = (za + zb) / den.clamp_min(1.0)
            ctx[den == 0] = 0

        B = bz.cpu().numpy().astype(np.float64, copy=False)
        C = ctx.cpu().numpy().astype(np.float64, copy=False)
        A = za.cpu().numpy().astype(np.float64, copy=False)
        BB = zb.cpu().numpy().astype(np.float64, copy=False)
        AV = (den > 0).cpu().numpy()
        BOTH = (av_a & av_b).cpu().numpy()

        for j, query_id in enumerate(qs):
            positives = [x for x in stable_order(query_id, groups[query_id]) if x in cindex]
            n_support = support_size(query_id, len(positives))
            if n_support <= 0:
                continue
            support = positives[:n_support]
            targets = positives[n_support:]
            target_rows = np.asarray([cindex[x] for x in targets], dtype=np.int64)
            if not len(target_rows):
                continue

            train_ids = {x for x in known.get(query_id, set()) if x in cindex}
            train_rows = [cindex[x] for x in train_ids]
            coherence = long_eval._query_neighbor_coherence(train_rows, candidates)
            long_weight = relation_weight(
                direction=direction,
                gate=long_gate,
                base_z=B[j],
                ctx=C[j],
                comp_a=A[j],
                comp_b=BB[j],
                available=AV[j],
                both=BOTH[j],
                query_degree=len(train_rows),
                coherence=coherence,
            )
            train_score = B[j] + long_weight * C[j]

            memory = build_episodic_memory(
                query_embedding=qemb[j],
                candidate_embeddings=candidates,
                candidate_ids=candidate_ids,
                candidate_index=cindex,
                requested_support_ids=support,
                training_positive_ids=train_ids,
                entity_seen_mask=seen_mask,
                base_score=train_score,
                train_memory_weight=long_weight,
            )
            if memory is None:
                continue

            delta = memory.score - B[j]
            masked_rows = np.asarray(
                sorted(set(train_rows + [cindex[x] for x in support])),
                dtype=np.int64,
            )
            ranks: list[int] = []
            for gate in GATES:
                score = train_score + float(gate) * delta
                if len(masked_rows):
                    score = score.copy()
                    score[masked_rows] = -np.inf
                ranks.append(best_rank(score, target_rows, lex))
            ranks_arr = np.asarray(ranks, dtype=np.int64)
            oracle_index = int(np.argmin(ranks_arr))
            oracle_gate = float(GATES[oracle_index])
            row: dict[str, object] = {
                "direction": direction,
                "query_id": query_id,
                "support_count": len(support),
                "target_count": len(targets),
                "support_ids": ";".join(support),
                "base_rank": int(ranks_arr[0]),
                "oracle_gate": oracle_gate,
                "oracle_rank": int(ranks_arr[oracle_index]),
                "train_memory_weight": float(long_weight),
            }
            row.update(
                {
                    name: float(value)
                    for name, value in zip(EPISODIC_FEATURE_NAMES, memory.features, strict=True)
                }
            )
            for gate, rank in zip(GATES, ranks_arr, strict=True):
                row[f"rank_g_{gate:g}"] = int(rank)
            records.append(row)

        print(direction, min(start + batch_size, len(query_ids)), "/", len(query_ids), flush=True)
    return pd.DataFrame(records)


def ranks_for_gate(frame: pd.DataFrame, gates: np.ndarray) -> np.ndarray:
    nearest = np.abs(gates[:, None] - GATES[None, :]).argmin(1)
    matrix = frame[[f"rank_g_{g:g}" for g in GATES]].to_numpy(np.int64)
    return matrix[np.arange(len(frame)), nearest]


def fit_direction(frame: pd.DataFrame, random_state: int = 20261006) -> tuple[dict, dict]:
    X = frame[list(EPISODIC_FEATURE_NAMES)].to_numpy(np.float64)
    y = frame["oracle_gate"].to_numpy(np.float64)
    base = frame["base_rank"].to_numpy(np.int64)
    base_metrics = metric(base)

    configs = [
        {"max_iter": 80, "learning_rate": 0.04, "max_leaf_nodes": 5, "l2_regularization": 4.0, "min_samples_leaf": 40},
        {"max_iter": 120, "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 8.0, "min_samples_leaf": 50},
        {"max_iter": 160, "learning_rate": 0.02, "max_leaf_nodes": 7, "l2_regularization": 16.0, "min_samples_leaf": 70},
    ]
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    trials = []
    best = None
    for config in configs:
        pred = np.zeros(len(frame), dtype=np.float64)
        for train_idx, valid_idx in splitter.split(X):
            scaler = StandardScaler().fit(X[train_idx])
            model = HistGradientBoostingRegressor(random_state=random_state, **config)
            model.fit(scaler.transform(X[train_idx]), y[train_idx])
            pred[valid_idx] = np.clip(model.predict(scaler.transform(X[valid_idx])), 0.0, 1.0)
        ranks = ranks_for_gate(frame, pred)
        metrics = metric(ranks)
        safe = (
            metrics["mrr"] >= base_metrics["mrr"]
            and metrics["hit10"] >= base_metrics["hit10"]
            and metrics["hit100"] >= base_metrics["hit100"]
            and metrics["hit1000"] >= base_metrics["hit1000"]
            and metrics["median_rank"] <= base_metrics["median_rank"]
        )
        item = {
            "config": config,
            "safe": bool(safe),
            "base": base_metrics,
            "oof": metrics,
            "mean_gate": float(pred.mean()),
            "median_gate": float(np.median(pred)),
            "near_zero_fraction": float(np.mean(pred < 0.05)),
        }
        trials.append(item)
        key = (
            int(safe),
            metrics["mrr"],
            metrics["hit10"],
            metrics["hit100"],
            metrics["hit1000"],
            -metrics["median_rank"],
        )
        if best is None or key > best[0]:
            best = (key, config, item)
    assert best is not None

    selected = best[1]
    scaler = StandardScaler().fit(X)
    model = HistGradientBoostingRegressor(random_state=random_state, **selected)
    model.fit(scaler.transform(X), y)
    asset = {
        "schema": "bridge-episodic-memory-gate-v1",
        "direction": str(frame.direction.iloc[0]),
        "feature_names": list(EPISODIC_FEATURE_NAMES),
        "gate_support": GATES.tolist(),
        "gate_semantics": "continuous trust in runtime episodic support memory; 0 keeps the frozen BRIDGE prior, 1 replaces only the Broad query component with the support-memory component",
        "fit_surface": "firewalled validation relations only; support and target relations are absent from clean2023 and all frozen outer tests",
        "external_metrics_used": False,
        "scaler": scaler,
        "gate_model": model,
        "selected_config": selected,
    }
    return {
        "trials": trials,
        "selected": best[2],
        "oracle_gate_distribution": {
            "mean": float(y.mean()),
            "median": float(np.median(y)),
            "zero_fraction": float(np.mean(y == 0.0)),
        },
    }, asset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = FibreCandidateIndex(device=args.device)
    train = pd.read_csv(base_eval.TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    validation = base_eval.load_dev(train)
    pproto, pmask, rproto, rmask = base_eval.build_prototypes(index, train)
    with open(long_eval.ASSET, "rb") as handle:
        long_gate = pickle.load(handle)

    r2e = collect_direction(
        "r2e", index, validation, train, pproto, pmask, rproto, rmask, long_gate, 16
    )
    r2e.to_csv(OUT / "validation_r2e_episodes.csv.gz", index=False)
    e2r = collect_direction(
        "e2r", index, validation, train, pproto, pmask, rproto, rmask, long_gate, 128
    )
    e2r.to_csv(OUT / "validation_e2r_episodes.csv.gz", index=False)

    r_summary, r_asset = fit_direction(r2e)
    e_summary, e_asset = fit_direction(e2r)
    bundle = {
        "schema": "bridge-episodic-memory-gate-bundle-v1",
        "training_graph": "clean2023",
        "support_semantics": "only query-positive relations absent from clean2023 are episodic; clean2023 positives are filtered into long-term memory",
        "attention": "parameter-free query-to-support attention in frozen Broad embedding space",
        "adapter": "score-space equivalent of a one-step low-rank query update: prior + g(q,S)*(support_memory - Broad_query_component)",
        "outer_test_metrics_used": False,
        "directions": {"r2e": r_asset, "e2r": e_asset},
    }
    with open(ASSET, "wb") as handle:
        pickle.dump(bundle, handle)

    result = {
        "schema": "bridge-episodic-memory-validation-v1",
        "protocol": {
            "training_graph": "clean2023 only",
            "validation_relations": int(len(validation)),
            "episode_rule": "one deterministic support set per multi-positive validation query; all remaining positives are targets",
            "support_sizes": list(SUPPORT_SIZES),
            "query_firewall": "one episode per query, so no query appears in both train and validation folds of a CV split",
            "outer_test_used": False,
            "gate_range": [0.0, 1.0],
            "fixed_global_seed_weight": False,
            "fixed_rank_band": False,
        },
        "r2e": {"episodes": int(len(r2e)), **r_summary},
        "e2r": {"episodes": int(len(e2r)), **e_summary},
        "asset": str(ASSET.relative_to(ROOT)),
    }
    (OUT / "validation_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
