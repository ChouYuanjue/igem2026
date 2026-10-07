from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
R2E_MEMBERSHIP = ROOT / "results/bridge_layered_v4_r2e_cage/pair_membership.csv.gz"
E2R_MEMBERSHIP = ROOT / "results/bridge_layered_v4_e2r_cage/pair_membership.csv.gz"
R2E_BASE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_BASE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_SCORE_BASE = ROOT / "results/bridge_layered_v4_r2e_cage/cage_inference"
E2R_SCORES = ROOT / "results/bridge_layered_v4_e2r_cage/cage_inference/pairs_epoch_19.csv"
R2E_RELEVANT = ROOT / "results/bridge_layered_v4_r2e_cage/relevant_shards"
R2E_REQUIRED_SCORES = (
    ROOT / "results/bridge_layered_v4_r2e_cage/cage_inference/required_scores.csv.gz"
)
OUT = ROOT / "results/bridge_layered_v4_cage_edgewise"


def parse_ids(value: object) -> list[str]:
    return [x for x in str(value).split(";") if x]


def filtered_target_rank(
    target: str,
    scored_rank: dict[str, int],
    all_targets: list[str],
    candidate_count_filtered: int,
) -> int:
    raw = scored_rank.get(target)
    if raw is None:
        return int(candidate_count_filtered)
    earlier = sum(
        1
        for other in all_targets
        if other != target
        and other in scored_rank
        and scored_rank[other] < raw
    )
    return int(raw - earlier)


def collect_r2e_scores() -> pd.DataFrame:
    if R2E_REQUIRED_SCORES.exists():
        scores = pd.read_csv(
            R2E_REQUIRED_SCORES,
            dtype=str,
            usecols=["reaction_id", "UniprotID", "pred_logit"],
        ).fillna("")
        scores["pred_logit"] = pd.to_numeric(scores["pred_logit"], errors="raise")
        if scores.duplicated(["reaction_id", "UniprotID"]).any():
            raise RuntimeError("duplicate R2E required CAGE scores")
        return scores

    frames: list[pd.DataFrame] = []
    seen_paths: set[Path] = set()

    primary_final = R2E_SCORE_BASE / "pairs_epoch_19.csv"
    primary_partial = R2E_SCORE_BASE / "pairs_epoch_19.partial.csv"
    if primary_final.exists():
        paths = [primary_final]
    else:
        paths = [primary_partial] if primary_partial.exists() else []

    for i in range(8):
        out = R2E_RELEVANT / f"out_{i}"
        final = out / f"pairs_relevant_shard_{i}_epoch_19.csv"
        partial = out / f"pairs_relevant_shard_{i}_epoch_19.partial.csv"
        if final.exists():
            paths.append(final)
        elif partial.exists():
            paths.append(partial)

    for path in paths:
        if path in seen_paths or not path.exists() or path.stat().st_size == 0:
            continue
        seen_paths.add(path)
        frame = pd.read_csv(
            path,
            dtype=str,
            usecols=["reaction_id", "UniprotID", "pred_logit"],
        ).fillna("")
        frame["pred_logit"] = pd.to_numeric(frame["pred_logit"], errors="raise")
        frames.append(frame)

    if not frames:
        raise RuntimeError("no R2E CAGE score files found")
    scores = pd.concat(frames, ignore_index=True)
    scores = scores.sort_values(
        ["reaction_id", "UniprotID", "pred_logit"],
        ascending=[True, True, False],
        kind="stable",
    ).drop_duplicates(["reaction_id", "UniprotID"], keep="first")
    return scores


def evaluate_e2r() -> pd.DataFrame:
    base = pd.read_csv(E2R_BASE, dtype=str).fillna("")
    base["candidate_count_filtered"] = pd.to_numeric(
        base["candidate_count_filtered"], errors="raise"
    ).astype(int)
    membership = pd.read_csv(E2R_MEMBERSHIP, dtype=str).fillna("")
    for col in ("native_pool", "broad_pool", "cage_pair_supported"):
        membership[col] = membership[col].str.lower().eq("true")

    scores = pd.read_csv(
        E2R_SCORES,
        dtype=str,
        usecols=["protein_id", "reaction_id", "pred_logit"],
    ).fillna("")
    scores["pred_logit"] = pd.to_numeric(scores["pred_logit"], errors="raise")
    if scores.duplicated(["protein_id", "reaction_id"]).any():
        raise RuntimeError("duplicate E2R CAGE scores")

    membership = membership.merge(
        scores,
        on=["protein_id", "reaction_id"],
        how="left",
        validate="one_to_one",
    )
    groups = {q: g for q, g in membership.groupby("protein_id", sort=False)}
    target_groups = {
        q: g.copy() for q, g in base.groupby("protein_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    for q, tg in target_groups.items():
        target_ids = tg["reaction_id"].astype(str).tolist()
        g = groups.get(q)
        route_ranks: dict[str, dict[str, int]] = {}
        for route, flag in (("enzymecage", "native_pool"), ("broad_cage", "broad_pool")):
            if g is None:
                rank_map = {}
            else:
                scored = g[g[flag] & g["pred_logit"].notna()].copy()
                scored = scored.sort_values(
                    ["pred_logit", "reaction_id"],
                    ascending=[False, True],
                    kind="stable",
                )
                rank_map = {
                    rid: i
                    for i, rid in enumerate(scored["reaction_id"].astype(str), 1)
                }
            route_ranks[route] = rank_map

        for rec in tg.itertuples(index=False):
            n = int(rec.candidate_count_filtered)
            rows.append(
                {
                    "protein_id": str(rec.protein_id),
                    "reaction_id": str(rec.reaction_id),
                    "protein_seen": str(rec.protein_seen),
                    "reaction_seen": str(rec.reaction_seen),
                    "candidate_count_filtered": n,
                    "enzymecage_rank": filtered_target_rank(
                        str(rec.reaction_id),
                        route_ranks["enzymecage"],
                        target_ids,
                        n,
                    ),
                    "broad_cage_rank": filtered_target_rank(
                        str(rec.reaction_id),
                        route_ranks["broad_cage"],
                        target_ids,
                        n,
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) != len(base):
        raise RuntimeError(f"E2R edge count mismatch: {len(out)} != {len(base)}")
    return out


def evaluate_r2e() -> pd.DataFrame:
    base = pd.read_csv(R2E_BASE, dtype=str).fillna("")
    base["candidate_count_filtered"] = pd.to_numeric(
        base["candidate_count_filtered"], errors="raise"
    ).astype(int)
    membership = pd.read_csv(
        R2E_MEMBERSHIP,
        dtype=str,
        usecols=[
            "route",
            "reaction_id",
            "logical_candidate_id",
            "score_uid",
            "target_protein_ids",
            "cage_pair_supported",
        ],
    ).fillna("")
    membership["cage_pair_supported"] = (
        membership["cage_pair_supported"].str.lower().eq("true")
    )
    scores = collect_r2e_scores()
    membership = membership.merge(
        scores,
        left_on=["reaction_id", "score_uid"],
        right_on=["reaction_id", "UniprotID"],
        how="left",
        validate="many_to_one",
    )

    # Only route-specific queries whose gate contains a held-out positive need
    # complete scorer coverage for that route. Runtime-unscored pairs remain
    # unavailable and therefore rank as misses; no replacement score is made.
    targets_by_q = (
        base.groupby("reaction_id")["protein_id"]
        .agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    expected_parts: list[pd.DataFrame] = []
    for route, route_frame in membership.groupby("route", sort=False):
        relevant_queries: set[str] = set()
        for rec in route_frame.itertuples(index=False):
            mapped = set(parse_ids(rec.target_protein_ids))
            if mapped & targets_by_q.get(str(rec.reaction_id), set()):
                relevant_queries.add(str(rec.reaction_id))
        expected_parts.append(
            route_frame[
                route_frame["cage_pair_supported"]
                & route_frame["reaction_id"].isin(relevant_queries)
            ][["reaction_id", "score_uid"]]
        )
    expected = pd.concat(expected_parts, ignore_index=True).drop_duplicates()
    actual = scores[["reaction_id", "UniprotID"]].rename(
        columns={"UniprotID": "score_uid"}
    ).drop_duplicates()
    coverage = expected.merge(actual, on=["reaction_id", "score_uid"], how="left", indicator=True)
    missing = coverage["_merge"].ne("both")
    if missing.any():
        print(
            "R2E runtime-unscored pairs retained as unavailable: "
            f"{int(missing.sum())}/{len(expected)}",
            flush=True,
        )

    groups = {
        (q, route): g
        for (q, route), g in membership.groupby(["reaction_id", "route"], sort=False)
    }
    target_groups = {
        q: g.copy() for q, g in base.groupby("reaction_id", sort=False)
    }

    rows: list[dict[str, object]] = []
    for q, tg in target_groups.items():
        target_ids = tg["protein_id"].astype(str).tolist()
        route_ranks: dict[str, dict[str, int]] = {}
        for route in ("enzymecage", "broad_cage"):
            g = groups.get((q, route))
            score_by_pid: dict[str, float] = {}
            if g is not None:
                scored = g[g["cage_pair_supported"] & g["pred_logit"].notna()]
                for rec in scored.itertuples(index=False):
                    score = float(rec.pred_logit)
                    for pid in parse_ids(rec.target_protein_ids):
                        old = score_by_pid.get(pid)
                        if old is None or score > old:
                            score_by_pid[pid] = score
            ordered = sorted(score_by_pid, key=lambda pid: (-score_by_pid[pid], pid))
            route_ranks[route] = {pid: i for i, pid in enumerate(ordered, 1)}

        for rec in tg.itertuples(index=False):
            n = int(rec.candidate_count_filtered)
            pid = str(rec.protein_id)
            rows.append(
                {
                    "protein_id": pid,
                    "reaction_id": str(rec.reaction_id),
                    "protein_seen": str(rec.protein_seen),
                    "reaction_seen": str(rec.reaction_seen),
                    "candidate_count_filtered": n,
                    "enzymecage_rank": filtered_target_rank(
                        pid,
                        route_ranks["enzymecage"],
                        target_ids,
                        n,
                    ),
                    "broad_cage_rank": filtered_target_rank(
                        pid,
                        route_ranks["broad_cage"],
                        target_ids,
                        n,
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) != len(base):
        raise RuntimeError(f"R2E edge count mismatch: {len(out)} != {len(base)}")
    return out


def print_metrics(frame: pd.DataFrame, direction: str) -> None:
    print(f"\n## {direction.upper()}")
    for col in ("enzymecage_rank", "broad_cage_rank"):
        r = frame[col].to_numpy(np.int64)
        print(
            col,
            {
                "edges": int(len(r)),
                "mrr": float((1.0 / r).mean()),
                "hit3": float((r <= 3).mean()),
                "hit10": float((r <= 10).mean()),
                "hit100": float((r <= 100).mean()),
                "median_rank": float(np.median(r)),
            },
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=("r2e", "e2r", "both"), default="both")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.direction in ("e2r", "both"):
        frame = evaluate_e2r()
        frame.to_csv(OUT / "e2r_edge_metrics.csv.gz", index=False)
        print_metrics(frame, "e2r")
    if args.direction in ("r2e", "both"):
        frame = evaluate_r2e()
        frame.to_csv(OUT / "r2e_edge_metrics.csv.gz", index=False)
        print_metrics(frame, "r2e")


if __name__ == "__main__":
    main()
