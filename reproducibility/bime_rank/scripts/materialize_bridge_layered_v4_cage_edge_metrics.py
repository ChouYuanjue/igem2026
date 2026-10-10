from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
R2E_BASE = ROOT / "results/bridge_current_edgewise_r2e_v1/edge_metrics.csv.gz"
E2R_BASE = ROOT / "results/bridge_current_edgewise_e2r_v1/edge_metrics.csv.gz"
R2E_MEMBERSHIP = ROOT / "results/bridge_layered_v4_r2e_cage/pair_membership.csv.gz"
E2R_MEMBERSHIP = ROOT / "results/bridge_layered_v4_e2r_cage/pair_membership.csv.gz"
R2E_SCORES = ROOT / "results/bridge_layered_v4_r2e_cage/cage_inference/pairs_epoch_19.csv"
R2E_REQUIRED_SCORES = ROOT / "results/bridge_layered_v4_r2e_cage/cage_inference/required_scores.csv.gz"
R2E_REQUIRED_AUDIT = ROOT / "results/bridge_layered_v4_r2e_cage/cage_inference/required_scores_audit.json"
E2R_SCORES = ROOT / "results/bridge_layered_v4_e2r_cage/cage_inference/pairs_epoch_19.csv"
PROTEIN_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
OUT = ROOT / "results/bridge_layered_v4_cage_edge_metrics"

KEYS = ["protein_id", "reaction_id"]


def parse_targets(value: object) -> set[str]:
    return {x for x in str(value).split(";") if x}


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def filtered_target_rank(
    target: str,
    raw_rank: dict[str, int],
    candidate_count_filtered: int,
) -> int:
    value = raw_rank.get(target)
    if value is None:
        return int(candidate_count_filtered)
    ahead = sum(
        1
        for other, other_rank in raw_rank.items()
        if other != target and other_rank < value
    )
    return max(1, min(int(candidate_count_filtered), int(value - ahead)))


def direct_metrics(frame: pd.DataFrame, rank_col: str) -> dict[str, float | int]:
    ranks = pd.to_numeric(frame[rank_col], errors="raise").to_numpy(np.int64)
    return {
        "edges": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "hit3": float(np.mean(ranks <= 3)),
        "hit10": float(np.mean(ranks <= 10)),
        "hit100": float(np.mean(ranks <= 100)),
        "median_rank": float(np.median(ranks)),
    }


def load_base(direction: str) -> pd.DataFrame:
    path = R2E_BASE if direction == "r2e" else E2R_BASE
    frame = pd.read_csv(
        path,
        dtype={"protein_id": str, "reaction_id": str},
        usecols=["protein_id", "reaction_id", "candidate_count_filtered"],
    ).fillna("")
    frame["candidate_count_filtered"] = pd.to_numeric(
        frame.candidate_count_filtered,
        errors="raise",
    ).astype(int)
    if len(frame) != 23773 or frame.duplicated(KEYS).any():
        raise RuntimeError(
            f"{direction}: invalid base edge registry rows={len(frame)} "
            f"duplicates={int(frame.duplicated(KEYS).sum())}"
        )
    return frame


def evaluate_e2r() -> dict[str, object]:
    if not E2R_SCORES.exists():
        raise FileNotFoundError(E2R_SCORES)

    base = load_base("e2r")
    membership = pd.read_csv(E2R_MEMBERSHIP, dtype=str).fillna("")
    membership["native_pool"] = bool_col(membership.native_pool)
    membership["broad_pool"] = bool_col(membership.broad_pool)
    membership["cage_pair_supported"] = bool_col(membership.cage_pair_supported)

    scored = pd.read_csv(E2R_SCORES, dtype=str).fillna("")
    scored["pred_logit"] = pd.to_numeric(scored.pred_logit, errors="raise")
    score_key = ["protein_id", "reaction_id"]
    if scored.duplicated(score_key).any():
        raise RuntimeError("duplicate E2R CAGE scorer pairs")
    membership = membership.merge(
        scored[score_key + ["pred_logit"]],
        on=score_key,
        how="left",
        validate="one_to_one",
    )
    unexpected = membership[
        membership.pred_logit.notna() & ~membership.cage_pair_supported
    ]
    if len(unexpected):
        raise RuntimeError("E2R scorer contains unsupported membership rows")

    groups = {
        q: g for q, g in membership.groupby("protein_id", sort=False)
    }
    base_groups = {
        q: g for q, g in base.groupby("protein_id", sort=False)
    }

    records: list[dict[str, object]] = []
    for q, bg in base_groups.items():
        targets = bg.reaction_id.astype(str).tolist()
        nmap = {
            str(rid): int(n)
            for rid, n in bg[
                ["reaction_id", "candidate_count_filtered"]
            ].itertuples(index=False, name=None)
        }
        group = groups.get(str(q))
        route_rank: dict[str, dict[str, int]] = {}
        for name, flag in (
            ("enzymecage", "native_pool"),
            ("broad_cage", "broad_pool"),
        ):
            if group is None:
                ordered = pd.DataFrame(columns=membership.columns)
            else:
                ordered = group[
                    group[flag] & group.pred_logit.notna()
                ].sort_values(
                    ["pred_logit", "reaction_id"],
                    ascending=[False, True],
                    kind="stable",
                )
            rank_map = {
                str(rid): rank
                for rank, rid in enumerate(
                    ordered.reaction_id.astype(str).tolist(),
                    start=1,
                )
            }
            route_rank[name] = {
                rid: rank_map[rid]
                for rid in targets
                if rid in rank_map
            }

        for rid in targets:
            n = nmap[rid]
            records.append(
                {
                    "protein_id": str(q),
                    "reaction_id": rid,
                    "enzymecage_rank": filtered_target_rank(
                        rid,
                        route_rank["enzymecage"],
                        n,
                    ),
                    "enzymecage_candidate_count_filtered": n,
                    "broad_cage_rank": filtered_target_rank(
                        rid,
                        route_rank["broad_cage"],
                        n,
                    ),
                    "broad_cage_candidate_count_filtered": n,
                }
            )

    out = pd.DataFrame(records)
    if len(out) != 23773 or out.duplicated(KEYS).any():
        raise RuntimeError(
            f"E2R edge output invalid rows={len(out)} "
            f"duplicates={int(out.duplicated(KEYS).sum())}"
        )
    path = OUT / "e2r_edge_metrics.csv.gz"
    out.to_csv(path, index=False)
    return {
        "status": "completed",
        "output": str(path.relative_to(ROOT)),
        "candidate_universe_policy": (
            "Both E2R CAGE routes share the native full reaction universe; "
            "unreturned or unsupported targets receive pessimistic last rank."
        ),
        "metrics": {
            "EnzymeCAGE": direct_metrics(out, "enzymecage_rank"),
            "Broad Retrieval + CAGE Reranking": direct_metrics(
                out, "broad_cage_rank"
            ),
        },
    }


def evaluate_r2e() -> dict[str, object]:
    score_path = (
        R2E_REQUIRED_SCORES
        if R2E_REQUIRED_SCORES.exists()
        else R2E_SCORES
    )
    if not score_path.exists():
        raise FileNotFoundError(
            "R2E scorer output is not complete yet: neither compact "
            f"required scores nor full scorer output exists"
        )
    if score_path == R2E_REQUIRED_SCORES:
        if not R2E_REQUIRED_AUDIT.exists():
            raise RuntimeError("required score audit is missing")
        required_audit = json.loads(R2E_REQUIRED_AUDIT.read_text())
        if not bool(required_audit.get("complete", False)):
            raise RuntimeError(
                "required score audit is incomplete: "
                f"{required_audit.get('missing_pairs')}"
            )
    else:
        required_audit = None

    base = load_base("r2e")
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
    membership["cage_pair_supported"] = bool_col(
        membership.cage_pair_supported
    )
    membership = membership.drop_duplicates(
        ["route", "reaction_id", "logical_candidate_id"],
        keep="first",
    )

    scored = pd.read_csv(score_path, dtype=str).fillna("")
    scored["pred_logit"] = pd.to_numeric(scored.pred_logit, errors="raise")
    score_key = ["reaction_id", "UniprotID"]
    if scored.duplicated(score_key).any():
        raise RuntimeError("duplicate R2E CAGE scorer pairs")
    membership = membership.merge(
        scored[score_key + ["pred_logit"]],
        left_on=["reaction_id", "score_uid"],
        right_on=["reaction_id", "UniprotID"],
        how="left",
        validate="many_to_one",
    )
    unexpected = membership[
        membership.pred_logit.notna() & ~membership.cage_pair_supported
    ]
    if len(unexpected):
        raise RuntimeError("R2E scorer contains unsupported membership rows")

    current_ids = set(
        pd.read_csv(
            PROTEIN_UNIVERSE,
            sep="\t",
            dtype=str,
            usecols=["protein_id"],
        ).protein_id.astype(str)
    )
    native_ids = set(
        membership.loc[
            membership.route.eq("enzymecage"),
            "logical_candidate_id",
        ].astype(str)
    )
    broad_ids = set(
        membership.loc[
            membership.route.eq("broad_cage"),
            "logical_candidate_id",
        ].astype(str)
    )
    native_external = native_ids - current_ids
    broad_external = broad_ids - current_ids
    if broad_external:
        raise RuntimeError(
            f"Broad+CAGE unexpectedly has {len(broad_external)} candidates "
            "outside the current protein universe"
        )

    groups = {
        (route, q): g
        for (route, q), g in membership.groupby(
            ["route", "reaction_id"],
            sort=False,
        )
    }
    base_groups = {
        q: g for q, g in base.groupby("reaction_id", sort=False)
    }

    records: list[dict[str, object]] = []
    for q, bg in base_groups.items():
        targets = bg.protein_id.astype(str).tolist()
        target_set = set(targets)
        std_n = {
            str(pid): int(n)
            for pid, n in bg[
                ["protein_id", "candidate_count_filtered"]
            ].itertuples(index=False, name=None)
        }

        raw_by_route: dict[str, dict[str, int]] = {}
        for route in ("enzymecage", "broad_cage"):
            group = groups.get((route, str(q)))
            if group is None:
                ordered = pd.DataFrame(columns=membership.columns)
            else:
                ordered = group[group.pred_logit.notna()].sort_values(
                    ["pred_logit", "logical_candidate_id"],
                    ascending=[False, True],
                    kind="stable",
                )
            target_raw: dict[str, int] = {}
            for rank, rec in enumerate(
                ordered.itertuples(index=False),
                start=1,
            ):
                mapped = parse_targets(rec.target_protein_ids) & target_set
                for pid in mapped:
                    previous = target_raw.get(pid)
                    if previous is None or rank < previous:
                        target_raw[pid] = rank
            raw_by_route[route] = target_raw

        for pid in targets:
            native_n = std_n[pid] + len(native_external)
            broad_n = std_n[pid]
            records.append(
                {
                    "protein_id": pid,
                    "reaction_id": str(q),
                    "enzymecage_rank": filtered_target_rank(
                        pid,
                        raw_by_route["enzymecage"],
                        native_n,
                    ),
                    "enzymecage_candidate_count_filtered": native_n,
                    "broad_cage_rank": filtered_target_rank(
                        pid,
                        raw_by_route["broad_cage"],
                        broad_n,
                    ),
                    "broad_cage_candidate_count_filtered": broad_n,
                }
            )

    out = pd.DataFrame(records)
    if len(out) != 23773 or out.duplicated(KEYS).any():
        raise RuntimeError(
            f"R2E edge output invalid rows={len(out)} "
            f"duplicates={int(out.duplicated(KEYS).sum())}"
        )
    path = OUT / "r2e_edge_metrics.csv.gz"
    out.to_csv(path, index=False)
    return {
        "status": "completed",
        "output": str(path.relative_to(ROOT)),
        "score_source": str(score_path.relative_to(ROOT)),
        "required_score_audit": required_audit,
        "candidate_universe_policy": {
            "EnzymeCAGE": (
                "native full protein universe extended by all logical "
                f"CAGE-only identities: +{len(native_external)}"
            ),
            "Broad Retrieval + CAGE Reranking": (
                "current full protein universe; no external logical IDs"
            ),
            "tail_policy": (
                "unreturned or unsupported targets receive pessimistic "
                "last rank; exact-sequence/historical aliases may satisfy "
                "their mapped current target edge"
            ),
        },
        "audit": {
            "current_protein_universe": len(current_ids),
            "native_unique_logical_candidates": len(native_ids),
            "native_external_logical_candidates": len(native_external),
            "broad_unique_logical_candidates": len(broad_ids),
            "broad_external_logical_candidates": len(broad_external),
        },
        "metrics": {
            "EnzymeCAGE": direct_metrics(out, "enzymecage_rank"),
            "Broad Retrieval + CAGE Reranking": direct_metrics(
                out, "broad_cage_rank"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--direction",
        choices=("r2e", "e2r", "both"),
        default="both",
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "schema": "bridge-layered-v4-cage-edge-metrics",
        "status": "completed",
        "protocol": {
            "evaluation_unit": "one held-out relation edge",
            "edge_filtering": (
                "all other held-out positives for the same query are "
                "removed by rank correction"
            ),
            "gate_completion": (
                "ranked/scored gate candidates precede the unreturned tail; "
                "a target absent from the returned/scored set receives the "
                "pessimistic last rank of the method's full eligible universe"
            ),
            "partial_scorer_outputs_allowed": False,
        },
    }

    if args.direction in ("e2r", "both"):
        result["e2r"] = evaluate_e2r()
    if args.direction in ("r2e", "both"):
        result["r2e"] = evaluate_r2e()

    summary_path = OUT / "summary.json"
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        if previous.get("schema") == result["schema"]:
            for direction in ("r2e", "e2r"):
                if direction in previous and direction not in result:
                    result[direction] = previous[direction]

    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
