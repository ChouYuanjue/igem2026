from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from projects.active.bridge.model.assets import ROOT

BASE = ROOT / "results/bridge_layered_v4_r2e_cage"
TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
MEMBERSHIP = BASE / "pair_membership.csv.gz"
CENTRAL_PARTIAL = BASE / "cage_inference/pairs_epoch_19.partial.csv"
OUT = BASE / "cage_inference/required_scores.csv.gz"
AUDIT = BASE / "cage_inference/required_scores_audit.json"


def parse_targets(value: object) -> set[str]:
    return {x for x in str(value).split(";") if x}


def required_pairs() -> pd.DataFrame:
    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(
        ["protein_id", "reaction_id"]
    )
    positives = (
        targets.groupby("reaction_id")
        .protein_id.agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    membership = pd.read_csv(
        MEMBERSHIP,
        dtype=str,
        usecols=[
            "route",
            "reaction_id",
            "target_protein_ids",
            "score_uid",
            "cage_pair_supported",
        ],
    ).fillna("")
    membership["supported"] = (
        membership.cage_pair_supported.astype(str).str.lower().eq("true")
    )

    parts = []
    route_queries = {}
    for route, group in membership.groupby("route", sort=True):
        good = set()
        for q, mapped in group[
            ["reaction_id", "target_protein_ids"]
        ].itertuples(index=False, name=None):
            if parse_targets(mapped) & positives.get(str(q), set()):
                good.add(str(q))
        route_queries[str(route)] = len(good)
        part = group[
            group.reaction_id.astype(str).isin(good)
            & group.supported
        ][["reaction_id", "score_uid"]]
        parts.append(part)

    need = pd.concat(parts, ignore_index=True).drop_duplicates()
    need = need.rename(columns={"score_uid": "UniprotID"})
    need.attrs["route_queries"] = route_queries
    return need


def source_paths() -> list[Path]:
    paths = [CENTRAL_PARTIAL]
    minimal_cache_final = (
        BASE
        / "minimal_cache/inference/pairs_minimal_all_epoch_19.csv"
    )
    if minimal_cache_final.exists():
        paths.append(minimal_cache_final)
    patterns = [
        str(BASE / "cage_inference/pairs_minimal_shard_*_epoch_19.csv"),
        str(BASE / "cage_inference/pairs_minimal_shard_*_epoch_19.partial.csv"),
        str(BASE / "relevant_shards/out_*/pairs_relevant_shard_*_epoch_19.csv"),
        str(BASE / "relevant_shards/out_*/pairs_relevant_shard_*_epoch_19.partial.csv"),
        str(BASE / "relevant16/out_*/pairs_relevant16_*_epoch_19.csv"),
        str(BASE / "relevant16/out_*/pairs_relevant16_*_epoch_19.partial.csv"),
        str(BASE / "required16/out_*/pairs_required16_*_epoch_19.csv"),
        str(BASE / "required16/out_*/pairs_required16_*_epoch_19.partial.csv"),
        str(BASE / "required_rescue/out/pairs_required_rescue_epoch_19.csv"),
        str(BASE / "required_rescue/out/pairs_required_rescue_epoch_19.partial.csv"),
    ]
    for pattern in patterns:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument(
        "--accept-runtime-missing",
        action="store_true",
        help=(
            "Write the covered scorer pairs even when a small set of pairs "
            "failed at runtime. Those pairs remain unavailable and are "
            "retained as misses by downstream evaluation."
        ),
    )
    args = ap.parse_args()

    need = required_pairs()
    need_index = pd.MultiIndex.from_frame(
        need[["reaction_id", "UniprotID"]]
    )
    pieces = []
    sources = []
    for path in source_paths():
        frame = pd.read_csv(
            path,
            dtype=str,
            usecols=["reaction_id", "UniprotID", "pred_logit"],
        ).fillna("")
        frame["pred_logit"] = pd.to_numeric(
            frame.pred_logit,
            errors="raise",
        )
        index = pd.MultiIndex.from_frame(
            frame[["reaction_id", "UniprotID"]]
        )
        frame = frame.loc[index.isin(need_index)].copy()
        if len(frame):
            pieces.append(frame)
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "rows_total": int(len(index)),
                "rows_required": int(len(frame)),
            }
        )

    if pieces:
        scores = pd.concat(pieces, ignore_index=True).drop_duplicates(
            ["reaction_id", "UniprotID"],
            keep="last",
        )
    else:
        scores = pd.DataFrame(
            columns=["reaction_id", "UniprotID", "pred_logit"]
        )

    coverage = need.merge(
        scores[["reaction_id", "UniprotID"]],
        on=["reaction_id", "UniprotID"],
        how="left",
        indicator=True,
    )
    missing = coverage[coverage._merge.eq("left_only")][
        ["reaction_id", "UniprotID"]
    ]
    missing_path = OUT.with_name("required_scores_missing.csv")
    missing.to_csv(missing_path, index=False)
    audit = {
        "schema": "bridge-layered-v4-r2e-cage-required-scores",
        "required_pairs": int(len(need)),
        "covered_pairs": int(len(need) - len(missing)),
        "missing_pairs": int(len(missing)),
        "route_queries_with_gate_positive": need.attrs["route_queries"],
        "sources": sources,
        "complete": len(missing) == 0,
        "runtime_unscored_pairs": int(len(missing)),
        "runtime_unscored_policy": (
            "unavailable scorer pair retained as miss; no replacement score"
            if len(missing)
            else "none"
        ),
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n")

    if len(missing) == 0:
        scores = need.merge(
            scores,
            on=["reaction_id", "UniprotID"],
            how="left",
            validate="one_to_one",
        )
        scores.to_csv(OUT, index=False)
        print(json.dumps(audit, indent=2))
        print(f"wrote {OUT}")
        return

    if args.accept_runtime_missing:
        scores.to_csv(OUT, index=False)
        print(json.dumps(audit, indent=2))
        print(f"wrote covered scores with runtime-missing audit to {OUT}")
        return

    partial = OUT.with_name("required_scores.partial.csv.gz")
    scores.to_csv(partial, index=False)
    print(json.dumps(audit, indent=2))
    print(f"wrote incomplete {partial}")
    if not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
