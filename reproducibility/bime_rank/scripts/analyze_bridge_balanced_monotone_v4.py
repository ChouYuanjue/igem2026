from __future__ import annotations

"""Two retained BRIDGE tables with monotone degree-balanced Hit@K.

The original 23,773 targets, candidate universe, Broad/CAGE rank files, and v3
score assets are never rewritten. No K-dependent chance correction is applied
to the metrics named Hit@K.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts import analyze_bridge_difficulty_standardized_v3 as v3

OUT = ROOT / "results/bridge_difficulty_balanced_query_gate_v4"
E2R_QUERY_GATE = ROOT / "results/bridge_e2r_query_gate_v3/edge_metrics.csv.gz"
E2R_ROUTE_V4 = ROOT / "results/bridge_e2r_route_gate_v4/edge_metrics.csv.gz"
SPLIT = ROOT / "results/bridge_gate_split_v2"
NOVELTIES = v3.NOVELTY_ORDER
METRICS = v3.METRICS


def metric(ranks: np.ndarray) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.int64)
    if not len(r) or np.any(r < 1):
        raise ValueError("Expected nonempty rank-positive evaluation cell")
    return {
        "mrr": float(np.mean(1.0 / r)),
        "hit3": float(np.mean(r <= 3)),
        "hit10": float(np.mean(r <= 10)),
        "hit100": float(np.mean(r <= 100)),
    }


def balanced(frame: pd.DataFrame, rank_col: str) -> dict:
    direct = metric(frame[rank_col].to_numpy(np.int64))
    groups = {}
    for novelty in NOVELTIES:
        sample = frame.loc[frame.novelty.eq(novelty)]
        if sample.empty:
            raise RuntimeError(f"Unrepresented novelty category {novelty}")
        cells = [metric(cell[rank_col].to_numpy(np.int64))
                 for _, cell in sample.groupby("difficulty_stratum", sort=True)]
        groups[novelty] = {
            key: float(np.mean([values[key] for values in cells])) for key in METRICS
        }
    standard = {
        key: float(np.mean([groups[name][key] for name in NOVELTIES]))
        for key in METRICS
    }
    for name, m in (("direct", direct), ("balanced", standard)):
        if not 0 <= m["hit3"] <= m["hit10"] <= m["hit100"] <= 1:
            raise AssertionError(f"Hit@K monotonicity broken in {rank_col} {name}: {m}")
    return {"direct": direct, "balanced": standard, "novelty": groups}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-new-e2r", action="store_true",
                    help="Read frozen E2R gate outputs only after gate validation and full test rank QA")
    ap.add_argument("--e2r-route-v4", action="store_true",
                    help="Substitute only the joint query route E2R ranks on the same frozen 21,505 test edges")
    ap.add_argument("--split-v2", action="store_true",
                    help="Reuse all original rank assets but select only frozen query-heldout evaluation pairs")
    args = ap.parse_args()
    if args.with_new_e2r and args.e2r_route_v4:
        raise ValueError("Choose only one already-frozen E2R query gate for scoring")
    use_gate = bool(args.with_new_e2r or args.e2r_route_v4)
    gate_path = E2R_ROUTE_V4 if args.e2r_route_v4 else E2R_QUERY_GATE
    split_v2 = bool(args.split_v2 or use_gate)
    subset = None
    if split_v2:
        manifest = json.loads((SPLIT / "manifest.json").read_text())
        subset = pd.read_csv(SPLIT / "evaluation_pairs.csv.gz",
                             dtype={"protein_id": str, "reaction_id": str})
        if len(subset) != manifest["parts"]["evaluation"]["edges"]:
            raise AssertionError("Gate-independent test denominator does not match frozen split")
        if subset.duplicated(v3.KEYS).any():
            raise AssertionError("Frozen evaluation set contains duplicate relations")
    train = pd.read_csv(v3.TRAIN, dtype=str).drop_duplicates(v3.KEYS)
    pdeg = train.groupby("protein_id").size().to_dict()
    rdeg = train.groupby("reaction_id").size().to_dict()
    table = {}
    if use_gate and not gate_path.exists():
        raise FileNotFoundError(gate_path)
    for direction in ("r2e", "e2r"):
        frame, sources, missing = v3.load_direction(direction)
        if subset is not None:
            frame = frame.merge(subset, on=v3.KEYS, how="inner", validate="one_to_one",
                                sort=False)
            if len(frame) != len(subset):
                raise AssertionError(f"{direction}: original ranking cache does not preserve all fixed evaluation pairs")
        frame = v3.annotate(frame, pdeg, rdeg)
        if use_gate and direction == "e2r":
            updated = pd.read_csv(
                gate_path,
                dtype={"protein_id": str, "reaction_id": str},
            )
            cols = ("full_rank", "minus_functional_rank",
                    "minus_structure_mechanism_rank", "minus_relational_memory_rank")
            if len(updated) != len(frame) or updated.duplicated(v3.KEYS).any():
                raise ValueError("New E2R gate must preserve all unique benchmark targets")
            changed = frame[v3.KEYS].merge(
                updated[v3.KEYS + list(cols) + ["broad_rank"]],
                on=v3.KEYS, validate="one_to_one", how="left", sort=False,
            )
            if changed[list(cols)].isna().any().any():
                raise AssertionError("New E2R gate does not cover all frozen evaluation relations")
            if not np.array_equal(changed.broad_rank.to_numpy(), frame.broad_rank.to_numpy()):
                raise AssertionError("Frozen Broad ranks were modified")
            for col in cols:
                frame[col] = changed[col].to_numpy(np.int64)
            frame["minus_family_domain_rank"] = frame["full_rank"]
            # Hybrid CAGE-Gate + BRIDGE is coupled to the earlier BRIDGE
            # scoring route and must not be silently represented as the
            # re-gated BRIDGE. Its old asset remains intact for later rerun.
            sources.pop("EnzymeCAGE Gate + BRIDGE Reranking", None)
        methods = {}
        for name in v3.METHOD_ORDER:
            if name not in sources:
                continue
            rank_col, _ = sources[name]
            methods[name] = balanced(frame, rank_col)
        table[direction] = {
            "edges": int(len(frame)),
            "queries": int(frame["reaction_id" if direction == "r2e" else "protein_id"].nunique()),
            "methods": methods,
            "unavailable": missing + (
                ["EnzymeCAGE Gate + BRIDGE Reranking (requires new gate hybrid rescoring)"]
                if use_gate and direction == "e2r" else []
            ),
        }
    output = {
        "schema": "bridge-direct-and-balanced-monotone-v4",
        "benchmark": (
            "same frozen parent 23,773 relation-unseen edges; 90% protein-query-heldout evaluation subset"
            if split_v2 else "same frozen 23,773 Rhea relation-unseen edges"
        ),
        "fixed_parent_relations": 23773,
        "heldout_evaluation_relations": int(len(subset)) if subset is not None else 23773,
        "gate_dev_queries_strictly_excluded": bool(split_v2),
        "frozen_parent_rank_assets_reused_by_row_join": bool(split_v2),
        "r2e_old_gate_is_archival_not_independently_refit": bool(split_v2),

        "balanced_formula": "equal four novelty classes; within each class equal frozen degree strata; within each stratum raw mean of 1[filtered_rank <= K]",
        "monotone_HitK": True,
        "chance_adjustment_in_headline": False,
        "uses_new_e2r_query_gate": bool(use_gate),
        "e2r_joint_route_v4": bool(args.e2r_route_v4),
        "legacy_metrics_and_rank_assets_left_untouched": True,
        "directions": table,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = ("split_v2_route_v4" if args.e2r_route_v4
              else "split_v2_query_gate" if args.with_new_e2r
              else "split_v2_cached" if split_v2 else "frozen")
    (OUT / f"summary_{suffix}.json").write_text(json.dumps(output, indent=2) + "\n")
    for direction, datum in table.items():
        for mode in ("direct", "balanced"):
            rows = [
                {"Method": name,
                 "MRR": f'{v[mode]["mrr"]:.5f}',
                 "Hit@3": f'{v[mode]["hit3"]*100:.2f}%',
                 "Hit@10": f'{v[mode]["hit10"]*100:.2f}%',
                 "Hit@100": f'{v[mode]["hit100"]*100:.2f}%'}
                for name, v in datum["methods"].items()
            ]
            pd.DataFrame(rows).to_csv(OUT / f"{direction}_{mode}_{suffix}.csv", index=False)
            print(f"=== {direction} {mode} {suffix} ===", flush=True)
            print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("SOURCE_RANK_FILES_UNMODIFIED", flush=True)


if __name__ == "__main__":
    main()
