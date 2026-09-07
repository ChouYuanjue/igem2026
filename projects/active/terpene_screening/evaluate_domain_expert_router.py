from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.bootstrap_general_known_recovery import paired_bootstrap_delta
from projects.active.terpene_screening.compare_general_known_recovery import KEY_COLUMNS, METRIC_COLUMNS, _validate_unique

DEFAULT_BASE_DIR = ROOT / "results/terpene_production_models/marts_adapted_drfp_pu"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean/entries.csv"
DEFAULT_REGISTERED_PROTEINS = ROOT / "data/terpene_open_world_registry/proteins/entries.csv"


def _id_column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str:
    for column in choices:
        if column in frame.columns:
            return column
    raise ValueError(f"none of identifier columns {choices} found in {list(frame.columns)}")


def load_specialist_query_ids(root: Path = ROOT, base_dir: Path = DEFAULT_BASE_DIR) -> tuple[set[str], set[str]]:
    protein_ids: set[str] = set()
    for path in [
        root / "data/terpene_embeddings/esmc600m_mean/entries.csv",
        root / "data/terpene_open_world_registry/proteins/entries.csv",
    ]:
        frame = pd.read_csv(path, dtype=str).fillna("")
        column = _id_column(frame, ("Entry", "protein_id"))
        protein_ids.update(value for value in frame[column].astype(str) if value)

    reaction_frame = pd.read_csv(base_dir / "reaction_registry.csv", dtype=str).fillna("")
    reaction_column = _id_column(reaction_frame, ("reaction_id", "rhea_id"))
    reaction_ids = {value for value in reaction_frame[reaction_column].astype(str) if value}
    return protein_ids, reaction_ids


def specialist_mask(frame: pd.DataFrame, *, protein_ids: set[str], reaction_ids: set[str]) -> pd.Series:
    return (
        (frame["direction"].eq("reaction_to_enzyme") & frame["query_id"].isin(reaction_ids))
        | (frame["direction"].eq("enzyme_to_reaction") & frame["query_id"].isin(protein_ids))
    )


def route_query_metrics(
    legacy: pd.DataFrame,
    general: pd.DataFrame,
    *,
    protein_ids: set[str],
    reaction_ids: set[str],
) -> pd.DataFrame:
    _validate_unique(legacy, "legacy")
    _validate_unique(general, "general")
    legacy_keys = set(map(tuple, legacy[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    general_keys = set(map(tuple, general[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    if legacy_keys != general_keys:
        raise ValueError("legacy and general query populations do not match exactly")
    joined = legacy.merge(
        general,
        on=list(KEY_COLUMNS),
        suffixes=("_legacy", "_general"),
        validate="one_to_one",
    )
    mask = specialist_mask(joined, protein_ids=protein_ids, reaction_ids=reaction_ids)
    routed = joined[list(KEY_COLUMNS)].copy()
    routed["expert"] = mask.map({True: "tps_legacy", False: "general"})
    routed["n_positives"] = joined["n_positives_legacy"].astype(int)
    if not (joined["n_positives_legacy"].astype(int) == joined["n_positives_general"].astype(int)).all():
        raise ValueError("legacy and general positive counts differ")
    for metric in METRIC_COLUMNS:
        routed[f"legacy_{metric}"] = joined[f"{metric}_legacy"].astype(float)
        routed[f"general_{metric}"] = joined[f"{metric}_general"].astype(float)
        routed[metric] = routed[f"general_{metric}"].where(~mask, routed[f"legacy_{metric}"])
    return routed


def aggregate_routed(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (direction, stratum), group in frame.groupby(["direction", "stratum"], sort=True):
        row: dict[str, object] = {
            "direction": direction,
            "stratum": stratum,
            "n_queries": int(len(group)),
            "tps_legacy_fraction": float(group["expert"].eq("tps_legacy").mean()),
        }
        for metric in METRIC_COLUMNS:
            legacy_mean = float(group[f"legacy_{metric}"].mean())
            general_mean = float(group[f"general_{metric}"].mean())
            routed_mean = float(group[metric].mean())
            row[f"legacy_{metric}"] = legacy_mean
            row[f"general_{metric}"] = general_mean
            row[f"routed_{metric}"] = routed_mean
            row[f"delta_{metric}"] = routed_mean - legacy_mean
            row[f"ratio_{metric}"] = routed_mean / legacy_mean if legacy_mean != 0 else None
        rows.append(row)

    all_known = frame[frame["stratum"].eq("all_known")]
    if not all_known.empty:
        row = {
            "direction": "both_micro",
            "stratum": "all_known",
            "n_queries": int(len(all_known)),
            "tps_legacy_fraction": float(all_known["expert"].eq("tps_legacy").mean()),
        }
        for metric in METRIC_COLUMNS:
            legacy_mean = float(all_known[f"legacy_{metric}"].mean())
            general_mean = float(all_known[f"general_{metric}"].mean())
            routed_mean = float(all_known[metric].mean())
            row[f"legacy_{metric}"] = legacy_mean
            row[f"general_{metric}"] = general_mean
            row[f"routed_{metric}"] = routed_mean
            row[f"delta_{metric}"] = routed_mean - legacy_mean
            row[f"ratio_{metric}"] = routed_mean / legacy_mean if legacy_mean != 0 else None
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_routed(
    frame: pd.DataFrame,
    *,
    samples: int = 30000,
    seed: int = 20260723,
    strata: tuple[str, ...] = ("all_known", "unseen_to_historical_training"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    index = 0
    for direction in ("reaction_to_enzyme", "enzyme_to_reaction"):
        for stratum in strata:
            group = frame[(frame["direction"] == direction) & (frame["stratum"] == stratum)]
            if group.empty:
                continue
            for metric in ("reciprocal_rank", "hit_at_10", "hit_at_20"):
                result = paired_bootstrap_delta(
                    group[f"legacy_{metric}"].to_numpy(),
                    group[metric].to_numpy(),
                    samples=samples,
                    seed=seed + index,
                )
                rows.append({"direction": direction, "stratum": stratum, "metric": metric, **result})
                index += 1
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a post-hoc TPS-specialist/generalist expert router on matched retrieval queries.")
    parser.add_argument("--legacy-eval-dir", type=Path, required=True)
    parser.add_argument("--general-eval-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=30000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()

    legacy = pd.read_csv(args.legacy_eval_dir / "query_metrics.csv")
    general = pd.read_csv(args.general_eval_dir / "query_metrics.csv")
    protein_ids, reaction_ids = load_specialist_query_ids(ROOT, args.base_dir)
    routed = route_query_metrics(
        legacy, general, protein_ids=protein_ids, reaction_ids=reaction_ids
    )
    metrics = aggregate_routed(routed)
    bootstrap = bootstrap_routed(
        routed, samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    routed.to_csv(args.output_dir / "routed_query_metrics.csv", index=False)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap.csv", index=False)
    summary = {
        "method": "post_hoc_domain_expert_routing",
        "router": "TPS-specialist query membership -> legacy production; otherwise general expert",
        "n_specialist_proteins": len(protein_ids),
        "n_specialist_reactions": len(reaction_ids),
        "legacy_eval_dir": str(args.legacy_eval_dir.resolve()),
        "general_eval_dir": str(args.general_eval_dir.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "outputs": {
            "query_metrics": str((args.output_dir / "routed_query_metrics.csv").resolve()),
            "metrics": str((args.output_dir / "metrics.csv").resolve()),
            "paired_bootstrap": str((args.output_dir / "paired_bootstrap.csv").resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
