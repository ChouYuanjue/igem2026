from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.core.conformal import (  # noqa: E402
    CONFORMAL_METHOD,
    CONFORMAL_RETRIEVAL_VERSION,
    applicability_group,
    conformal_set_size,
    finite_sample_quantile,
    normalized_rank_score,
)
from projects.active.terpene_screening.core.evidence import (  # noqa: E402
    APPLICABILITY_MODEL_VERSION,
    compute_query_applicability,
)
from projects.active.terpene_screening.core.provenance import identifier_set_hash  # noqa: E402
from projects.active.terpene_screening.core.registry_snapshots import (  # noqa: E402
    resolve_protein_dir,
    resolve_reaction_path,
)
from projects.active.terpene_screening.core.routing import (  # noqa: E402
    DEFAULT_ROUTE_MANIFEST,
    load_route_manifest,
)


DEFAULT_FEATURES = (
    ROOT
    / "results/terpene_open_world_uncertainty_rrf_routing/e2r_query_uncertainty_features.csv"
)
DEFAULT_OUTPUT = ROOT / "results/terpene_conformal_retrieval_sets"
DEFAULT_CURRENT_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean/entries.csv"
DEFAULT_REGISTERED_PROTEINS = ROOT / "data/terpene_open_world_registry/proteins"
DEFAULT_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_REGISTERED_REACTIONS = ROOT / "data/terpene_open_world_registry/reactions.csv"
DEFAULT_BENCHMARK_PROTEINS = ROOT / "data/terpene_marts_adaptation/protein_entities.csv"
DEFAULT_BENCHMARK_REACTIONS = ROOT / "data/terpene_marts_adaptation/reaction_entities.csv"
DEFAULT_ALPHAS = (0.20, 0.10, 0.05)
MIN_GROUP_CALIBRATION = 20
MIN_GROUP_TEST = 20


def _hash_partition(query_id: str) -> str:
    digest = hashlib.sha256(
        f"{CONFORMAL_RETRIEVAL_VERSION}:{query_id}".encode()
    ).hexdigest()
    return "calibration" if int(digest[:8], 16) % 2 == 0 else "test"


def _read_ids(path: Path, candidates: tuple[str, ...]) -> list[str]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    for column in candidates:
        if column in frame.columns:
            return sorted({str(value) for value in frame[column] if str(value)})
    raise ValueError(f"No identifier column {candidates} in {path}")


def production_candidate_universes(
    current_proteins: Path,
    registered_proteins: Path,
    positives: Path,
    registered_reactions: Path,
) -> dict[str, dict[str, Any]]:
    current_protein_ids = _read_ids(current_proteins, ("Entry", "protein_id", "enzyme_id"))
    registered_dir = resolve_protein_dir(registered_proteins)
    registered_protein_ids = _read_ids(
        registered_dir / "entries.csv", ("Entry", "protein_id", "enzyme_id")
    )
    positive_frame = pd.read_csv(positives, sep="\t", dtype=str).fillna("")
    current_reaction_ids = sorted({str(value) for value in positive_frame["rhea_id"] if str(value)})
    registered_reaction_path = resolve_reaction_path(registered_reactions)
    registered_reaction_ids = _read_ids(
        registered_reaction_path, ("reaction_id", "rhea_id")
    )
    protein_ids = sorted(set(current_protein_ids) | set(registered_protein_ids))
    reaction_ids = sorted(set(current_reaction_ids) | set(registered_reaction_ids))
    return {
        "reaction_to_enzyme": {
            "ids": protein_ids,
            "count": len(protein_ids),
            "hash": identifier_set_hash(protein_ids),
        },
        "enzyme_to_reaction": {
            "ids": reaction_ids,
            "count": len(reaction_ids),
            "hash": identifier_set_hash(reaction_ids),
        },
    }


def _query_applicability(record: pd.Series) -> float:
    values = record.to_dict()
    values["query_nearest_library_similarity"] = values.get(
        "query_nearest_train_similarity", np.nan
    )
    return float(compute_query_applicability(values)["score"])


def collapse_query_units(
    frame: pd.DataFrame,
    benchmark_counts: dict[str, int],
) -> pd.DataFrame:
    required = {
        "split_id",
        "direction",
        "budget",
        "query_id",
        "best_positive_rank",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Calibration features missing columns: {sorted(missing)}")
    work = frame.copy()
    work["query_applicability_score"] = work.apply(_query_applicability, axis=1)
    rows: list[dict[str, Any]] = []
    for (direction, budget, query_id), group in work.groupby(
        ["direction", "budget", "query_id"], sort=True
    ):
        candidate_count = int(benchmark_counts[str(direction)])
        best_rank = int(pd.to_numeric(group["best_positive_rank"]).max())
        rows.append(
            {
                "direction": str(direction),
                "budget": int(budget),
                "query_id": str(query_id),
                "n_occurrences": int(len(group)),
                "best_positive_rank": best_rank,
                "benchmark_candidate_count": candidate_count,
                "nonconformity_score": normalized_rank_score(best_rank, candidate_count),
                "query_applicability_score": float(
                    pd.to_numeric(group["query_applicability_score"]).min()
                ),
                "applicability_group": applicability_group(
                    compute_query_applicability(
                        {
                            "query_nearest_library_similarity": float(
                                pd.to_numeric(
                                    group["query_nearest_train_similarity"]
                                ).min()
                            ),
                            "ensemble_top1_vote_fraction": float(
                                pd.to_numeric(group["ensemble_top1_vote_fraction"]).min()
                            ),
                            "ensemble_top1_rank_std": float(
                                pd.to_numeric(group["ensemble_top1_rank_std"]).max()
                            ),
                            "ensemble_topk_jaccard": float(
                                pd.to_numeric(group["ensemble_topk_jaccard"]).min()
                            ),
                            "ensemble_topk_vote_mean": float(
                                pd.to_numeric(group["ensemble_topk_vote_mean"]).min()
                            ),
                            "ensemble_boundary_margin_z": float(
                                pd.to_numeric(group["ensemble_boundary_margin_z"]).min()
                            ),
                        }
                    )["tier"]
                ),
                "holdout_partition": _hash_partition(str(query_id)),
            }
        )
    return pd.DataFrame(rows)


def _validation_payload(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    alpha: float,
    candidate_count: int,
) -> dict[str, Any]:
    qhat = finite_sample_quantile(calibration["nonconformity_score"], alpha)
    set_size = conformal_set_size(qhat, candidate_count)
    coverage = float(test["best_positive_rank"].le(set_size).mean()) if len(test) else None
    observed_misses = int(test["best_positive_rank"].gt(set_size).sum()) if len(test) else None
    allowed_misses = int(binom.ppf(0.99, len(test), alpha)) if len(test) else None
    meets_nominal = bool(
        len(test) > 0
        and coverage is not None
        and coverage + 1e-12 >= float(1.0 - alpha)
    )
    return {
        "qhat": qhat,
        "set_size": set_size,
        "set_fraction": float(set_size / candidate_count),
        "n_calibration": int(len(calibration)),
        "n_test": int(len(test)),
        "empirical_coverage": coverage,
        "target_coverage": float(1.0 - alpha),
        "observed_misses": observed_misses,
        "allowed_misses_99pct": allowed_misses,
        "empirical_coverage_meets_nominal": meets_nominal,
        "coverage_passed": bool(
            observed_misses is not None
            and allowed_misses is not None
            and observed_misses <= allowed_misses
        ),
    }


def _production_payload(
    frame: pd.DataFrame,
    *,
    alpha: float,
    production_candidate_count: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    qhat = finite_sample_quantile(frame["nonconformity_score"], alpha)
    return {
        "qhat": qhat,
        "production_set_size": conformal_set_size(qhat, production_candidate_count),
        "production_set_fraction": float(
            conformal_set_size(qhat, production_candidate_count)
            / production_candidate_count
        ),
        "n_calibration": int(len(frame)),
        "validation": validation,
    }


def build_calibrators(
    units: pd.DataFrame,
    *,
    route_manifest: Path,
    production_universes: dict[str, dict[str, Any]],
    benchmark_counts: dict[str, int],
    alphas: tuple[float, ...],
) -> tuple[dict[str, Any], pd.DataFrame]:
    routes = load_route_manifest(str(route_manifest.resolve()))
    model_bundle_version = str(routes["model_bundle_version"])
    calibrators: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    for (direction, budget), frame in units.groupby(["direction", "budget"], sort=True):
        direction = str(direction)
        budget = int(budget)
        objective = f"top{budget}"
        key = f"{direction}_{objective}"
        route_id = str(routes["routes"][direction]["external"][objective]["route_id"])
        production_universe = production_universes[direction]
        alpha_payloads: dict[str, Any] = {}
        calibration = frame[frame["holdout_partition"].eq("calibration")]
        test = frame[frame["holdout_partition"].eq("test")]
        for alpha in alphas:
            global_validation = _validation_payload(
                calibration,
                test,
                alpha=alpha,
                candidate_count=benchmark_counts[direction],
            )
            global_payload = _production_payload(
                frame,
                alpha=alpha,
                production_candidate_count=int(production_universe["count"]),
                validation=global_validation,
            )
            groups: dict[str, Any] = {}
            for group_name in ("strong", "moderate", "weak"):
                group_frame = frame[frame["applicability_group"].eq(group_name)]
                group_calibration = group_frame[
                    group_frame["holdout_partition"].eq("calibration")
                ]
                group_test = group_frame[group_frame["holdout_partition"].eq("test")]
                if len(group_calibration):
                    validation = _validation_payload(
                        group_calibration,
                        group_test,
                        alpha=alpha,
                        candidate_count=benchmark_counts[direction],
                    )
                    production = _production_payload(
                        group_frame,
                        alpha=alpha,
                        production_candidate_count=int(production_universe["count"]),
                        validation=validation,
                    )
                else:
                    validation = {
                        "n_calibration": 0,
                        "n_test": int(len(group_test)),
                        "empirical_coverage": None,
                        "target_coverage": float(1.0 - alpha),
                        "observed_misses": None,
                        "allowed_misses_99pct": None,
                        "empirical_coverage_meets_nominal": False,
                        "coverage_passed": False,
                    }
                    production = {
                        "qhat": global_payload["qhat"],
                        "production_set_size": global_payload["production_set_size"],
                        "production_set_fraction": global_payload[
                            "production_set_fraction"
                        ],
                        "n_calibration": 0,
                        "validation": validation,
                    }
                enabled = bool(
                    len(group_calibration) >= MIN_GROUP_CALIBRATION
                    and len(group_test) >= MIN_GROUP_TEST
                    and validation.get("coverage_passed", False)
                    and validation.get("empirical_coverage_meets_nominal", False)
                )
                production["enabled"] = enabled
                production["fallback"] = "global" if not enabled else "none"
                production["selection_rule"] = (
                    "query_feature_only_mondrian_group"
                )
                groups[group_name] = production
                metrics.append(
                    {
                        "calibrator": key,
                        "alpha": alpha,
                        "scope": f"mondrian:{group_name}",
                        "enabled": enabled,
                        **validation,
                        "production_set_size": production["production_set_size"],
                        "production_set_fraction": production[
                            "production_set_fraction"
                        ],
                    }
                )
            alpha_payloads[f"{alpha:.2f}"] = {
                "global": global_payload,
                "groups": groups,
                "validation": global_validation,
            }
            metrics.append(
                {
                    "calibrator": key,
                    "alpha": alpha,
                    "scope": "global",
                    "enabled": True,
                    **global_validation,
                    "production_set_size": global_payload["production_set_size"],
                    "production_set_fraction": global_payload[
                        "production_set_fraction"
                    ],
                }
            )
        calibrators[key] = {
            "direction": direction,
            "ranking_objective": objective,
            "benchmark_candidate_count": int(benchmark_counts[direction]),
            "production_candidate_count": int(production_universe["count"]),
            "compatibility": {
                "route_id": route_id,
                "candidate_universe_hash": str(production_universe["hash"]),
                "model_bundle_version": model_bundle_version,
                "binding_scope": "external_zero_shot_default_candidate_universe",
            },
            "alphas": alpha_payloads,
            "guarantee_scope": (
                "marginal_at_least_one_known_positive_under_exchangeability_"
                "in_locked_query_disjoint_double_cold_protocol"
            ),
            "production_transport_note": (
                "normalized-rank thresholds are transported to the bound production universe; "
                "runtime output is a retrieval coverage diagnostic, not an activity guarantee"
            ),
        }
    manifest = {
        "manifest_version": 1,
        "conformal_retrieval_version": CONFORMAL_RETRIEVAL_VERSION,
        "method": CONFORMAL_METHOD,
        "applicability_model_version": APPLICABILITY_MODEL_VERSION,
        "source_features": str(DEFAULT_FEATURES.resolve()),
        "query_unit": (
            "unique query id; repeated double-cold occurrences collapsed by worst best-positive rank"
        ),
        "holdout": (
            "deterministic query-disjoint SHA256 split; one half calibration and one half test"
        ),
        "group_policy": {
            "strong": "reference_library or in_domain",
            "moderate": "near_domain",
            "weak": "weakly_supported or far_out_of_domain",
            "minimum_group_calibration": MIN_GROUP_CALIBRATION,
            "minimum_group_test": MIN_GROUP_TEST,
            "enablement": "minimum sizes and empirical holdout coverage >= nominal target",
        },
        "calibrators": calibrators,
    }
    return manifest, pd.DataFrame(metrics)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build route-bound conformal retrieval-set calibrators for TPS retrieval."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--route-manifest", type=Path, default=DEFAULT_ROUTE_MANIFEST)
    parser.add_argument("--alphas", default=",".join(str(value) for value in DEFAULT_ALPHAS))
    args = parser.parse_args()
    alphas = tuple(sorted({float(value) for value in args.alphas.split(",") if value}, reverse=True))
    if not alphas or any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("alphas must contain values strictly between 0 and 1")

    features = pd.read_csv(args.features.resolve())
    benchmark_counts = {
        "reaction_to_enzyme": len(pd.read_csv(DEFAULT_BENCHMARK_PROTEINS)),
        "enzyme_to_reaction": len(pd.read_csv(DEFAULT_BENCHMARK_REACTIONS)),
    }
    production_universes = production_candidate_universes(
        DEFAULT_CURRENT_PROTEINS,
        DEFAULT_REGISTERED_PROTEINS,
        DEFAULT_POSITIVES,
        DEFAULT_REGISTERED_REACTIONS,
    )
    units = collapse_query_units(features, benchmark_counts)
    manifest, metrics = build_calibrators(
        units,
        route_manifest=args.route_manifest.resolve(),
        production_universes=production_universes,
        benchmark_counts=benchmark_counts,
        alphas=alphas,
    )
    manifest["source_features"] = str(args.features.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(output_dir / "query_calibration_units.csv", index=False)
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False)
    (output_dir / "calibrators.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    global_metrics = metrics[metrics["scope"].eq("global")]
    summary = {
        "status": "passed" if bool(global_metrics["coverage_passed"].all()) else "failed",
        "conformal_retrieval_version": CONFORMAL_RETRIEVAL_VERSION,
        "method": CONFORMAL_METHOD,
        "n_query_units": int(len(units)),
        "n_calibrators": int(len(manifest["calibrators"])),
        "alphas": list(alphas),
        "global_holdout_coverage_all_passed": bool(
            global_metrics["coverage_passed"].all()
        ),
        "enabled_mondrian_groups": int(
            metrics[metrics["scope"].ne("global")]["enabled"].sum()
        ),
        "production_candidate_counts": {
            key: int(value["count"]) for key, value in production_universes.items()
        },
        "production_candidate_hashes": {
            key: str(value["hash"]) for key, value in production_universes.items()
        },
        "outputs": {
            "calibrators": str(output_dir / "calibrators.json"),
            "query_units": str(output_dir / "query_calibration_units.csv"),
            "validation_metrics": str(output_dir / "validation_metrics.csv"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit("Global conformal holdout coverage failed for at least one route/alpha")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
