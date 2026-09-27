from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

SCORECARD = ROOT / "reproducibility/bime_rank/records/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json"
R2E_INTERNAL = ROOT / "configs/retrieval/enzyme_ranking_validation.json"
E2R_INTERNAL = ROOT / "configs/retrieval/reaction_model_validation.json"
ORPHAN = ROOT / "results/orphan335_fixed_pool_v1/summary.json"
ATLAS_TPS = ROOT / "results/fibre_interaction_atlas_full_v1"
ATLAS_TPS_BROAD = ROOT / "results/fibre_interaction_atlas_tps_broad_universe_v1"
HISTORICAL_TPS_MULTI = ROOT / "results/terpene_multi_expert_marts_rankings_v1/metrics.csv"
OUTPUT = ROOT / "reproducibility/bime_rank/records/FIBRE_INTERACTION_ATLAS_SCORECARD_V1.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics_row(
    path: Path,
    protocol: str,
    direction: str | None = None,
    method: str | None = None,
) -> dict:
    table = pd.read_csv(path)
    row = table[table["protocol"] == protocol]
    if direction is not None:
        row = row[row["direction"] == direction]
    if method is not None:
        row = row[row["method"] == method]
    if len(row) != 1:
        raise ValueError(f"expected one {protocol} row in {path}, found {len(row)}")
    return row.iloc[0].to_dict()


def clean_number(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def clean_dict(values: dict) -> dict:
    return {key: clean_number(value) for key, value in values.items()}


def main() -> None:
    for path in (
        SCORECARD,
        R2E_INTERNAL,
        E2R_INTERNAL,
        ORPHAN,
        ATLAS_TPS / "summary.json",
        ATLAS_TPS / "metrics.csv",
        ATLAS_TPS_BROAD / "summary.json",
        ATLAS_TPS_BROAD / "metrics.csv",
        HISTORICAL_TPS_MULTI,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    scorecard = read_json(SCORECARD)
    r2e_internal = read_json(R2E_INTERNAL)
    e2r_internal = read_json(E2R_INTERNAL)
    orphan = read_json(ORPHAN)
    atlas_tps_summary = read_json(ATLAS_TPS / "summary.json")
    atlas_broad_summary = read_json(ATLAS_TPS_BROAD / "summary.json")
    historical_tps = pd.read_csv(HISTORICAL_TPS_MULTI)

    zero = scorecard["zero_shot_external"]
    conditional = scorecard["conditional_known_positive"]

    atlas_double = clean_dict(
        metrics_row(
            ATLAS_TPS / "metrics.csv",
            "double_cold_25cell",
            "reaction_to_enzyme",
        )
    )
    atlas_double_e2r = clean_dict(
        metrics_row(
            ATLAS_TPS / "metrics.csv",
            "double_cold_25cell",
            "enzyme_to_reaction",
        )
    )
    atlas_exact = clean_dict(
        metrics_row(
            ATLAS_TPS / "metrics.csv",
            "legacy_exact",
            "reaction_to_enzyme",
        )
    )
    atlas_broad = clean_dict(
        metrics_row(
            ATLAS_TPS_BROAD / "metrics.csv",
            "tps_strict_double_cold_general_merged",
            method="full_atlas",
        )
    )
    atlas_broad_universal = clean_dict(
        metrics_row(
            ATLAS_TPS_BROAD / "metrics.csv",
            "tps_strict_double_cold_general_merged",
            method="universal_chart_only",
        )
    )

    old_tps = historical_tps[
        (historical_tps["method"] == "multi_expert_marts_only")
        & (historical_tps["direction"] == "reaction_to_enzyme")
    ]
    if len(old_tps) != 1:
        raise ValueError("historical TPS multi-expert R2E row is not unique")
    old_tps = old_tps.iloc[0].to_dict()

    current_tps = historical_tps[
        (historical_tps["method"] == "current_pretrained_multi_expert")
        & (historical_tps["direction"] == "reaction_to_enzyme")
    ]
    if len(current_tps) != 1:
        raise ValueError("historical current-pretrained TPS R2E row is not unique")
    current_tps = current_tps.iloc[0].to_dict()

    old_tps_e2r = historical_tps[
        (historical_tps["method"] == "multi_expert_marts_only")
        & (historical_tps["direction"] == "enzyme_to_reaction")
    ]
    current_tps_e2r = historical_tps[
        (historical_tps["method"] == "current_pretrained_multi_expert")
        & (historical_tps["direction"] == "enzyme_to_reaction")
    ]
    if len(old_tps_e2r) != 1 or len(current_tps_e2r) != 1:
        raise ValueError("historical TPS multi-expert E2R rows are not unique")
    old_tps_e2r = old_tps_e2r.iloc[0].to_dict()
    current_tps_e2r = current_tps_e2r.iloc[0].to_dict()

    atlas_vs_old = {
        "mrr_delta": (
            atlas_double["mean_reciprocal_rank"]
            - float(old_tps["mean_reciprocal_rank"])
        ),
        "hit_at_10_delta": (
            atlas_double["hit_probability_at_10"]
            - float(old_tps["hit_probability_at_10"])
        ),
        "hit_at_20_delta": (
            atlas_double["hit_probability_at_20"]
            - float(old_tps["hit_probability_at_20"])
        ),
    }
    atlas_vs_unadapted = {
        "mrr_delta": (
            atlas_double["mean_reciprocal_rank"]
            - float(current_tps["mean_reciprocal_rank"])
        ),
        "hit_at_10_delta": (
            atlas_double["hit_probability_at_10"]
            - float(current_tps["hit_probability_at_10"])
        ),
        "hit_at_20_delta": (
            atlas_double["hit_probability_at_20"]
            - float(current_tps["hit_probability_at_20"])
        ),
    }
    atlas_e2r_vs_old = {
        "mrr_delta": (
            atlas_double_e2r["mean_reciprocal_rank"]
            - float(old_tps_e2r["mean_reciprocal_rank"])
        ),
        "hit_at_10_delta": (
            atlas_double_e2r["hit_probability_at_10"]
            - float(old_tps_e2r["hit_probability_at_10"])
        ),
        "hit_at_20_delta": (
            atlas_double_e2r["hit_probability_at_20"]
            - float(old_tps_e2r["hit_probability_at_20"])
        ),
    }
    atlas_e2r_vs_unadapted = {
        "mrr_delta": (
            atlas_double_e2r["mean_reciprocal_rank"]
            - float(current_tps_e2r["mean_reciprocal_rank"])
        ),
        "hit_at_10_delta": (
            atlas_double_e2r["hit_probability_at_10"]
            - float(current_tps_e2r["hit_probability_at_10"])
        ),
        "hit_at_20_delta": (
            atlas_double_e2r["hit_probability_at_20"]
            - float(current_tps_e2r["hit_probability_at_20"])
        ),
    }

    output = {
        "schema": "fibre-interaction-atlas-scorecard-v1",
        "method": "FIBRE — Factorized Interaction Basis for Reaction–Enzyme",
        "architecture": {
            "object": "reaction-enzyme catalytic interaction atlas",
            "global_formula": (
                "K(r,e)=sum_alpha rho_alpha(r,e) "
                "phi_alpha(r)^T G_alpha psi_alpha(e)"
            ),
            "universal_chart": True,
            "local_charts": (
                "multi-view biochemical/information regimes; TPS is a family chart"
            ),
            "partition_of_unity": True,
            "overlap_gluing": True,
            "valid_input_abstention": False,
            "production_agent_changed": False,
        },
        "evidence_policy": {
            "principle": (
                "Judge the architecture as a whole across complementary protocols. "
                "Do not collapse unlike candidate universes into one scalar."
            ),
            "retained_frozen_numbers": (
                "Existing production/general BiME-Rank routes are unchanged by the "
                "atlas reformulation and remain the numeric authority for those scopes."
            ),
            "atlas_native_numbers": (
                "New atlas-native training evidence comes only from the fixed "
                "fibre-reproduction TPS assets."
            ),
            "application_data_used_for_benchmark_claims": False,
        },
        "replay_commands": {
            "atlas_tps": (
                ".venv/bin/python "
                "reproducibility/bime_rank/support/"
                "evaluate_multi_expert_protocol_comparison.py "
                "--output-dir results/fibre_interaction_atlas_full_v1 "
                "--protocols legacy_exact,double_cold_25cell "
                "--reaction-feature-mode multiview --n-experts 8 "
                "--expert-dim 32 --reaction-loss-weight 0.5 "
                "--glue-weight 0.02 --epochs 80 --seeds 20260723 "
                "--device cuda"
            ),
            "atlas_tps_broad_universe": (
                ".venv/bin/python "
                "reproducibility/bime_rank/scripts/"
                "evaluate_fibre_atlas_tps_broad_universe_v1.py "
                "--output-dir "
                "results/fibre_interaction_atlas_tps_broad_universe_v1 "
                "--strict-partition all --device cuda"
            ),
            "assemble_scorecard": (
                ".venv/bin/python "
                "reproducibility/bime_rank/scripts/"
                "assemble_fibre_atlas_scorecard_v1.py"
            ),
        },
        "candidate_universe": scorecard["candidate_universes"],
        "retained_general_zero_shot": {
            "clipzyme_strict_temporal_same_support": zero[
                "clipzyme_strict_temporal_same_support"
            ],
            "enzyme405": zero["enzyme405"],
        },
        "retained_internal_frozen_confirmation": {
            "r2e": {
                "queries": r2e_internal["query_count"],
                "candidates": r2e_internal["candidate_count"],
                "baseline": r2e_internal["baseline"],
                "candidate": r2e_internal["candidate"],
                "delta": r2e_internal["delta"],
            },
            "e2r": {
                "queries": e2r_internal["query_count"],
                "candidates": e2r_internal["candidate_count"],
                "baseline": e2r_internal["baseline"],
                "candidate": e2r_internal["candidate"],
                "delta": e2r_internal["delta"],
            },
        },
        "atlas_native_tps": {
            "recipe": {
                "reaction_feature_mode": atlas_tps_summary["reaction_feature_mode"],
                "n_experts": atlas_tps_summary["config"]["n_experts"],
                "expert_dim": atlas_tps_summary["config"]["expert_dim"],
                "global_dim": atlas_tps_summary["config"]["global_dim"],
                "glue_weight": atlas_tps_summary["glue_weight"],
                "epochs": atlas_tps_summary["epochs"],
                "seed": atlas_tps_summary["seeds"],
            },
            "legacy_exact_full_tps_universe": atlas_exact,
            "strict_double_cold_25cell": {
                "reaction_to_enzyme": atlas_double,
                "enzyme_to_reaction": atlas_double_e2r,
            },
            "comparison_to_historical_tps_multi_expert": {
                "reaction_to_enzyme": {
                    "historical_unadapted": clean_dict(current_tps),
                    "historical_marts_only": clean_dict(old_tps),
                    "atlas_minus_unadapted": atlas_vs_unadapted,
                    "atlas_minus_marts_only": atlas_vs_old,
                },
                "enzyme_to_reaction": {
                    "historical_unadapted": clean_dict(current_tps_e2r),
                    "historical_marts_only": clean_dict(old_tps_e2r),
                    "atlas_minus_unadapted": atlas_e2r_vs_unadapted,
                    "atlas_minus_marts_only": atlas_e2r_vs_old,
                },
            },
            "broad_general_merged_r2e": {
                "protocol": atlas_broad_summary["protocol"],
                "candidate_count": atlas_broad_summary["candidate_count"],
                "strict_partition": atlas_broad_summary["strict_partition"],
                "full_atlas": atlas_broad,
                "universal_chart_only": atlas_broad_universal,
                "delta_full_minus_universal": {
                    "mrr": (
                        atlas_broad["mean_reciprocal_rank"]
                        - atlas_broad_universal["mean_reciprocal_rank"]
                    ),
                    "hit_at_10": (
                        atlas_broad["hit_probability_at_10"]
                        - atlas_broad_universal["hit_probability_at_10"]
                    ),
                    "hit_at_20": (
                        atlas_broad["hit_probability_at_20"]
                        - atlas_broad_universal["hit_probability_at_20"]
                    ),
                },
            },
        },
        "supplemental_generalization": {
            "orphan335_author_pool": {
                "scope": (
                    "Historical/supplemental frozen author-pool comparison; not a "
                    "current BiME-v2 zero-shot canonical claim."
                ),
                "queries": orphan["queries"],
                "candidate_uids": orphan["candidate_uids"],
                "selenzyme_r2e": orphan["selenzyme_metrics"]["reaction_to_enzyme"],
                "project_r2e": orphan["catalyst_metrics"]["reaction_to_enzyme"],
            }
        },
        "conditional_experimental_feedback": {
            "one_seed_r2e": conditional["r2e"],
            "one_seed_e2r": conditional["e2r"],
            "multi_seed_scaling": conditional["multi_seed_scaling"],
        },
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                SCORECARD,
                R2E_INTERNAL,
                E2R_INTERNAL,
                ORPHAN,
                ATLAS_TPS / "summary.json",
                ATLAS_TPS / "metrics.csv",
                ATLAS_TPS_BROAD / "summary.json",
                ATLAS_TPS_BROAD / "metrics.csv",
                HISTORICAL_TPS_MULTI,
            )
        },
    }

    OUTPUT.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "atlas_tps_double_cold": atlas_double,
        "atlas_tps_double_cold_e2r": atlas_double_e2r,
        "atlas_tps_broad": atlas_broad,
        "atlas_tps_broad_universal": atlas_broad_universal,
        "atlas_minus_marts_only": atlas_vs_old,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
