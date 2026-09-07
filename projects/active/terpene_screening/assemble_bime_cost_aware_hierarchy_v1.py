from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "projects/active/terpene_screening/BIME_RANK_COST_AWARE_HIERARCHY_V1_RESULT.json"
TIMING_PATH = "reproducibility/bime_rank/enzgfm_stage2_530_timing_20260907.json"


def J(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def assembled_payload() -> dict:
    retention_path = "results/bime_rank_unified_v1/cost_aware_shortlist_retention_v1/summary.json"
    retention = J(retention_path)
    wetlab_stage1 = J("results/requested_r2e20_bime_v2_20260906/summary.json")
    stage2 = J("results/requested_r2e20_bime_v2_20260906/stage2_summary.json")
    stage2_manifest = J("results/requested_r2e20_bime_v2_20260906/enzgfm_stage2_530/manifest.json")
    wetlab = J("results/requested_r2e20_bime_v2_20260906/MANUAL_SUCCESS_FIRST_SUMMARY.json")
    timing = J(TIMING_PATH)
    admission = J("projects/active/terpene_screening/BIME_RANK_EXPERT_ADMISSION_V1.json")

    candidate_universe = int(wetlab_stage1["lab_candidates"])
    stage2_unique = int(stage2["unique_stage2_proteins"])
    if stage2_unique != int(stage2_manifest["protein_count"]):
        raise RuntimeError("stage2 unique protein count disagrees with EnzGFM manifest")
    match = re.search(r"stage1 top(\d+)/reaction", str(stage2["candidate_cascade"]))
    if not match:
        raise RuntimeError("cannot recover stage1 top-K from retained stage2 cascade")
    stage1_top = int(match.group(1))
    if int(timing["inputs"]["sequence_count"]) != stage2_unique:
        raise RuntimeError("timing sequence count disagrees with stage2 candidate count")
    if timing["status"] != "verified_rerun_exact_output_identity":
        raise RuntimeError("timing provenance is not verified")

    sweep_by_k = {int(row["k_per_base_expert"]): row for row in retention["sweep"]}
    for required in (100, 1000, 2000):
        if required not in sweep_by_k:
            raise RuntimeError(f"missing retention sweep K={required}")

    clip_admission = admission["experts"]["r2e_clipzyme_structure"]
    cage_admission = admission["experts"]["enzymecage_top20_structure"]

    return {
      "name": "BiME-Rank cost-aware hierarchical expert execution",
      "version": "bime-rank-cost-aware-hierarchy-v1",
      "status": "validated_execution_layer_current_ranking_semantics_preserved",
      "scientific_role": "execution policy over independently admitted experts; not a new expert and not a replacement for clean-development expert admission",
      "execution_modes": {
        "global_cached": {
          "examples": ["ESM-C", "EnzGFM on registered candidate universes"],
          "policy": "score full candidate universe because candidate-side features are already materialized"
        },
        "query_conditional_cached_candidate_generator": {
          "examples": ["CLIPZyme"],
          "policy": "when query and cached candidate assets are supported, allow full-supported scoring so the specialist may introduce candidates absent from cheap expert prefixes; otherwise exact frozen fallback"
        },
        "shortlist_on_demand": {
          "examples": [
            "EnzGFM on a temporary uncached candidate library",
            "future pocket/geometry specialists that independently pass expert admission"
          ],
          "policy": "freeze a shortlist using cheaper available experts, materialize expensive candidate-side features only inside that shortlist, rerank the prefix, preserve frozen tail/fallback"
        }
      },
      "internal_shortlist_retention": retention,
      "key_internal_observations": {
        "clip_supported_queries": int(retention["clip_supported_queries"]),
        "specialist_rescue_queries_over_base100": int(retention["specialist_rescue_queries_over_base100"]),
        "base_k_100_retention": float(sweep_by_k[100]["retention_of_full_specialist_positive_queries"]),
        "base_k_1000_retention": float(sweep_by_k[1000]["retention_of_full_specialist_positive_queries"]),
        "base_k_2000_retention": float(sweep_by_k[2000]["retention_of_full_specialist_positive_queries"]),
        "lesson": "a blanket Top-50/Top-100 restriction is inappropriate for candidate-generating specialists; shortlist restriction is reserved for uncached reranking specialists"
      },
      "wet_lab_case": {
        "candidate_universe": candidate_universe,
        "stage1_top_per_reaction": stage1_top,
        "stage2_unique_candidates": stage2_unique,
        "stage2_fraction": stage2_unique / candidate_universe,
        "exact_enzgfm_650m_elapsed_seconds": float(timing["rerun"]["wall_seconds"]),
        "exact_enzgfm_650m_timing_provenance": TIMING_PATH,
        "unique_reactions": int(wetlab["unique_reactions"]),
        "unique_primary_constructs": int(wetlab["unique_primary_constructs"]),
        "unique_primary_plus_backup_constructs": int(wetlab["unique_primary_plus_backup_constructs"]),
        "role": "application case study of on-demand feature materialization plus task-specific decision-layer constraints; not a universal benchmark configuration"
      },
      "expert_admission_context": {
        "clipzyme_r2e": {
          "status": "admitted" if clip_admission["status"] == "promoted_production" else clip_admission["status"],
          "internal_delta": clip_admission["internal_delta"],
          "reason": "clean-development structural complementarity passed the existing admission gate"
        },
        "enzymecage_tps_top20": {
          "status": "not_admitted" if cage_admission["status"] == "rejected_internal_oof" else cage_admission["status"],
          "delta_vs_same_capacity": cage_admission["delta"],
          "reason": "official EnzymeCAGE Top20 conditional expert did not improve the same-capacity internal OOF ranker and remains inactive"
        }
      },
      "invariants": [
        "external benchmark labels are not used to choose expert admission or shortlist budget",
        "expert absence is never encoded as a low score",
        "unsupported specialists preserve the exact frozen fallback path",
        "task-specific motif/expression/taxonomy/inventory constraints remain outside the universal retrieval ranker",
        "a specialist may generate new candidates only when it has an independently indexed/cached candidate representation; uncached specialists are reranking-only"
      ],
      "implementation": {
        "planner": "projects/active/terpene_screening/hierarchical_expert_routing.py",
        "r2e_runtime": "projects/active/terpene_screening/bime_rank_r2e_runtime.py",
        "e2r_runtime": "projects/active/terpene_screening/bime_rank_e2r_runtime.py",
        "retention_evaluator": "projects/active/terpene_screening/evaluate_bime_cost_aware_shortlist_retention_v1.py",
        "contract_test": "projects/active/terpene_screening/tests/test_hierarchical_expert_routing.py"
      }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically assemble the BiME-Rank cost-aware execution aggregate from retained evidence.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true", help="Require assembled semantic content to equal the existing output without modifying it.")
    args = parser.parse_args()
    target = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        display = str(target.relative_to(ROOT))
    except ValueError:
        display = str(target)
    value = assembled_payload()
    if args.verify_existing:
        if not target.is_file():
            raise FileNotFoundError(target)
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != value:
            raise SystemExit(f"assembled cost-aware payload differs from {target}")
        print(json.dumps({"status":"match","output":display}, indent=2))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status":"written","output":display}, indent=2))


if __name__ == "__main__":
    main()
