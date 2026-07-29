from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/terpene_protocol_reassessment/validation.json"


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    focused = (ROOT / "docs/terpene_retrieval_protocol_reassessment_zh.md").read_text(encoding="utf-8")
    main_report = (ROOT / "docs/terpene_candidate_retrieval_comprehensive_report_zh.md").read_text(encoding="utf-8")
    taxonomy = pd.read_csv(ROOT / "results/terpene_protocol_reassessment/protocol_taxonomy.csv")
    capability = pd.read_csv(ROOT / "results/terpene_protocol_reassessment/capability_spectrum.csv")
    cold = pd.read_csv(ROOT / "results/terpene_protocol_reassessment/same_model_cold_protocol_matrix.csv")
    few = pd.read_csv(ROOT / "results/terpene_protocol_reassessment/fewshot_protocol_matrix.csv")
    exact = pd.read_csv(ROOT / "results/terpene_protocol_reassessment/current_library_exact_matrix.csv")

    def cold_value(protocol: str, direction: str, field: str) -> float:
        return float(cold[(cold.protocol == protocol) & (cold.direction == direction)].iloc[0][field])

    checks = {
        "taxonomy_has_three_axes": set([
            "seed_status", "reaction_exact_seen_in_fold_training", "reaction_cluster_may_be_seen",
            "positive_protein_cluster_may_be_seen", "homology_allowed"
        ]).issubset(taxonomy.columns),
        "taxonomy_has_six_tracks": len(taxonomy) == 6,
        "double_cold_not_zero_shot_synonym": "double-cold 只是二维平面的一个角，不是 zero-shot 的同义词" in main_report,
        "focused_has_two_dimensional_matrix": "无 seed 的二维新颖性矩阵" in focused,
        "focused_has_seed_third_axis": "Few-shot 是第三个轴" in focused,
        "exact_semantics_whole_reaction_held_out": "整条 exact reaction ID" in focused and "整条 exact reaction ID" in main_report,
        "old_incorrect_pair_wording_absent": "只隐藏 exact 关联" not in focused and "隐藏单条 pair" in focused,
        "same_model_vs_practical_routes_separated": "不能拿来宣称同一个模型在不同 split 上提升了多少" in focused and "不能被解释成同一个模型从 6% 提升到 70%" in main_report,
        "homology_is_legal_evidence": "相似蛋白是合法且极有价值的生产证据" in focused,
        "exact_top10": close(float(exact[exact.budget == 10].iloc[0].hit), 0.48148148148148145),
        "reaction_cold_r2e_top10": close(cold_value("reaction_cold", "reaction_to_enzyme", "hit_at_10"), 0.28654970760233917),
        "protein_cold_r2e_top10": close(cold_value("protein_cold", "reaction_to_enzyme", "hit_at_10"), 0.18067226890756302),
        "double_cold_r2e_top10": close(cold_value("double_cold", "reaction_to_enzyme", "hit_at_10"), 0.06382978723404255),
        "random_1_kmer_top10": close(float(few[(few.protocol == "random_positive") & (few.n_seeds == 1) & (few.method == "kmer3_max_jaccard")].iloc[0].hit_at_10), 0.7365671641791045),
        "random_5_kmer_top10": close(float(few[(few.protocol == "random_positive") & (few.n_seeds == 5) & (few.method == "kmer3_max_jaccard")].iloc[0].hit_at_10), 0.928169014084507),
        "capability_marks_comparison_groups": set(capability.comparison_group) == {"best_practical_route", "same_model_split_ablation"},
        "main_calibration_scope_clarified": "不能解释为 homolog-visible 或 few-shot 场景的实际成功概率" in main_report,
    }
    status = "passed" if all(checks.values()) else "failed"
    payload = {
        "status": status,
        "checks": checks,
        "principle": "Seed availability, reaction novelty, and protein novelty are orthogonal axes; homology-enabled practical retrieval and cold-start extrapolation are co-primary tracks.",
        "files": {
            "focused_report": "docs/terpene_retrieval_protocol_reassessment_zh.md",
            "main_report": "docs/terpene_candidate_retrieval_comprehensive_report_zh.md",
            "taxonomy": "results/terpene_protocol_reassessment/protocol_taxonomy.csv",
            "capability_spectrum": "results/terpene_protocol_reassessment/capability_spectrum.csv",
        },
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
