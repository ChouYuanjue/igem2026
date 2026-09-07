"""Validate judge-facing BiME-Rank headline numbers against canonical primaries.

No inference, training, benchmark execution, or asset mutation is performed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "docs/release/bime_rank/BiME_Rank_Judge_Report.tex"
INDEX = ROOT / "reproducibility/bime_rank/canonical.json"
OUT = ROOT / "reproducibility/bime_rank/judge_report_validation.json"


def load_claim(index: dict, key: str) -> dict:
    return json.loads((ROOT / index["claims"][key]["primary"]).read_text())


def pct(value: float) -> str:
    return f"{100 * value:.2f}\\%"


def num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def require(text: str, fragment: str, label: str, checks: list[str]) -> None:
    if fragment not in text:
        raise AssertionError(f"judge report mismatch for {label}: expected fragment {fragment!r}")
    checks.append(label)


def main() -> None:
    index = json.loads(INDEX.read_text())
    text = TEX.read_text()
    checks: list[str] = []

    r2e = load_claim(index, "clipzyme_r2e")
    rm = r2e["metrics"]
    rb = r2e["paired_bootstrap_50000"]
    require(text, f"8.33\\% & \\textbf{{{pct(rm['hit_at_20'])}}}", "CLIP R2E Hit@20", checks)
    require(text, f"9.72\\% & \\textbf{{{pct(rm['hit_at_50'])}}}", "CLIP R2E Hit@50", checks)
    ci50 = rb["hit_at_50"]["clipzyme"]["ci95"]
    require(text, f"[+{100*ci50[0]:.2f},+{100*ci50[1]:.2f}]\\pp", "CLIP R2E Hit@50 CI", checks)

    e2r = load_claim(index, "clipzyme_e2r")
    em = e2r["models"]
    require(text, f"{pct(em['official_clipzyme']['hit_at_20'])} & \\textbf{{{pct(em['clipzyme_five_expert_v4']['hit_at_20'])}}}", "CLIP E2R Hit@20", checks)
    require(text, f"{pct(em['official_clipzyme']['hit_at_50'])} & \\textbf{{{pct(em['clipzyme_five_expert_v4']['hit_at_50'])}}}", "CLIP E2R Hit@50", checks)

    e405 = load_claim(index, "enzyme405")
    m405 = e405["paired_bootstrap_50000_vs_enzymecage"]["metrics"]
    require(text, f"51.33\\% & \\textbf{{{pct(m405['hit_at_10']['catalyst'])}}}", "Enzyme-405 Hit@10", checks)
    require(text, f"0.2517 & \\textbf{{{num(m405['reciprocal_rank']['catalyst'])}}}", "Enzyme-405 MRR", checks)
    require(text, f"0.2521 & \\textbf{{{num(m405['average_precision']['catalyst'])}}}", "Enzyme-405 MAP", checks)

    # Pre-BiME Catalyst/MARTS quantitative results are deliberately absent from the current judge release.
    if "Catalyst 前身路线" in text or r"前 10 命中率从 30.28\% 提高到 47.49\%" in text:
        raise AssertionError("pre-BiME Catalyst quantitative result re-entered current judge tables")

    scorecard = load_claim(index, "judge_evidence")
    cond = scorecard["conditional_known_positive"]
    rseed = cond["r2e"]["metrics"]
    eseed = cond["e2r"]["metrics"]
    require(text, rf"R2E & \textbf{{{num(rseed['mrr'])}}} & \textbf{{{pct(rseed['hit_at_10'])}}} & \textbf{{{pct(rseed['hit_at_20'])}}} & \textbf{{{pct(rseed['hit_at_50'])}}}", "R2E one-seed conditional", checks)
    require(text, rf"E2R & \textbf{{{num(eseed['mrr'])}}} & \textbf{{{pct(eseed['hit_at_10'])}}} & \textbf{{{pct(eseed['hit_at_20'])}}} & \textbf{{{pct(eseed['hit_at_50'])}}}", "E2R one-seed conditional", checks)
    mult = cond["multi_seed_scaling"]
    r5 = mult["r2e"]["metrics"]["context"]["5"]
    e5 = mult["e2r"]["metrics"]["context"]["5"]
    require(text, rf"R2E & 5 & \textbf{{{num(r5['mrr'])}}} & \textbf{{{pct(r5['hit_at_10'])}}} & \textbf{{{pct(r5['hit_at_50'])}}}", "R2E five-seed scaling", checks)
    require(text, rf"E2R & 5 & \textbf{{{num(e5['mrr'])}}} & \textbf{{{pct(e5['hit_at_10'])}}} & {pct(e5['hit_at_50'])}", "E2R five-seed scaling", checks)

    cost = load_claim(index, "cost_aware")
    wet = cost["wet_lab_case"]
    require(text, rf"\textbf{{{wet['stage2_unique_candidates']} 个蛋白，占原库 {pct(wet['stage2_fraction'])}}}", "Cost-aware wet-lab shortlist", checks)
    require(text, rf"\textbf{{{wet['exact_enzgfm_650m_elapsed_seconds']:.2f} s}}", "Cost-aware exact EnzGFM timing", checks)

    admission = load_claim(index, "expert_admission")["experts"]["enzymecage_top20_structure"]
    base = admission["same_capacity_baseline"]
    cand = admission["candidate"]
    require(text, f"MRR 由 {num(base['mrr'])} 变为 {num(cand['mrr'])}", "Rejected EnzymeCAGE expert MRR", checks)
    require(text, "三者都不属于当前 BiME-Rank 排序专家", "Rejected experts excluded from current model", checks)

    stale = [
        "\\textbf{11.11\\%}",
        "\\textbf{21.53\\%}",
        "[+4.17,+19.44]\\pp",
        "BiME_Rank_Judge_Report_20260907_v9.bib",
        "673 个 query",
        "62.60\\%",
        "68.14\\%",
        "82.58\\%",
        "86.69\\%",
        "38.85\\%",
        "50.94\\%",
        "54.27\\%",
        "40.54\\%",
        "31.24\\%",
        "15.48\\%",
        "43.37\\%",
        "31.73\\%",
        "10.58\\%",
        "17.31\\%",
        "23.93\\%",
        "29.44\\%",
    ]
    found_stale = [x for x in stale if x in text]
    if found_stale:
        raise AssertionError(f"stale judge-report tokens remain: {found_stale}")
    if "\\addbibresource{references.bib}" not in text:
        raise AssertionError("release bibliography path is not stable")
    if not (TEX.parent / "references.bib").is_file():
        raise AssertionError("release bibliography is missing")

    result = {
        "report": str(TEX.relative_to(ROOT)),
        "checks_passed": len(checks),
        "checks": checks,
        "stale_tokens_absent": True,
        "canonical_claims_used": ["clipzyme_r2e", "clipzyme_e2r", "enzyme405", "multi_seed", "r2e_seed_retention", "e2r_seed_retention", "expert_admission", "cost_aware", "judge_evidence"],
        "inference_or_experiments_run": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
