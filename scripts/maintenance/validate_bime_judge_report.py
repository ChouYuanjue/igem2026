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

    tps = load_claim(index, "tps_practical")["main_external_comparison"]
    cage = tps["enzymecage_official_algorithm_reproduction"]
    bime = tps["catalyst_locked_practical_route"]
    require(text, f"{pct(cage['hit10'])}/{pct(cage['hit20'])}", "TPS EnzymeCAGE Hit@10/20", checks)
    require(text, f"{pct(bime['hit10'])}/{pct(bime['hit20'])}", "TPS BiME Hit@10/20", checks)

    sel = load_claim(index, "selenzyme")
    require(text, f"Selenzyme 的前 10 命中率为 {pct(sel['selenzyme_metrics']['reaction_to_enzyme']['hit_at_10'])}，BiME-Rank 为 {pct(sel['catalyst_metrics']['reaction_to_enzyme']['hit_at_10'])}", "Selenzyme Hit@10", checks)
    require(text, f"MRR 为 {num(sel['selenzyme_metrics']['reaction_to_enzyme']['mrr'])} 对 {num(sel['catalyst_metrics']['reaction_to_enzyme']['mrr'])}", "Selenzyme MRR", checks)

    stale = ["\\textbf{11.11\\%}", "\\textbf{21.53\\%}", "[+4.17,+19.44]\\pp", "BiME_Rank_Judge_Report_20260907_v9.bib"]
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
        "canonical_claims_used": ["clipzyme_r2e", "clipzyme_e2r", "enzyme405", "tps_practical", "selenzyme"],
        "inference_or_experiments_run": False,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
