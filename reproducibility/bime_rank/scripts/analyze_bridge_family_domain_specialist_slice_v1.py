from __future__ import annotations

import json

from projects.active.bridge.model.assets import ROOT

SOURCE = ROOT / "results/fibre_autonomous_domain_heldout/summary.json"
OUT = ROOT / "reproducibility/bime_rank/records/BRIDGE_FAMILY_DOMAIN_SPECIALIST_SLICE_V1_RESULT.json"
DOC = ROOT / "projects/active/bridge/docs/family_domain_specialist_slice_v1.md"

DISPLAY = {
    "p450": "P450",
    "phosphatase": "Phosphatase",
    "terpene": "Terpene",
}


def main() -> None:
    source = json.loads(SOURCE.read_text())
    families = source["families"]

    rows = []
    for family in ("p450", "phosphatase", "terpene"):
        item = families[family]
        before = item["broad"]
        after = item["autonomous_domain_system"]
        rows.append(
            {
                "family": family,
                "display": DISPLAY[family],
                "mrr_before": float(before["mrr"]),
                "mrr_after": float(after["mrr"]),
                "hit10_before": float(before["hit10"]),
                "hit10_after": float(after["hit10"]),
            }
        )

    result = {
        "schema": "bridge-family-domain-specialist-slice-v1",
        "status": "completed",
        "source": str(SOURCE.relative_to(ROOT)),
        "dataset": "dedicated family-specific held-out evaluation",
        "split": source["family_test_split"],
        "before": "Broad baseline before Family/Domain specialist correction",
        "after": "family/domain integrated system on the same held-out family test",
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "# BRIDGE Family/Domain specialist evaluation",
        "",
        "使用三个家族各自的专用 held-out test；Before 与 After 在完全相同的家族测试集和候选池上比较。",
        "",
        "| Family | MRR Before | MRR After | Hit@10 Before | Hit@10 After |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['mrr_before']:.4f} | {row['mrr_after']:.4f} | "
            f"{100*row['hit10_before']:.2f}% | {100*row['hit10_after']:.2f}% |"
        )
    lines.append("")
    DOC.write_text("\n".join(lines))

    print(DOC)
    print(OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
