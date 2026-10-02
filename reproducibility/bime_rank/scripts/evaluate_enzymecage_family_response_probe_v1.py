from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/enzymecage_family_response_v1"
FAMILIES = ("p450", "phosphatase", "terpene")
TAGS = {f: f"{f}_broad_top64_probe" for f in FAMILIES}


def main() -> None:
    rows = []
    confusion = {truth: {pred: 0 for pred in FAMILIES} for truth in FAMILIES}
    per_family = {}

    for truth, tag in TAGS.items():
        frame = pd.read_csv(BASE / tag / "query_features_calibrated.csv")
        fam_percentiles = np.column_stack(
            [
                frame[f"{fam}_response_top10pct_mean_percentile"].to_numpy(float)
                for fam in FAMILIES
            ]
        )
        order = np.argsort(-fam_percentiles, axis=1, kind="stable")
        truth_idx = FAMILIES.index(truth)
        truth_rank = np.asarray(
            [int(np.flatnonzero(order[i] == truth_idx)[0] + 1) for i in range(len(frame))]
        )
        pred = frame["family_calibrated_winner"].astype(str).tolist()
        for value in pred:
            confusion[truth][value] += 1

        top1 = float((truth_rank == 1).mean())
        top2 = float((truth_rank <= 2).mean())
        expected_percentile = fam_percentiles[:, truth_idx]
        other_max = np.max(
            np.delete(fam_percentiles, truth_idx, axis=1), axis=1
        )
        per_family[truth] = {
            "queries": int(len(frame)),
            "top1_domain_identification": top1,
            "top2_domain_identification": top2,
            "expected_family_percentile_mean": float(expected_percentile.mean()),
            "expected_family_percentile_median": float(np.median(expected_percentile)),
            "expected_minus_best_other_percentile_mean": float(
                (expected_percentile - other_max).mean()
            ),
            "calibrated_margin_median": float(
                frame["family_calibrated_margin"].median()
            ),
            "calibrated_entropy_median": float(
                frame["family_calibrated_entropy"].median()
            ),
        }
        for i, row in frame.iterrows():
            record = {
                "truth_family": truth,
                "query": row["CANO_RXN_SMILES"],
                "predicted_family": row["family_calibrated_winner"],
                "truth_rank": int(truth_rank[i]),
                "calibrated_margin": float(row["family_calibrated_margin"]),
                "calibrated_entropy": float(row["family_calibrated_entropy"]),
            }
            for fam in FAMILIES:
                record[f"{fam}_response_percentile"] = float(
                    row[f"{fam}_response_top10pct_mean_percentile"]
                )
            rows.append(record)

    detail = pd.DataFrame(rows)
    detail.to_csv(BASE / "external_probe_domain_diagnostics.csv", index=False)
    total = len(detail)
    result = {
        "schema": "enzymecage-family-response-external-probe-v1",
        "status": "completed",
        "calibration": "author training/valid, labels unused",
        "external_probe": "Broad Top-64 candidate pools; pair labels zeroed before extraction",
        "external_domain_identity_used_for": "diagnostic evaluation only",
        "external_domain_identity_used_for_fit_or_threshold_selection": False,
        "queries": total,
        "macro_top1_domain_identification": float(
            np.mean([per_family[f]["top1_domain_identification"] for f in FAMILIES])
        ),
        "micro_top1_domain_identification": float(
            (detail["truth_family"] == detail["predicted_family"]).mean()
        ),
        "macro_top2_domain_identification": float(
            np.mean([per_family[f]["top2_domain_identification"] for f in FAMILIES])
        ),
        "per_family": per_family,
        "confusion": confusion,
    }
    (BASE / "external_probe_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
