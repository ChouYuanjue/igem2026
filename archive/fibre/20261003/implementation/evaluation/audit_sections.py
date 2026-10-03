from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _expand_r2e(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"query_id": str})
    rows = []
    for r in d.itertuples(index=False):
        for x in json.loads(r.positive_nodal_scores_json):
            rows.append({
                "fold": int(r.fold),
                "reaction_id": str(r.query_id),
                "protein_id": str(x["protein_id"]),
                "r2e_prior": float(x["prior_score"]),
                "r2e_flow": float(x["flow_score"]),
                "r2e_delta": float(x["delta"]),
                "r2e_chart_index": int(x["chart_index"]),
                "r2e_characteristic_scale": float(r.characteristic_scale),
            })
    return pd.DataFrame(rows)


def _expand_e2r(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"query_id": str})
    rows = []
    for r in d.itertuples(index=False):
        for x in json.loads(r.positive_nodal_scores_json):
            rows.append({
                "fold": int(r.fold),
                "reaction_id": str(x["reaction_id"]),
                "protein_id": str(r.query_id),
                "e2r_prior": float(x["prior_score"]),
                "e2r_flow": float(x["flow_score"]),
                "e2r_delta": float(x["delta"]),
                "e2r_chart_index": int(x["chart_index"]),
                "e2r_characteristic_scale": float(r.characteristic_scale),
            })
    return pd.DataFrame(rows)


def _quantiles(x: pd.Series) -> dict[str, float]:
    a = x.to_numpy(dtype=float)
    if not len(a):
        return {}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p90": float(np.quantile(a, 0.90)),
        "p95": float(np.quantile(a, 0.95)),
        "max": float(np.max(a)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r2e-diagnostics", type=Path, required=True)
    ap.add_argument("--e2r-diagnostics", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    r = _expand_r2e(args.r2e_diagnostics)
    e = _expand_e2r(args.e2r_diagnostics)
    if r.empty or e.empty:
        raise RuntimeError("no positive nodal observations found in one of the probes")
    z = r.merge(e, on=["fold", "reaction_id", "protein_id"], how="inner", validate="one_to_one")
    if z.empty:
        raise RuntimeError("no held-out pair was represented as a node in both local sections")

    z["prior_defect"] = z.r2e_prior - z.e2r_prior
    z["flow_defect"] = z.r2e_flow - z.e2r_flow
    z["delta_defect"] = z.r2e_delta - z.e2r_delta
    z["abs_prior_defect"] = z.prior_defect.abs()
    z["abs_flow_defect"] = z.flow_defect.abs()
    z["abs_delta_defect"] = z.delta_defect.abs()
    z["same_deformation_sign"] = np.sign(z.r2e_delta) == np.sign(z.e2r_delta)
    z["mean_section_scale"] = 0.5 * (z.r2e_characteristic_scale + z.e2r_characteristic_scale)
    z["relative_flow_defect"] = z.abs_flow_defect / np.maximum(z.mean_section_scale.abs(), 1e-12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    z.to_csv(args.output, index=False)
    summary = {
        "schema": "product-field-local-section-overlap-audit-v1",
        "role": "numerical atlas consistency audit; held-out positives select evaluation overlap points but never enter scoring",
        "pairs_represented_as_nodes_in_both_sections": int(len(z)),
        "unique_reactions": int(z.reaction_id.nunique()),
        "unique_proteins": int(z.protein_id.nunique()),
        "prior_abs_defect": _quantiles(z.abs_prior_defect),
        "postflow_abs_defect": _quantiles(z.abs_flow_defect),
        "deformation_abs_defect": _quantiles(z.abs_delta_defect),
        "relative_postflow_defect_over_mean_characteristic_scale": _quantiles(z.relative_flow_defect),
        "deformation_sign_agreement_fraction": float(z.same_deformation_sign.mean()),
        "r2e_e2r_deformation_pearson": float(z.r2e_delta.corr(z.e2r_delta, method="pearson")),
        "r2e_e2r_deformation_spearman": float(z.r2e_delta.corr(z.e2r_delta, method="spearman")),
        "labels_used_for_scoring": False,
        "external_metrics_used": False,
    }
    sp = args.output.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
