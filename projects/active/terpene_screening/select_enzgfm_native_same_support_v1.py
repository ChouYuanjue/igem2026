from __future__ import annotations

import json
from pathlib import Path

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import RESULT_ROOT, balanced_selection_score, sha256_file

CANDIDATES = ("dual_tower", "author_pairwise")


def main() -> None:
    out_dir = RESULT_ROOT / "selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "summary.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite frozen selection: {out}")
    rows = []
    for candidate in CANDIDATES:
        path = RESULT_ROOT / "development" / candidate / "dev_evaluation.json"
        if not path.exists():
            raise SystemExit(f"missing preregistered development evaluation: {path}")
        d = json.loads(path.read_text())
        if d.get("status") != "development_internal_only" or d.get("stage") != "development":
            raise AssertionError(f"invalid development evidence for {candidate}")
        e2r, r2e = d["e2r"], d["r2e"]
        rows.append({
            "candidate": candidate,
            "balanced_map_harmonic_mean": balanced_selection_score(e2r, r2e),
            "mean_map": 0.5 * (float(e2r["map"]) + float(r2e["map"])),
            "mean_hit_at_5": 0.5 * (float(e2r["hit_at_5"]) + float(r2e["hit_at_5"])),
            "mean_ndcg_at_10": 0.5 * (float(e2r["ndcg_at_10"]) + float(r2e["ndcg_at_10"])),
            "e2r": e2r,
            "r2e": r2e,
            "evaluation_sha256": sha256_file(path),
        })
    # Frozen priority: balanced bidirectional MAP, then mean MAP, then mean Hit@5,
    # then mean NDCG@10, then fixed candidate name for deterministic tie resolution.
    selected = max(rows, key=lambda x: (
        x["balanced_map_harmonic_mean"], x["mean_map"], x["mean_hit_at_5"], x["mean_ndcg_at_10"], -CANDIDATES.index(x["candidate"])
    ))
    result = {
        "status": "frozen_development_selection_before_native_test_reveal",
        "selection_source": "ReactZyme author train_val only; deterministic protein-disjoint internal dev",
        "selection_rule": ["balanced_map_harmonic_mean", "mean_map", "mean_hit_at_5", "mean_ndcg_at_10", "fixed_candidate_order"],
        "candidates": rows,
        "selected_candidate": selected["candidate"],
        "native_test_metrics_read": False,
        "native_test_metrics_used_for_selection": False,
        "alternative_candidate_outer_scoring_allowed": False,
        "post_selection_rule": "Retrain only the selected candidate on all retained author train_val pairs with the exact frozen recipe, then reveal native test exactly once.",
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
