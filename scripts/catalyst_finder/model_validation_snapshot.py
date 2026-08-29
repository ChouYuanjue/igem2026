from __future__ import annotations

from typing import Any

# Fixed retrospective audit captured on 2026-08-28. The query target is inside the
# original project-aligned catalog, while the evaluated positives come only from the
# current official Rhea↔Swiss-Prot mapping after removing every relation present in the
# project pair catalog. This is intentionally a product-facing retrospective check, not
# an independent held-out benchmark and never a training/runtime membership source.
PROJECT_ALIGNED_EXTERNAL_RELATION_AUDIT: dict[str, dict[str, Any]] = {
    "reaction_to_enzyme": {
        "audit_id": "rhea-sprot-project-aligned-r2e-20260828-v1",
        "queries": 32,
        "hit_at_1": 0.0,
        "hit_at_3": 0.3125,
        "hit_at_5": 0.5,
        "hit_at_10": 0.625,
        "hit_at_20": 0.6562,
        "median_best_rank_among_hits": 4,
    },
    "enzyme_to_reaction": {
        "audit_id": "rhea-sprot-project-aligned-e2r-20260828-v1",
        "queries": 60,
        "hit_at_1": 0.2333,
        "hit_at_3": 0.3833,
        "hit_at_5": 0.4333,
        "hit_at_10": 0.4667,
        "hit_at_20": 0.4833,
        "median_best_rank_among_hits": 2,
    },
}

AUDIT_CONTEXT = {
    "evaluated_relation_source": "Rhea official Swiss-Prot mapping",
    "project_pairs_excluded": True,
    "query_targets": "project-aligned catalog",
    "purpose": "retrospective external-relation recovery check",
    "independent_test_set": False,
    "captured_date": "2026-08-28",
}
