from pathlib import Path

import pandas as pd

from projects.active.terpene_screening.evaluate_pure_cage_baseline import evaluate, load_cage_union


def test_union_prefers_later_real_score_source(tmp_path: Path) -> None:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame({"reaction_id": ["R1", "R1"], "uniprot_id": ["P1", "P2"], "cage_score": [0.1, 0.2]}).to_csv(a, index=False)
    pd.DataFrame({"reaction_id": ["R1"], "uniprot_id": ["P1"], "pred": [0.9]}).to_csv(b, index=False)
    union = load_cage_union([a, b]).set_index(["reaction_id", "uniprot_id"])
    assert union.loc[("R1", "P1"), "cage_score"] == 0.9
    assert union.loc[("R1", "P2"), "cage_score"] == 0.2


def test_native_and_end_to_end_metrics_separate_missing_support(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    positives = tmp_path / "positives.tsv"
    out = tmp_path / "out"
    pd.DataFrame(
        {
            "reaction_id": ["R1", "R1", "R2"],
            "uniprot_id": ["P1", "N1", "N2"],
            "cage_score": [0.8, 0.9, 0.7],
        }
    ).to_csv(scores, index=False)
    pd.DataFrame(
        {"rhea_id": ["R1", "R2"], "Entry": ["P1", "P2"]}
    ).to_csv(positives, sep="\t", index=False)
    summary = evaluate([scores], positives, out)
    r2e = summary["reaction_to_enzyme"]
    assert r2e["query_count_end_to_end"] == 2
    assert r2e["query_count_native_evaluable"] == 1
    assert r2e["query_positive_coverage"] == 0.5
    assert r2e["hit_at_1_native"] == 0.0
    assert r2e["hit_at_3_native"] == 1.0
    assert r2e["hit_at_3_end_to_end"] == 0.5
