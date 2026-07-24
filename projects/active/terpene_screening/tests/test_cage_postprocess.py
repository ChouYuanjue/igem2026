from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.run_cage_inference import _postprocess_scores


def test_postprocess_uses_logit_to_break_saturated_probability_ties(tmp_path):
    raw_path = tmp_path / "raw.csv"
    final_path = tmp_path / "final.csv"
    pd.DataFrame(
        [
            {
                "reaction_id": "r1",
                "rhea_id": "r1",
                "enzyme_id": "e1",
                "uniprot_id": "e1",
                "sequence": "AAAA",
                "CANO_RXN_SMILES": "CC>>CC",
                "label": 0,
                "pred": 0.0,
                "pred_logit": -100.0,
            },
            {
                "reaction_id": "r1",
                "rhea_id": "r1",
                "enzyme_id": "e2",
                "uniprot_id": "e2",
                "sequence": "AAAA",
                "CANO_RXN_SMILES": "CC>>CC",
                "label": 1,
                "pred": 0.0,
                "pred_logit": -10.0,
            },
        ]
    ).to_csv(raw_path, index=False)

    summary = _postprocess_scores(raw_path, final_path)
    result = pd.read_csv(final_path)

    assert result["uniprot_id"].tolist() == ["e2", "e1"]
    assert result["rank_within_reaction"].tolist() == [1, 2]
    assert result["cage_rank_score"].tolist() == [1.0, 0.0]
    assert not result["cage_all_logits_tied"].any()
    assert summary["n_reactions_all_logits_tied"] == 0
    assert summary["median_logit_range_within_reaction"] == 90.0
