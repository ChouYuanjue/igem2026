from projects.active.terpene_screening.compare_broad_rhea_full_candidate import compare_summaries


def test_comparison_respects_rank_directionality_and_reports_regressions() -> None:
    base={"metrics":{"reaction_to_enzyme":{"mrr":.1,"map":.1,"median_best_positive_rank":100,"hit_at_1":.1},"enzyme_to_reaction":{"mrr":.2,"map":.2,"median_best_positive_rank":50,"hit_at_1":.2}}}
    cand={"metrics":{"reaction_to_enzyme":{"mrr":.2,"map":.15,"median_best_positive_rank":80,"hit_at_1":.09},"enzyme_to_reaction":{"mrr":.25,"map":.21,"median_best_positive_rank":40,"hit_at_1":.2}}}
    frame, summary=compare_summaries(base,cand)
    rank=frame[(frame.direction=='reaction_to_enzyme') & (frame.metric=='median_best_positive_rank')].iloc[0]
    assert rank.improvement_delta == 20
    assert rank.status == 'improved'
    assert any(x['metric']=='hit_at_1' and x['direction']=='reaction_to_enzyme' for x in summary['regressions'])
