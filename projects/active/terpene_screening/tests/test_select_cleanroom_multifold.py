import pandas as pd
from projects.active.terpene_screening.select_cleanroom_multifold import METRICS, select_multifold


def test_multifold_selection_prefers_consistent_multi_metric_candidate() -> None:
    rows=[]
    for fold in range(3):
        for candidate, value in [('stable',0.8),('spiky',0.6)]:
            row={'candidate':candidate,'fold':fold}
            for metric in METRICS: row[metric]=value
            rows.append(row)
    # spiky wins one metric/fold dramatically, but should not overturn broad consistency.
    rows[-1][METRICS[0]]=2.0
    _, agg=select_multifold(pd.DataFrame(rows))
    assert agg.iloc[0].candidate=='stable'
    assert bool(agg.iloc[0].selected)


def test_multifold_requires_equal_fold_coverage() -> None:
    rows=[]
    for candidate,folds in [('a',[0,1]),('b',[0])]:
        for fold in folds:
            row={'candidate':candidate,'fold':fold}; row.update({m:.5 for m in METRICS}); rows.append(row)
    try:
        select_multifold(pd.DataFrame(rows))
    except ValueError as exc:
        assert 'unequal fold counts' in str(exc)
    else:
        raise AssertionError('expected unequal folds to fail')
