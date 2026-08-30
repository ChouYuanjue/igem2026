from __future__ import annotations

import pandas as pd

from projects.active.terpene_screening.evaluate_common_reservoir_lambdarank import build_features, source_columns


def test_features_include_cage_and_query_local_expert_ranks():
    frame=pd.DataFrame({
        'reaction_id':['R1','R1','R2','R2'],
        'pure_cage':[.2,.8,.5,.1],
        'direct:a':[.9,.1,.3,.7],
        'direct:b':[.7,.2,.4,.6],
    })
    sources=source_columns(frame)
    features,names=build_features(frame,'reaction_id',sources)
    assert sources == ['pure_cage','direct:a','direct:b']
    assert 'pure_cage|pct' in names and 'expert_pct_std' in names
    assert features.loc[0,'direct:a|pct'] > features.loc[1,'direct:a|pct']
    assert features.loc[1,'pure_cage|pct'] > features.loc[0,'pure_cage|pct']
