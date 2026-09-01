from __future__ import annotations

import pytest

pytest.importorskip("xgboost", reason="optional LambdaRank research route is not part of the portable terpene runtime")

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


def test_cli_help_runs_as_standalone_script():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    script = root / "projects/active/terpene_screening/evaluate_common_reservoir_lambdarank.py"
    completed = subprocess.run([str(root / ".venv/bin/python"), str(script), "--help"], cwd=root, capture_output=True, text=True, timeout=20)
    assert completed.returncode == 0, completed.stderr
    assert "LambdaRank" in completed.stdout
