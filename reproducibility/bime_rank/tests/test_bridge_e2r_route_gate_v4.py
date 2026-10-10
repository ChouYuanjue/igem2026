"""Joint E2R evidence-route invariants under frozen source/validation/test partitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.e2r_query_gate import FEATURE_NAMES
from reproducibility.bime_rank.scripts.fit_bridge_e2r_route_gate_v4 import (
    ROUTES, checks, select_route,
)


class _Predict:
    def __init__(self, gain: float):
        self.gain = gain

    def predict(self, x):
        assert x.shape[0] == 1
        return np.asarray([self.gain])


def _model(preferred: str, bonus: float = 0.25) -> dict:
    return {
        "feature_names": FEATURE_NAMES,
        "route_authorities": ROUTES,
        "predictors": {
            name: _Predict(bonus if name == preferred else 0.01)
            for name in ROUTES if name != "broad"
        },
        "baseline_guard": 0.0,
    }


def test_immutable_split_and_external_development_union():
    report = checks()
    assert report["fixed_source_edges"] == 23773
    assert report["fixed_test_edges"] == 21505
    assert report["fixed_development_edges"] == 5216
    assert report["fixed_train_queries"] == 2597
    assert report["fixed_validation_queries"] == 857
    frozen = json.loads(
        (ROOT/"results/bridge_gate_split_v2/manifest.json").read_text()
    )
    assert report["parent_sha256"] == frozen["original_sha256"]
    assert len(ROUTES) == 7


def test_query_route_can_abstain_on_unavailable_evidence():
    features=np.ones(len(FEATURE_NAMES))
    route,authority,predicted=select_route(
        _model("all_evidence"),features,
        {"functional_av":True, "structure_av":False, "relation_av":False},
    )
    assert route == "functional"
    np.testing.assert_array_equal(authority,[1.,0.,0.])
    assert predicted["all_evidence"] == -np.inf
    route,authority,_=select_route(
        _model("relation_emphasis"),features,
        {"functional_av":False,"structure_av":False,"relation_av":False},
    )
    assert route=="broad"
    np.testing.assert_array_equal(authority,[0.,0.,0.])


def test_query_route_uses_joint_full_strengths():
    features=np.zeros(len(FEATURE_NAMES))
    route,weights,_=select_route(
        _model("relation_emphasis"),features,
        {"functional_av":True,"structure_av":True,"relation_av":True},
    )
    assert route=="relation_emphasis"
    np.testing.assert_array_equal(weights,[1.,1.,3.])


def test_frozen_validation_pair_keys_and_hitk():
    p=ROOT/"results/bridge_e2r_route_gate_v4/validation_edge_metrics.csv.gz"
    if not p.exists():
        raise AssertionError("Combined query gate must pass immutable heldout validation first")
    df=pd.read_csv(p,dtype={"protein_id":str,"reaction_id":str})
    assert len(df)==1316
    assert not df.duplicated(["protein_id","reaction_id"]).any()
    assert set(df.novelty)=={"both_seen","protein_cold","reaction_cold","double_cold"}
    for column in ("broad_rank","new_gate_rank","old_independent_gate_rank","fixed_full_rank"):
        ranks=df[column].to_numpy(np.int64)
        assert np.all(ranks>=1)
        assert (ranks<=3).mean()<=(ranks<=10).mean()<=(ranks<=100).mean()


def test_versioned_optional_release_asset_is_complete():
    import hashlib
    import pickle

    from projects.active.bridge.runtime.e2r_query_gate import JOINT_ROUTES

    base = ROOT/"projects/active/bridge/release/runtime/final_bridge_v1"
    manifest = json.loads((base/"e2r_joint_query_v4.manifest.json").read_text())
    artifact = base/manifest["asset_file"]
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert digest == manifest["asset_sha256"]
    assert manifest["status"] == "experimental_opt_in_only"
    with artifact.open("rb") as handle:
        model = pickle.load(handle)
    assert model["schema"] == manifest["model_schema"]
    assert model["route_authorities"] == JOINT_ROUTES
    assert model["frozen_provenance"]["parent_sha256"] == manifest["original_parent_test_sha256"]
    assert model["frozen_provenance"]["split_sha256"] == manifest["outer_split_sha256"]
    assert model["frozen_provenance"]["fixed_test_edges"] == 21505


def test_production_optin_refuses_unvalidated_protocols():
    from types import SimpleNamespace

    import pytest

    from projects.active.bridge.runtime.final_system import FinalBridgeRuntime

    rt = FinalBridgeRuntime.__new__(FinalBridgeRuntime)
    rt.index = SimpleNamespace(protein_index={"P": 0})
    with pytest.raises(ValueError,match="explicit mask_clean2023"):
        rt.rank_reactions({
            "enzyme_id":"P",
            "e2r_gate_profile":"joint-query-v4",
            "top_k":3,
        })
    with pytest.raises(ValueError,match="up to Broad Top1000"):
        rt.rank_reactions({
            "enzyme_id":"P",
            "e2r_gate_profile":"joint-query-v4",
            "mask_clean2023":True,
            "top_k":1001,
        })
    with pytest.raises(ValueError,match="Unknown E2R gate profile"):
        rt.rank_reactions({
            "enzyme_id":"P",
            "e2r_gate_profile":"missing-experiment",
        })
