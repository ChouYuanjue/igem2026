from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.evaluate_fast_r2e_similarity_router_v1 import evaluate
ROOT=Path(__file__).resolve().parents[4]
def test_protocol_freezes_threshold_and_confirmation() -> None:
    p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_FAST_R2E_SIMILARITY_ROUTER_V1.json').read_text())
    assert p['selection_boundary']['threshold']==0.9
    assert p['selection_boundary']['labels_in_router'] is False
    assert p['untouched_confirmation']['dev_fold']==6
    assert p['untouched_confirmation']['no_threshold_search_on_confirmation'] is True

def test_router_selects_candidate_only_below_threshold(tmp_path: Path) -> None:
    def row(query_id: str, rank: int, roc_auc: float) -> dict[str, object]:
        metrics=evaluate_full_candidate_ranks(
            __import__("numpy").asarray([rank],dtype=__import__("numpy").int64),
            100,
        )
        metrics["roc_auc"]=roc_auc
        return {"direction":"reaction_to_enzyme","query_id":query_id,**metrics}
    base=pd.DataFrame([row("r1",20,.5),row("r2",1,.8)])
    cand=pd.DataFrame([row("r1",2,.7),row("r2",100,.1)])
    diff=pd.DataFrame([{"reaction_id":"r1","max_train_drfp_tanimoto":.2},{"reaction_id":"r2","max_train_drfp_tanimoto":.95}])
    bp,cp,dp=tmp_path/"b.csv",tmp_path/"c.csv",tmp_path/"d.csv"
    base.to_csv(bp,index=False); cand.to_csv(cp,index=False); diff.to_csv(dp,index=False)
    routed,result=evaluate(bp,cp,dp,.9)
    by_id=routed.set_index("query_id")
    assert by_id.loc["r1","router_use_candidate"]==1
    assert by_id.loc["r2","router_use_candidate"]==0
    assert result["pass"] is True
