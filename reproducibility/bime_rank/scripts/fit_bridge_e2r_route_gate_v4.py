"""Joint query-level E2R evidence routing over interpretable biochemical configurations.

The predeclared evidence packages are alternatives under a single query router.
All 5,216 gate-development relations and the fixed 21,505 evaluation subset are
inherited unchanged from v3; no new splitting, label collection, or checkpoint
training. The independent validation queries determine readiness, not training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.e2r_query_gate import (
    FEATURE_NAMES, FrozenE2RSurface, filtered_target_ranks,
    JOINT_ROUTES, predict_joint_route,
)
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from reproducibility.bime_rank.scripts.fit_bridge_e2r_query_gate_v3 import (
    balanced_query_weights, checksum, ranks_from_weights, score_report, target_rows,
)

V3 = ROOT / "results/bridge_e2r_query_gate_v3"
OUT = ROOT / "results/bridge_e2r_route_gate_v4"
SPLIT = ROOT / "results/bridge_gate_split_v2"
FROZEN = ROOT / "results/bridge_e2r_four_group_ablation_v1/edge_metrics.csv.gz"
SOURCE = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
ROUTES = JOINT_ROUTES
CHANNEL_NAMES = ("functional", "structure", "relation")
PHYSICAL_ASSETS = (
    ROOT / "projects/active/bridge/release/manifests/score_evidence_v1/e2r_bundle.json",
)


def checks() -> dict:
    frozen = json.loads((SPLIT / "manifest.json").read_text())
    v3 = json.loads((V3 / "gate_development_manifest.json").read_text())
    if checksum(SOURCE) != frozen["original_sha256"]:
        raise AssertionError("Unmodified parent query benchmark checksum changed")
    if frozen["parts"]["evaluation"]["edges"] != 21505:
        raise AssertionError("Changed established evaluation denominator")
    if frozen["parts"]["gate_learn"]["edges"] != 1706 or frozen["parts"]["gate_validation"]["edges"] != 562:
        raise AssertionError("Modified original parent train/validation query split")
    if v3["combined_development_edges"] != 5216:
        raise AssertionError("Combined development inventory changed")
    if v3["original_target_sha256"] != frozen["original_sha256"]:
        raise AssertionError("Combined gate evidence no longer matches frozen parent")
    source_membership = pd.read_csv(V3 / "gate_development_pairs.csv.gz", dtype=str)
    reserved = pd.read_csv(SPLIT / "evaluation_pairs.csv.gz", dtype=str)
    if set(source_membership.protein_id) & set(reserved.protein_id):
        raise AssertionError("Development/evaluation query leakage")
    if source_membership.groupby("protein_id").partition.nunique().max() != 1:
        raise AssertionError("Development train/validation query leakage")
    return {
        "split_sha256": checksum(SPLIT / "manifest.json"),
        "parent_sha256": frozen["original_sha256"],
        "development_manifest_sha256": checksum(V3 / "gate_development_manifest.json"),
        "development_pairs_sha256": checksum(V3 / "gate_development_pairs.csv.gz"),
        "development_features_sha256": checksum(V3 / "development_query_features.csv.gz"),
        "fixed_source_edges": 23773,
        "fixed_test_edges": 21505,
        "fixed_development_edges": 5216,
        "fixed_train_queries": 2597,
        "fixed_validation_queries": 857,
        "route_set": {k: list(v) for k, v in ROUTES.items()},
    }


def utility(ranks: np.ndarray) -> float:
    ranks = np.asarray(ranks, np.int64)
    return float(np.mean(1.0 / ranks + .20 * (ranks <= 10) + .04 * (ranks <= 100)))


def scores_for(surface: FrozenE2RSurface, item: dict, target: np.ndarray) -> dict:
    return {
        label: ranks_from_weights(surface, item, target, np.asarray(authority))
        for label, authority in ROUTES.items()
    }


def prepare() -> None:
    provenance = checks()
    OUT.mkdir(parents=True, exist_ok=True)
    cached = pd.read_csv(V3 / "development_query_features.csv.gz")
    train = cached[cached.partition.eq("learn")].copy()
    rt = FinalBridgeRuntime(device="cuda")
    surface = FrozenE2RSurface(rt)
    rows = []
    for i, row in enumerate(train.itertuples(index=False), 1):
        item = surface.score(row.query_id)
        targets = target_rows(surface, json.loads(row.target_reactions))
        result = scores_for(surface, item, targets)
        base = utility(result["broad"])
        record = {"query_id": row.query_id, "baseline_utility": base,
                  "any_reachable_top1000": bool(np.any(result["broad"] <= 1000))}
        for route in ROUTES:
            record["gain_" + route] = utility(result[route]) - base
        rows.append(record)
        if i % 400 == 0 or i == len(train):
            print(f"TRAIN_ROUTE_GAIN {i}/{len(train)}", flush=True)
    gains = pd.DataFrame(rows)
    if len(gains) != len(train) or gains.duplicated("query_id").any():
        raise AssertionError("Route label production lost query alignment")
    gains.to_csv(OUT / "train_route_gains.csv.gz", index=False)
    (OUT / "frozen_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("GAIN_SUMMARY",json.dumps({
        name: {
            "mean_delta": float(gains["gain_"+name].mean()),
            "positive_query_fraction": float((gains["gain_"+name]>0).mean()),
        } for name in ROUTES
    }, indent=2),flush=True)


def fit() -> None:
    provenance = checks()
    expected = json.loads((OUT / "frozen_provenance.json").read_text())
    if provenance != expected:
        raise AssertionError("Frozen data or development features mutated after route targets were computed")
    cached = pd.read_csv(V3 / "development_query_features.csv.gz")
    frame = cached.loc[cached.partition.eq("learn")].copy()
    gains = pd.read_csv(OUT / "train_route_gains.csv.gz")
    train = frame.merge(gains, on="query_id", validate="one_to_one", sort=False)
    if len(train)!=len(frame):
        raise AssertionError("Route gains missing training queries")
    X = train[list(FEATURE_NAMES)].to_numpy(np.float64)
    weights, balance = balanced_query_weights(train)
    # If all positive partners are outside Broad Top1000, the chosen route
    # has no capacity to correct them. Downweight ambiguous, uninformative
    # action labels without discarding the query or its provenance.
    reach = train.any_reachable_top1000.to_numpy(bool)
    weights *= np.where(reach, 1., .35)
    weights *= len(weights) / weights.sum()
    models = {}
    diagnostics = {}
    for name, cfg in ROUTES.items():
        if name=="broad":continue
        y = train["gain_"+name].to_numpy(np.float64)
        clf = HistGradientBoostingRegressor(
            max_iter=95, max_leaf_nodes=9, min_samples_leaf=35,
            l2_regularization=16, learning_rate=.05, random_state=26,
        )
        clf.fit(X, y, sample_weight=weights)
        models[name] = clf
        predictions = clf.predict(X)
        diagnostics[name] = {
            "average_training_gain": float(y.mean()),
            "train_positive_gain_ratio": float((y>0).mean()),
            "average_predicted_gain": float(predictions.mean()),
        }
    asset = {
        "schema": "bridge-e2r-joint-query-route-v4",
        "frozen_provenance": provenance,
        "feature_names": FEATURE_NAMES,
        "route_authorities": ROUTES,
        "predictors": models,
        # No inference-positive labels and no tuning on final test. All
        # numerical action coefficients predeclared.
        "baseline_guard": 0.0,
        "training_sample_weights": {
            "novelty_balance": balance,
            "unreachable_query_multiplier": .35,
            "effective_sample_size": float(weights.sum()**2/np.square(weights).sum()),
        },
        "route_fit": diagnostics,
    }
    with (OUT / "gate.pkl").open("wb") as stream:
        pickle.dump(asset, stream)
    (OUT / "train_summary.json").write_text(json.dumps({
        "schema": asset["schema"],
        "train_queries":len(train),
        "train_pairs":3900,
        "usable_non_test_validation_queries":857,
        "reachable_query_fraction":float(reach.mean()),
        "evidence_routes": diagnostics,
        "balance":balance,
        "source_and_split_hashes":provenance,
    },indent=2)+"\n")
    print("ROUTE_FIT", json.dumps(diagnostics,indent=2),flush=True)


select_route = predict_joint_route


def validate() -> None:
    provenance=checks()
    with (OUT / "gate.pkl").open("rb") as stream:
        asset=pickle.load(stream)
    if provenance!=asset["frozen_provenance"]:
        raise AssertionError("Gate fitted to different frozen split")
    frame=pd.read_csv(V3/"development_query_features.csv.gz")
    valid=frame.loc[frame.partition.eq("validation")]
    if len(valid)!=857:
        raise AssertionError("Gate validation queries changed")
    rt=FinalBridgeRuntime(device="cuda");surface=FrozenE2RSurface(rt)
    rows=[];qa=[]
    for i,row in enumerate(valid.itertuples(index=False),1):
        item=surface.score(row.query_id)
        targets=target_rows(surface,json.loads(row.target_reactions))
        choice,auth,preds=select_route(asset,item["features"],item)
        ranks=scores_for(surface,item,targets)
        selected=ranks[choice]
        for j,(nov,degree) in enumerate(zip(json.loads(row.target_novelty),json.loads(row.target_degree_strata))):
            rows.append({
                "protein_id":row.query_id,
                "reaction_id":json.loads(row.target_reactions)[j],
                "novelty":nov,
                "difficulty_stratum":degree,
                "broad_rank":int(ranks["broad"][j]),
                "new_gate_rank":int(selected[j]),
                "old_independent_gate_rank":None,
                "fixed_general_rank":int(ranks["general"][j]),
                "fixed_full_rank":int(ranks["all_evidence"][j]),
            })
        qa.append({"protein_id":row.query_id,"action":choice,
                   "auth_functional":auth[0],"auth_structure":auth[1],"auth_relation":auth[2],
                   "predicted_gain":float(preds[choice]),
                   "positive_count":len(targets)})
        if i%200==0 or i==len(valid):
            print(f"VALIDATE_ROUTE {i}/{len(valid)}",flush=True)
    edges=pd.DataFrame(rows)
    if len(edges)!=1316:
        raise AssertionError("Validation has changed")
    old=pd.read_csv(V3/"validation_edges.csv.gz")
    paired=[
        (row.query_id, reaction_id, novelty, degree)
        for row in valid.itertuples(index=False)
        for reaction_id, novelty, degree in zip(
            json.loads(row.target_reactions),
            json.loads(row.target_novelty),
            json.loads(row.target_degree_strata),
        )
    ]
    if len(old)!=len(paired):
        raise AssertionError("Original frozen validation rows changed")
    expected=pd.DataFrame(paired,columns=["protein_id","reaction_id","novelty","difficulty_stratum"])
    if not (old.query_id.astype(str).to_numpy()==expected.protein_id.to_numpy()).all():
        raise AssertionError("Original frozen validation query order changed")
    if not (old.novelty.to_numpy()==expected.novelty.to_numpy()).all():
        raise AssertionError("Original frozen validation novelty order changed")
    if not (old.difficulty_stratum.to_numpy()==expected.difficulty_stratum.to_numpy()).all():
        raise AssertionError("Original frozen validation degree strata changed")
    expected["old_independent_gate_rank"]=old.gated_rank.to_numpy(np.int64)
    edges=edges.drop(columns=["old_independent_gate_rank"])
    edges=edges.merge(expected,on=["protein_id","reaction_id","novelty","difficulty_stratum"],
                      validate="one_to_one",sort=False)
    if len(edges)!=1316 or edges.old_independent_gate_rank.isna().any():
        raise AssertionError("Old validation baseline not conserved by exact pair join")
    from reproducibility.bime_rank.scripts.analyze_bridge_balanced_monotone_v4 import balanced
    report={
        "schema":"bridge-e2r-joint-route-validation-v4",
        "parent_23773_sha256":provenance["parent_sha256"],
        "source_split_sha256":provenance["split_sha256"],
        "queries":len(qa),
        "edges":len(edges),
        "methods":{
            key:balanced(edges,column)
            for key,column in (
                ("Broad","broad_rank"),
                ("new_joint_route","new_gate_rank"),
                ("old_gate_v3","old_independent_gate_rank"),
                ("fixed_general","fixed_general_rank"),
                ("fixed_full","fixed_full_rank"),
            )
        },
        "query_route_counts":pd.DataFrame(qa).action.value_counts().to_dict(),
    }
    edges.to_csv(OUT/"validation_edge_metrics.csv.gz",index=False)
    pd.DataFrame(qa).to_csv(OUT/"validation_query_actions.csv.gz",index=False)
    (OUT/"validation_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    for name,values in report["methods"].items():
        print("VALIDATION",name,
              "direct10",round(values["direct"]["hit10"]*100,2),
              "balanced10",round(values["balanced"]["hit10"]*100,2),
              "direct100",round(values["direct"]["hit100"]*100,2),
              flush=True)
    print("ROUTE_COUNTS",report["query_route_counts"],flush=True)


def evaluate() -> None:
    """One scoring pass on the previously frozen 21,505 positive edges.

    Model structure, hyperparameters and validation-accepted state are immutable.
    Authoritative Broad/CAGE/historical BRIDGE ranks come from existing caches.
    """
    provenance=checks()
    with (OUT / "gate.pkl").open("rb") as stream:
        asset=pickle.load(stream)
    if asset["frozen_provenance"]!=provenance:
        raise AssertionError("Model trained from different frozen source or split")
    validation=json.loads((OUT/"validation_summary.json").read_text())
    if validation["queries"]!=857 or validation["edges"]!=1316:
        raise AssertionError("Independent validation result changed")
    reference=pd.read_csv(V3/"edge_metrics.csv.gz",
                          dtype={"protein_id":str,"reaction_id":str})
    official=pd.read_csv(SPLIT/"evaluation_pairs.csv.gz",dtype=str)
    if len(reference)!=len(official)!=21505:
        raise AssertionError("Changed heldout evaluation denominator")
    old=reference.set_index(["protein_id","reaction_id"])
    if len(old)!=len(reference) or set(old.index)!=set(zip(official.protein_id,official.reaction_id)):
        raise AssertionError("Legacy E2R ranking cache does not exactly cover frozen evaluation")
    positives=official.groupby("protein_id",sort=True).reaction_id.apply(list)
    if len(positives)!=15751:
        raise AssertionError("Changed heldout E2R protein query count")
    rt=FinalBridgeRuntime(device="cuda")
    surface=FrozenE2RSurface(rt)
    row_out=[];qa=[]
    for i,(protein_id,positive_ids) in enumerate(positives.items(),1):
        item=surface.score(protein_id)
        names,authority,predicted=select_route(asset,item["features"],item)
        ids=target_rows(surface,positive_ids)
        masks={
            "full_rank": authority,
            "minus_functional_rank": authority * np.array([0.,1.,1.]),
            "minus_structure_mechanism_rank": authority * np.array([1.,0.,1.]),
            "minus_relational_memory_rank": authority * np.array([1.,1.,0.]),
        }
        ranks={
            label: ranks_from_weights(surface,item,ids,weights)
            for label,weights in masks.items()
        }
        for j,rid in enumerate(positive_ids):
            rec={"protein_id":protein_id,"reaction_id":rid}
            for col in ["broad_rank","fixed_evidence_previous_rank"]:
                rec[col]=int(old.loc[(protein_id,rid),col])
            rec["old_independent_gate_rank"]=int(old.loc[(protein_id,rid),"full_rank"])
            for key,values in ranks.items():
                rec[key]=int(values[j])
            rec["minus_family_domain_rank"]=int(ranks["full_rank"][j])
            row_out.append(rec)
        qa.append({"protein_id":protein_id,"route":names,"positive_count":len(ids),
                   "functional_authority":float(authority[0]),
                   "structural_authority":float(authority[1]),
                   "relation_authority":float(authority[2]),
                   "predicted_route_gain":float(predicted[names])})
        if i%1000==0 or i==len(positives):
            print(f"ROUTE_FINAL_QUERY {i}/{len(positives)} edges={len(row_out)}",flush=True)
    ranks=pd.DataFrame(row_out)
    if len(ranks)!=21505 or ranks.duplicated(["protein_id","reaction_id"]).any():
        raise AssertionError("Final E2R gate dropped or duplicated frozen rows")
    if set(zip(ranks.protein_id,ranks.reaction_id))!=set(old.index):
        raise AssertionError("Final E2R gate changed evaluation edge identities")
    for column in ("broad_rank","old_independent_gate_rank","fixed_evidence_previous_rank"):
        ground={
            "broad_rank":"broad_rank",
            "old_independent_gate_rank":"full_rank",
            "fixed_evidence_previous_rank":"fixed_evidence_previous_rank",
        }[column]
        for q,r,a in ranks[["protein_id","reaction_id",column]].itertuples(index=False,name=None):
            if int(a)!=int(old.loc[(q,r),ground]):
                raise AssertionError("Historical cached baseline rank changed")
    ranks.to_csv(OUT/"edge_metrics.csv.gz",index=False)
    audit=pd.DataFrame(qa)
    audit.to_csv(OUT/"final_query_actions.csv.gz",index=False)
    report={
        "schema":"bridge-e2r-joint-query-route-v4-fixed-test",
        "source_sha256":provenance["parent_sha256"],
        "split_manifest_sha256":provenance["split_sha256"],
        "no_resplit":True,"no_test_tuned_gate_parameters":True,
        "evaluation_edges":len(ranks),"evaluation_queries":len(audit),
        "original_parent_edges":23773,
        "all_old_cage_broad_expert_scores_unchanged":True,
        "pretrained_expert_weights_changed":False,
        "candidate_universe":len(rt.index.reaction_ids),
        "broad_head_1000_only":True,
        "ranking":{
            c:score_report(ranks[c].to_numpy(np.int64))
            for c in ("broad_rank","fixed_evidence_previous_rank",
                      "old_independent_gate_rank","full_rank",
                      "minus_functional_rank","minus_structure_mechanism_rank",
                      "minus_relational_memory_rank")
        },
        "route_frequency":audit.route.value_counts().to_dict(),
    }
    (OUT/"final_test_summary.json").write_text(json.dumps(report,indent=2)+chr(10))
    print("FROZEN_TEST_GATE",json.dumps({
        k:{"mrr":round(v["mrr"],5),
           "hit10":round(100*v["hit10"],2),
           "hit100":round(100*v["hit100"],2)}
        for k,v in report["ranking"].items()
    },indent=2),flush=True)



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",required=True,choices=["prepare","fit","validate","evaluate"])
    phase=parser.parse_args().phase
    if phase=="prepare":prepare()
    elif phase=="fit":fit()
    elif phase=="validate":validate()
    else:evaluate()


if __name__=="__main__":
    main()
