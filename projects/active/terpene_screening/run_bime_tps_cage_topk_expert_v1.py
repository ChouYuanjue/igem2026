from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb

BASE = ROOT / "results/terpene_pure_cage_full_support_v1/catalyst_same_support_query_metrics.csv"
CAGE = ROOT / "results/terpene_pure_cage_full_support_v1/pretrain/pure_cage_native_full_pairs_epoch_19.csv.gz"
OUT = ROOT / "results/bime_rank_unified_v1/tps_cage_top20_expert_v1"
FOLDS = (0, 1, 2, 3, 4)
TOPK = 20
PARAMS = {
    "objective": "rank:ndcg",
    "eval_metric": "ndcg@10",
    "tree_method": "hist",
    "max_depth": 2,
    "eta": 0.05,
    "min_child_weight": 5.0,
    "lambda": 5.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": 10,
    "seed": 20260905,
    "verbosity": 0,
}
ROUNDS = 60
BASE_FEATURES = ["baseline_reciprocal_rank", "baseline_rank_fraction"]
CAGE_FEATURES = [
    "cage_logit", "cage_query_zscore", "cage_global_log_rank_fraction",
    "cage_global_reciprocal_rank", "cage_global_top5", "cage_global_top10",
    "cage_global_top20", "cage_top20_rank_fraction", "cage_top20_reciprocal_rank",
    "baseline_cage_rank_gap", "cage_score_spread", "cage_unique_score_fraction",
]
ALL_FEATURES = BASE_FEATURES + CAGE_FEATURES


def _metrics(order: list[str], positives: set[str]) -> dict[str, float]:
    ranks = [i + 1 for i, p in enumerate(order) if p in positives]
    best = min(ranks) if ranks else None
    rr = 0.0 if best is None else 1.0 / best
    ap = 0.0
    if ranks:
        ap = float(np.mean([(i + 1) / rank for i, rank in enumerate(sorted(ranks))]))
    dcg = sum((1.0 / math.log2(rank + 1)) for rank in ranks if rank <= 10)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(positives), 10)))
    return {
        "reciprocal_rank": rr,
        "average_precision_at20": ap,
        "ndcg_at_10": dcg / ideal if ideal else 0.0,
        "hit_at_3": float(best is not None and best <= 3),
        "hit_at_5": float(best is not None and best <= 5),
        "hit_at_10": float(best is not None and best <= 10),
        "hit_at_20": float(best is not None and best <= 20),
        "best_positive_rank": float(best) if best is not None else 0.0,
    }


def prepare() -> None:
    q = pd.read_csv(BASE, dtype=str).fillna("")
    q = q[q["budget"].astype(int).eq(TOPK)].copy()
    if len(q) != 462 or q["reaction_id"].nunique() != 462:
        raise RuntimeError("unexpected Catalyst TPS Top20 query set")
    cage = pd.read_csv(CAGE, dtype=str).fillna("")
    cage["pred_logit"] = cage["pred_logit"].astype(float)
    cage["label"] = cage["label"].astype(int)
    score = {(r, p): float(s) for r, p, s in cage[["reaction_id", "uniprot_id", "pred_logit"]].itertuples(index=False)}
    positives = cage[cage.label.eq(1)].groupby("reaction_id")["uniprot_id"].apply(lambda x: set(map(str, x))).to_dict()

    global_rank: dict[tuple[str, str], int] = {}
    stats: dict[str, tuple[float, float, float, float, int]] = {}
    for rid, g in cage.groupby("reaction_id", sort=False):
        vals = g["pred_logit"].to_numpy(float); ids = g["uniprot_id"].astype(str).to_numpy(object)
        order = np.lexsort((ids, -vals))
        for rank, loc in enumerate(order, 1):
            global_rank[(str(rid), str(ids[loc]))] = rank
        stats[str(rid)] = (float(vals.mean()), max(float(vals.std()), 1e-6), float(vals.max() - vals.min()), float(len(np.unique(vals)) / len(vals)), int(len(vals)))

    rows=[]; qmeta=[]
    for item in q.itertuples(index=False):
        rid=str(item.reaction_id); fold=int(item.target_fold); ids=[p for p in str(item.ranking).split(";") if p]
        if len(ids) != TOPK: raise RuntimeError((rid, len(ids)))
        if any((rid,p) not in score for p in ids): raise RuntimeError(f"CAGE coverage drift {rid}")
        vals=np.asarray([score[(rid,p)] for p in ids],dtype=np.float64)
        top_order=np.lexsort((np.asarray(ids,dtype=object),-vals)); top_rank=np.empty(TOPK,dtype=np.int32); top_rank[top_order]=np.arange(1,TOPK+1)
        mean,std,spread,uniq,cage_n=stats[rid]
        pos=positives.get(rid,set())
        base_m=_metrics(ids,pos)
        qmeta.append({"reaction_id":rid,"target_fold":fold,**{f"baseline_{k}":v for k,v in base_m.items()}})
        for br,p in enumerate(ids,1):
            gr=global_rank[(rid,p)]; cr=int(top_rank[br-1]); logit=score[(rid,p)]
            rows.append({
                "reaction_id":rid,"target_fold":fold,"protein_id":p,"label":int(p in pos),"baseline_rank":br,
                "baseline_reciprocal_rank":1.0/br,"baseline_rank_fraction":br/TOPK,
                "cage_logit":logit,"cage_query_zscore":(logit-mean)/std,
                "cage_global_log_rank_fraction":math.log1p(gr)/math.log1p(cage_n),
                "cage_global_reciprocal_rank":1.0/gr,"cage_global_top5":float(gr<=5),"cage_global_top10":float(gr<=10),"cage_global_top20":float(gr<=20),
                "cage_top20_rank_fraction":cr/TOPK,"cage_top20_reciprocal_rank":1.0/cr,
                "baseline_cage_rank_gap":abs(br-cr)/TOPK,"cage_score_spread":spread,"cage_unique_score_fraction":uniq,
            })
    frame=pd.DataFrame(rows); meta=pd.DataFrame(qmeta)
    OUT.mkdir(parents=True,exist_ok=True); frame.to_csv(OUT/"pairs.csv",index=False); meta.to_csv(OUT/"query_baseline.csv",index=False)
    summary={"queries":len(meta),"pairs":len(frame),"folds":meta.target_fold.value_counts().sort_index().to_dict(),"topk":TOPK,"cage_pair_coverage":1.0,"external_metrics_used":False,"features":ALL_FEATURES}
    (OUT/"prepare_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


def _train(train: pd.DataFrame, features: list[str], seed: int) -> xgb.Booster:
    train=train.sort_values(["reaction_id","baseline_rank"],kind="stable")
    groups=train.groupby("reaction_id",sort=False).size().tolist()
    dm=xgb.DMatrix(train[features].to_numpy(np.float32),label=train.label.to_numpy(np.float32)); dm.set_group(groups)
    params=dict(PARAMS); params["seed"]=seed
    return xgb.train(params,dm,num_boost_round=ROUNDS)


def _eval(model: xgb.Booster, hold: pd.DataFrame, features: list[str], method: str) -> pd.DataFrame:
    rec=[]
    for rid,g in hold.groupby("reaction_id",sort=False):
        g=g.sort_values("baseline_rank",kind="stable").copy(); pred=model.predict(xgb.DMatrix(g[features].to_numpy(np.float32)))
        order=np.lexsort((g.protein_id.astype(str).to_numpy(object),g.baseline_rank.to_numpy(int),-pred))
        ranked=g.iloc[order].protein_id.astype(str).tolist(); pos=set(g.loc[g.label.eq(1),"protein_id"].astype(str))
        rec.append({"reaction_id":rid,"target_fold":int(g.target_fold.iloc[0]),"method":method,**_metrics(ranked,pos)})
    return pd.DataFrame(rec)


def crossfit() -> None:
    pairs=pd.read_csv(OUT/"pairs.csv",dtype={"reaction_id":str,"protein_id":str}); frames=[]
    for holdout in FOLDS:
        train=pairs[~pairs.target_fold.eq(holdout)].copy(); hold=pairs[pairs.target_fold.eq(holdout)].copy()
        for method,features in [("baseline_ranker",BASE_FEATURES),("cage_top20_expert",ALL_FEATURES)]:
            m=_train(train,features,20260905+holdout+(0 if method=="baseline_ranker" else 100)); frames.append(_eval(m,hold,features,method))
        print(f"fold {holdout} done",flush=True)
    oof=pd.concat(frames,ignore_index=True); oof.to_csv(OUT/"oof_query_metrics.csv",index=False)
    raw=pd.read_csv(OUT/"query_baseline.csv")
    def summarize(df,prefix=""):
        return {"mrr":float(df[f"{prefix}reciprocal_rank"].mean()),"map_at20":float(df[f"{prefix}average_precision_at20"].mean()),"ndcg_at_10":float(df[f"{prefix}ndcg_at_10"].mean()),"hit_at_3":float(df[f"{prefix}hit_at_3"].mean()),"hit_at_5":float(df[f"{prefix}hit_at_5"].mean()),"hit_at_10":float(df[f"{prefix}hit_at_10"].mean()),"hit_at_20":float(df[f"{prefix}hit_at_20"].mean())}
    rawm=summarize(raw,"baseline_"); basem=summarize(oof[oof.method.eq("baseline_ranker")]); cagem=summarize(oof[oof.method.eq("cage_top20_expert")])
    dc={k:cagem[k]-basem[k] for k in cagem}; dr={k:cagem[k]-rawm[k] for k in cagem}
    fold_delta={}
    for f in FOLDS:
        b=summarize(oof[(oof.method.eq("baseline_ranker"))&oof.target_fold.eq(f)]); c=summarize(oof[(oof.method.eq("cage_top20_expert"))&oof.target_fold.eq(f)])
        fold_delta[str(f)]={k:c[k]-b[k] for k in c}
    admitted=bool(dc["mrr"]>=0.002 and dc["ndcg_at_10"]>=0 and dc["hit_at_10"]>=0.005 and dr["mrr"]>=-0.002 and dr["hit_at_10"]>=-0.005 and all(x["mrr"]>=-0.01 for x in fold_delta.values()))
    payload={"status":"admitted_tps_conditional_expert" if admitted else "not_admitted","scope":"TPS-only Top20 conditional reranker; candidate set fixed; official EnzymeCAGE pretrained scores only when available","raw_catalyst":rawm,"same_capacity_baseline_ranker":basem,"cage_top20_expert":cagem,"delta_vs_same_capacity":dc,"delta_vs_raw":dr,"fold_delta_vs_same_capacity":fold_delta,"admitted":admitted,"external_metrics_used":False,"selection":"single predeclared fixed-capacity five-fold OOF comparison; no parameter sweep"}
    (OUT/"development_result.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","crossfit"]); a=ap.parse_args()
    if a.stage=="prepare": prepare()
    else: crossfit()

if __name__=="__main__": main()
