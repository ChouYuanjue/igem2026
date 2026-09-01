from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from projects.active.terpene_screening.broad_rhea_metrics import summarize_query_metrics
from projects.active.terpene_screening.fair_benchmark import DEFAULT_BUDGETS, DEFAULT_TOP_PERCENTS

def load_r2e(path: Path) -> pd.DataFrame:
    x=pd.read_csv(path,dtype={'query_id':str}).fillna(np.nan)
    return x[x['direction']=='reaction_to_enzyme'].copy()

def summarize(x: pd.DataFrame) -> dict[str, object]:
    return summarize_query_metrics(x,budgets=DEFAULT_BUDGETS,top_percents=DEFAULT_TOP_PERCENTS)

def evaluate(baseline_csv: Path,candidate_csv: Path,difficulty_csv: Path,threshold: float) -> tuple[pd.DataFrame,dict[str,object]]:
    b=load_r2e(baseline_csv); c=load_r2e(candidate_csv)
    d=pd.read_csv(difficulty_csv,dtype={'reaction_id':str}).rename(columns={'reaction_id':'query_id'})
    x=b.merge(c,on=['direction','query_id'],suffixes=('_baseline','_candidate'),validate='one_to_one').merge(d[['query_id','max_train_drfp_tanimoto']],on='query_id',validate='one_to_one')
    use=pd.to_numeric(x['max_train_drfp_tanimoto'],errors='raise') < float(threshold)
    metric_names=[col[:-9] for col in x.columns if col.endswith('_baseline') and col[:-9]+'_candidate' in x.columns]
    routed=pd.DataFrame({'direction':'reaction_to_enzyme','query_id':x['query_id'].astype(str)})
    for m in metric_names:
        routed[m]=np.where(use,x[m+'_candidate'].to_numpy(),x[m+'_baseline'].to_numpy())
    routed['router_use_candidate']=use.astype(int).to_numpy()
    routed['max_train_drfp_tanimoto']=pd.to_numeric(x['max_train_drfp_tanimoto']).to_numpy()
    bs=summarize(b); rs=summarize(routed.drop(columns=['router_use_candidate','max_train_drfp_tanimoto']))
    keys=('mrr','map','macro_roc_auc','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50')
    delta={k:float(rs[k])-float(bs[k]) for k in keys}
    checks={
      'mrr_strict':delta['mrr']>0,
      'map_strict':delta['map']>0,
      'hit_at_10_strict':delta['hit_at_10']>0,
      'ndcg_at_10_no_regress':delta['ndcg_at_10']>=0,
      'hit_at_20_no_regress':delta['hit_at_20']>=0,
      'hit_at_50_no_regress':delta['hit_at_50']>=0,
      'macro_roc_auc_no_regress':delta['macro_roc_auc']>=0,
    }
    result={'threshold':float(threshold),'query_count':int(len(routed)),'candidate_fraction':float(use.mean()),'baseline':bs,'routed':rs,'delta':delta,'checks':checks,'pass':bool(all(checks.values()))}
    return routed,result

def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-query-metrics',type=Path,required=True); ap.add_argument('--candidate-query-metrics',type=Path,required=True); ap.add_argument('--difficulty',type=Path,required=True)
    ap.add_argument('--threshold',type=float,default=0.9); ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    routed,result=evaluate(a.baseline_query_metrics,a.candidate_query_metrics,a.difficulty,a.threshold)
    routed.to_csv(a.output_dir/'query_metrics.csv',index=False); (a.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
