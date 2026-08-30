from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = (
    "native_top10_dcg",
    "native_top1_percent_ef",
    "native_top1_sr",
    "native_top3_sr",
    "native_top5_sr",
    "native_top10_sr",
    "common_mrr",
    "common_map",
    "common_macro_roc_auc",
    "common_ndcg_at_10",
)


def load_summary(path: Path, candidate: str, fold: int) -> dict[str, object]:
    x=json.loads(path.read_text(encoding='utf-8'))
    native=x['dev_metrics']['author_native_r2e']; common=x['dev_metrics']['common_ir_r2e']
    return {
        'candidate': candidate, 'fold': int(fold), 'summary_path': str(path.resolve()),
        'native_top10_dcg': native['top10_dcg'], 'native_top1_percent_ef': native['top1_percent_ef'],
        'native_top1_sr': native['top1_sr'], 'native_top3_sr': native['top3_sr'], 'native_top5_sr': native['top5_sr'], 'native_top10_sr': native['top10_sr'],
        'common_mrr': common['mrr'], 'common_map': common['map'], 'common_macro_roc_auc': common['macro_roc_auc'],
        'common_ndcg_at_10': common['ndcg_at_10'],
    }


def select_multifold(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing=(set(METRICS)|{'candidate','fold'})-set(frame.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    counts=frame.groupby('candidate')['fold'].nunique()
    if counts.nunique()!=1:
        raise ValueError(f'candidates have unequal fold counts: {counts.to_dict()}')
    work=frame.copy()
    rank_cols=[]
    for metric in METRICS:
        col=f'{metric}_percentile'
        work[col]=work.groupby('fold')[metric].rank(method='average',pct=True,ascending=True)
        rank_cols.append(col)
    work['joint_percentile']=work[rank_cols].mean(axis=1)
    rows=[]
    for candidate, group in work.groupby('candidate',sort=True):
        rows.append({
            'candidate':candidate,
            'folds':int(group['fold'].nunique()),
            'mean_joint_percentile':float(group['joint_percentile'].mean()),
            'worst_fold_joint_percentile':float(group['joint_percentile'].min()),
            'joint_percentile_std':float(group['joint_percentile'].std(ddof=0)),
            **{f'mean_{m}':float(group[m].mean()) for m in METRICS},
        })
    aggregate=pd.DataFrame(rows).sort_values(
        ['mean_joint_percentile','worst_fold_joint_percentile','joint_percentile_std','candidate'],
        ascending=[False,False,True,True],kind='mergesort').reset_index(drop=True)
    aggregate['selected']=False
    if len(aggregate): aggregate.loc[0,'selected']=True
    return work, aggregate


def main() -> None:
    parser=argparse.ArgumentParser(description='Select cleanroom hyperparameters from multi-fold internal development only.')
    parser.add_argument('--run',action='append',required=True,help='CANDIDATE:FOLD:summary.json')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    for spec in args.run:
        candidate, fold, path=spec.split(':',2)
        rows.append(load_summary(Path(path),candidate,int(fold)))
    scored, aggregate=select_multifold(pd.DataFrame(rows))
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    scored.to_csv(out/'fold_scores.csv',index=False); aggregate.to_csv(out/'candidate_summary.csv',index=False)
    payload={
        'protocol':'target-label-free_multi-fold_joint_metric_selection',
        'target_benchmark_labels_used':False,
        'metrics':list(METRICS),
        'selection_rule':'highest mean fold-wise cross-metric percentile; then highest worst-fold percentile; then lower variability',
        'selected_candidate': None if aggregate.empty else str(aggregate.iloc[0]['candidate']),
    }
    (out/'summary.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(aggregate.to_string(index=False))

if __name__=='__main__': main()
