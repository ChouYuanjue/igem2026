from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_RESIDUAL_V2_CONFIRMATION_V1.json"


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(frame: pd.DataFrame) -> dict[str, float | int | list[int]]:
    def mean(column: str) -> float:
        return float(pd.to_numeric(frame[column],errors='coerce').mean())
    return {
        'n_queries':int(len(frame)),
        'candidate_count_unique':sorted(map(int,pd.to_numeric(frame['candidate_count'],errors='raise').unique())),
        'mrr':mean('reciprocal_rank'),'map':mean('average_precision'),
        'macro_roc_auc':float(pd.to_numeric(frame['roc_auc'],errors='coerce').dropna().mean()),
        'ndcg_at_10':mean('ndcg_at_10'),'hit_at_10':mean('hit_at_10'),
        'hit_at_20':mean('hit_at_20'),'hit_at_50':mean('hit_at_50'),
        'median_best_positive_rank':float(pd.to_numeric(frame['best_positive_rank'],errors='coerce').dropna().median()),
        'mean_best_positive_rank_fraction':mean('best_positive_rank_fraction'),
    }


def deltas(base: dict[str, object], candidate: dict[str, object]) -> dict[str,float]:
    keys=['mrr','map','macro_roc_auc','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50','median_best_positive_rank','mean_best_positive_rank_fraction']
    return {key:float(candidate[key])-float(base[key]) for key in keys}


def evaluate(base: pd.DataFrame,candidate: pd.DataFrame,difficulty: pd.DataFrame,*,candidate_count:int=185918) -> dict[str,object]:
    base=base[base['direction'].eq('reaction_to_enzyme')].copy()
    candidate=candidate[candidate['direction'].eq('reaction_to_enzyme')].copy()
    if base['query_id'].duplicated().any() or candidate['query_id'].duplicated().any():
        raise ValueError('R2E query IDs must be unique')
    if set(base['query_id']) != set(candidate['query_id']):
        raise ValueError('Baseline/candidate query support differs')
    difficulty=difficulty[['reaction_id','reaction_similarity_bucket','max_train_drfp_tanimoto']].copy()
    base=base.merge(difficulty,left_on='query_id',right_on='reaction_id',validate='one_to_one').sort_values('query_id').reset_index(drop=True)
    candidate=candidate.merge(difficulty,left_on='query_id',right_on='reaction_id',validate='one_to_one').sort_values('query_id').reset_index(drop=True)
    if not base['query_id'].equals(candidate['query_id']):
        raise ValueError('Paired query order differs after merge')
    for name,frame in [('baseline',base),('candidate',candidate)]:
        counts=set(map(int,pd.to_numeric(frame['candidate_count'],errors='raise').unique()))
        if counts != {int(candidate_count)}:
            raise ValueError(f'{name} candidate universe mismatch: {counts}')
    all_base=summarize(base); all_candidate=summarize(candidate)
    hard_base=summarize(base[base['reaction_similarity_bucket'].eq('lt0p3')])
    hard_candidate=summarize(candidate[candidate['reaction_similarity_bucket'].eq('lt0p3')])
    if int(hard_base['n_queries']) <= 0:
        raise ValueError('Fresh confirmation has no reaction_similarity_lt0p3 queries')
    checks={
        'lt0p3_mrr_strict_improve':float(hard_candidate['mrr'])>float(hard_base['mrr']),
        'lt0p3_map_strict_improve':float(hard_candidate['map'])>float(hard_base['map']),
        'lt0p3_ndcg10_no_regress':float(hard_candidate['ndcg_at_10'])>=float(hard_base['ndcg_at_10']),
        'lt0p3_macro_auc_no_regress':float(hard_candidate['macro_roc_auc'])>=float(hard_base['macro_roc_auc']),
        'lt0p3_hit10_no_regress':float(hard_candidate['hit_at_10'])>=float(hard_base['hit_at_10']),
        'lt0p3_hit50_no_regress':float(hard_candidate['hit_at_50'])>=float(hard_base['hit_at_50']),
        'lt0p3_median_rank_decrease':float(hard_candidate['median_best_positive_rank'])<float(hard_base['median_best_positive_rank']),
        'all_mrr_guard':float(all_candidate['mrr'])>=float(all_base['mrr'])-0.002,
        'all_map_guard':float(all_candidate['map'])>=float(all_base['map'])-0.002,
        'all_hit10_guard':float(all_candidate['hit_at_10'])>=float(all_base['hit_at_10'])-0.005,
        'all_hit20_guard':float(all_candidate['hit_at_20'])>=float(all_base['hit_at_20'])-0.005,
        'all_hit50_guard':float(all_candidate['hit_at_50'])>=float(all_base['hit_at_50'])-0.005,
    }
    return {
        'all':{'baseline':all_base,'candidate':all_candidate,'delta':deltas(all_base,all_candidate)},
        'lt0p3':{'baseline':hard_base,'candidate':hard_candidate,'delta':deltas(hard_base,hard_candidate)},
        'checks':checks,'pass':all(checks.values()),
        'decision':'passed_fresh_internal_salted_confirmation' if all(checks.values()) else 'rejected_fresh_internal_salted_confirmation_no_retuning',
    }


def main() -> None:
    parser=argparse.ArgumentParser(description='Apply the frozen reaction-center residual-v2 fresh confirmation gate.')
    parser.add_argument('--cell',required=True)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--candidate-root',type=Path,required=True)
    parser.add_argument('--difficulty-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    frozen=protocol['confirmation_split']
    expected=f"clean2023_internal_double_cold_salted_{frozen['split_salt']}_fold{frozen['dev_fold']}"
    if args.cell != expected:
        raise ValueError(f'Only frozen confirmation cell {expected} is allowed')
    bp=args.baseline_root.resolve()/args.cell/'query_metrics.csv'
    cp=args.candidate_root.resolve()/args.cell/'query_metrics.csv'
    dp=args.difficulty_root.resolve()/args.cell/'reaction_slices.csv'
    result=evaluate(pd.read_csv(bp,dtype={'query_id':str}),pd.read_csv(cp,dtype={'query_id':str}),pd.read_csv(dp,dtype={'reaction_id':str}),candidate_count=int(protocol['evaluation']['candidate_proteins']))
    result.update({'protocol':str(PROTOCOL),'protocol_sha256':sha256_file(PROTOCOL),'cell':args.cell,
                   'baseline_query_metrics_sha256':sha256_file(bp),'candidate_query_metrics_sha256':sha256_file(cp),'difficulty_slices_sha256':sha256_file(dp),
                   'outer_labels_used':False,'model_selection_allowed_after_confirmation':False})
    args.output.resolve().parent.mkdir(parents=True,exist_ok=True)
    args.output.resolve().write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
