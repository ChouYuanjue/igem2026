from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_EVAL=ROOT/'results/cleanroom_internal_full_candidate_rdkitplus_v1'
DEFAULT_DIFFICULTY=ROOT/'results/cleanroom_internal_full_candidate_difficulty_v1'
DEFAULT_OUTPUT=ROOT/'results/cleanroom_internal_shortlist_headroom_v1'
DEFAULT_BUDGETS=(10,50,100,200,500,1000,2000,5000)


def main() -> None:
    parser=argparse.ArgumentParser(description='Summarize the perfect-reranker upper bound implied by frozen coarse positive ranks on internal full-candidate folds.')
    parser.add_argument('--eval-root',type=Path,default=DEFAULT_EVAL)
    parser.add_argument('--difficulty-root',type=Path,default=DEFAULT_DIFFICULTY)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--budgets',default=','.join(map(str,DEFAULT_BUDGETS)))
    args=parser.parse_args()
    budgets=tuple(sorted({int(x) for x in args.budgets.split(',') if x.strip()}))
    if not budgets or budgets[0]<=0: raise ValueError('budgets must be positive')
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    query_rows=[]; summary_rows=[]
    for fold in [0,1,2]:
        cell=f'clean2023_internal_double_cold_fold{fold}'
        ranks=pd.read_csv(args.eval_root/cell/'positive_ranks.csv',dtype={'query_id':str,'positive_id':str})
        ranks=ranks[ranks.direction.eq('reaction_to_enzyme')].copy()
        best=ranks.groupby('query_id',as_index=False).positive_rank.min().rename(columns={'query_id':'reaction_id','positive_rank':'best_positive_rank'})
        slices=pd.read_csv(args.difficulty_root/cell/'reaction_slices.csv',dtype=str).fillna('')
        slices['max_train_drfp_tanimoto']=pd.to_numeric(slices.max_train_drfp_tanimoto,errors='coerce')
        data=best.merge(slices[['reaction_id','max_train_drfp_tanimoto','reaction_similarity_bucket']],on='reaction_id',how='left',validate='one_to_one')
        if data.reaction_similarity_bucket.eq('').any(): raise ValueError(f'missing difficulty rows in {cell}')
        data.insert(0,'fold',fold); data.insert(1,'cell',cell)
        for n in budgets: data[f'hit_at_{n}']=(data.best_positive_rank<=n).astype(int)
        query_rows.append(data)
        for bucket,group in list(data.groupby('reaction_similarity_bucket',sort=True))+[('all',data)]:
            row={'fold':fold,'cell':cell,'reaction_similarity_bucket':bucket,'n_queries':len(group),'median_best_positive_rank':float(group.best_positive_rank.median()),'mean_best_positive_rank':float(group.best_positive_rank.mean())}
            for n in budgets:
                # Any reranker restricted to the top-N shortlist cannot make Hit@10 exceed coarse Hit@N.
                row[f'coarse_hit_at_{n}']=float((group.best_positive_rank<=n).mean())
                row[f'perfect_top{n}_reranker_hit10_upper_bound']=row[f'coarse_hit_at_{n}']
            summary_rows.append(row)
    queries=pd.concat(query_rows,ignore_index=True); summary=pd.DataFrame(summary_rows)
    queries.to_csv(out/'query_headroom.csv',index=False); summary.to_csv(out/'fold_bucket_headroom.csv',index=False)
    aggregate=[]
    for bucket,group in summary.groupby('reaction_similarity_bucket',sort=True):
        row={'reaction_similarity_bucket':bucket,'folds':int(group.fold.nunique()),'mean_n_queries':float(group.n_queries.mean()),'total_queries':int(group.n_queries.sum())}
        for n in budgets:
            values=group[f'coarse_hit_at_{n}']
            row[f'mean_coarse_hit_at_{n}']=float(values.mean()); row[f'min_fold_coarse_hit_at_{n}']=float(values.min()); row[f'max_fold_coarse_hit_at_{n}']=float(values.max())
        aggregate.append(row)
    agg=pd.DataFrame(aggregate); agg.to_csv(out/'aggregate_headroom.csv',index=False)
    manifest={
        'protocol':'internal_full_candidate_shortlist_headroom_v1',
        'outer_labels_used':False,
        'direction':'reaction_to_enzyme',
        'candidate_proteins':185918,
        'budgets':list(budgets),
        'interpretation':'For a reranker restricted to frozen coarse Top-N, coarse Hit@N is the absolute query-level upper bound on post-rerank Hit@10; this diagnostic measures shortlist coverage, not reranker quality.',
    }
    (out/'summary.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(agg.to_string(index=False))

if __name__=='__main__': main()
