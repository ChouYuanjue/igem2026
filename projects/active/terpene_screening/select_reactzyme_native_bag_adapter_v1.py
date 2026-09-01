from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
DEFAULT=ROOT/'results/reactzyme_native_bag_adapter_v1'

def aggregate(root:Path):
 frames=[]; fold_eval=[]
 for f in range(3):
  p=root/f'fold{f}/dev_r2e_query_metrics.csv'; e=root/f'fold{f}/evaluation.json'
  if not p.exists() or not e.exists(): raise FileNotFoundError(f'missing fold {f} evaluation')
  q=pd.read_csv(p); q['fold']=f; frames.append(q); fold_eval.append(json.load(open(e)))
 q=pd.concat(frames,ignore_index=True)
 def side(prefix):
  return {'mrr':float(q[f'{prefix}_mrr'].mean()),'map':float(q[f'{prefix}_ap'].mean()),'ndcg_at_10':float(q[f'{prefix}_ndcg_at_10'].mean()),'hit_at_10':float(q[f'{prefix}_hit_at_10'].mean()),'hit_at_20':float(q[f'{prefix}_hit_at_20'].mean()),'hit_at_50':float(q[f'{prefix}_hit_at_50'].mean()),'median_best_positive_rank':float(q[f'{prefix}_best_positive_rank'].median())}
 t=side('teacher'); a=side('adapter')
 ret={k:a[k]/t[k] if t[k] else None for k in ['mrr','map','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']}
 fold_mrr=[float(x['retention']['mrr']) for x in fold_eval]
 checks={
  'teacher_cosine_mean_ge_0p75':float(q.teacher_cosine.mean())>=0.75,
  'mrr_retention_ge_0p75':ret['mrr']>=0.75,
  'map_retention_ge_0p75':ret['map']>=0.75,
  'hit10_retention_ge_0p75':ret['hit_at_10']>=0.75,
  'hit50_retention_ge_0p75':ret['hit_at_50']>=0.75,
  'at_least_2_folds_mrr_retention_ge_0p65':sum(x>=0.65 for x in fold_mrr)>=2,
 }
 return {'status':'pass' if all(checks.values()) else 'reject','n_queries':int(len(q)),'teacher_cosine_mean':float(q.teacher_cosine.mean()),'teacher_cosine_median':float(q.teacher_cosine.median()),'teacher':t,'adapter':a,'retention':ret,'fold_mrr_retention':fold_mrr,'checks':checks,'external_reactzyme_metrics_used':False,'retuning_v1_allowed':False,'next':'freeze_fresh_salted_confirmation_before_confirmation_performance' if all(checks.values()) else 'reject_v1_without_retuning_under_v1'}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=DEFAULT); a=ap.parse_args(); out=aggregate(a.root); (a.root/'selection.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
