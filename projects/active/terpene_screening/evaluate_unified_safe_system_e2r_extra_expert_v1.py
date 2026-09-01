from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd,torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_unified_safe_system_e2r_full_reaction_v1 import BASE,OUT as BASE_OUT,evaluate_one,load_registered_reaction_feature_library,load_feature_schema
ROOTS={'esmc':ROOT/'results/enzgfm_gate_a_bidirectional_v1/esmc','equalblock':ROOT/'results/enzgfm_gate_a_bidirectional_v1/esmc_enzgfm_equalblock'}
OUT=ROOT/'results/unified_safe_system_v1/e2r_four_expert_portfolio_v1'
def agg(q): return {'n_queries':len(q),'mrr':q.reciprocal_rank.mean(),'map':q.average_precision.mean(),'macro_roc_auc':q.roc_auc.mean(),'ndcg_at_10':q.ndcg_at_10.mean(),'hit_at_1':q.hit_at_1.mean(),'hit_at_3':q.hit_at_3.mean(),'hit_at_5':q.hit_at_5.mean(),'hit_at_10':q.hit_at_10.mean(),'hit_at_20':q.hit_at_20.mean(),'hit_at_50':q.hit_at_50.mean(),'median_best_positive_rank':q.best_positive_rank.median()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--expert',choices=ROOTS,required=True); ap.add_argument('--fold',type=int,choices=[0,1,2],required=True); a=ap.parse_args(); root=ROOTS[a.expert]; f=a.fold; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 bdev=pd.read_csv(BASE/f'fold{f}/dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates(); xdev=pd.read_csv(root/f'fold{f}/dev_pairs.csv',dtype=str)[['protein_id','reaction_id']].drop_duplicates(); assert bdev.equals(xdev)
 bs=json.loads((BASE/f'fold{f}/summary.json').read_text()); xs=json.loads((root/f'fold{f}/summary.json').read_text()); _,br=load_registered_reaction_feature_library(Path(bs['reaction_feature_dir']),load_feature_schema(BASE/f'fold{f}')); _,xr=load_registered_reaction_feature_library(Path(xs['reaction_feature_dir']),load_feature_schema(root/f'fold{f}')); common=sorted(set(br)&set(xr)); assert len(common)==11081
 q,meta=evaluate_one(root,f,common,xdev,device); out=OUT/a.expert/f'fold{f}'; out.mkdir(parents=True,exist_ok=True); q.to_csv(out/'query_metrics.csv',index=False); (out/'result.json').write_text(json.dumps({'expert':a.expert,'fold':f,'metrics':agg(q),'provenance':meta,'external_test_labels_used':False},indent=2)); print(json.dumps({'expert':a.expert,'fold':f,'queries':len(q)},indent=2))
if __name__=='__main__': main()
