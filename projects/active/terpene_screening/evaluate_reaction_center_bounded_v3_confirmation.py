from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.evaluate_reaction_center_bounded_v3_selection import paired_summary
PROTOCOL=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3_CONFIRMATION_V1.json'

def sha256_file(path:Path)->str:
 h=hashlib.sha256();
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()

def gate(report:dict[str,object])->dict[str,object]:
 h=report['lt0p3']; a=report['all']; hb=h['baseline']; hc=h['candidate']; ab=a['baseline']; ac=a['candidate']
 checks={
  'lt0p3_mrr_strict_improve':float(hc['mrr'])>float(hb['mrr']),
  'lt0p3_map_strict_improve':float(hc['map'])>float(hb['map']),
  'lt0p3_ndcg10_no_regress':float(hc['ndcg_at_10'])>=float(hb['ndcg_at_10']),
  'lt0p3_macro_auc_no_regress':float(hc['macro_roc_auc'])>=float(hb['macro_roc_auc']),
  'lt0p3_hit10_no_regress':float(hc['hit_at_10'])>=float(hb['hit_at_10']),
  'lt0p3_hit50_no_regress':float(hc['hit_at_50'])>=float(hb['hit_at_50']),
  'lt0p3_median_rank_decrease':float(hc['median_best_positive_rank'])<float(hb['median_best_positive_rank']),
  'all_mrr_no_regress':float(ac['mrr'])>=float(ab['mrr']),
  'all_map_no_regress':float(ac['map'])>=float(ab['map']),
  'all_ndcg10_no_regress':float(ac['ndcg_at_10'])>=float(ab['ndcg_at_10']),
  'all_hit10_no_regress':float(ac['hit_at_10'])>=float(ab['hit_at_10']),
  'all_hit20_no_regress':float(ac['hit_at_20'])>=float(ab['hit_at_20']),
  'all_hit50_no_regress':float(ac['hit_at_50'])>=float(ab['hit_at_50']),
 }
 return {'checks':checks,'pass':all(checks.values())}

def main()->None:
 ap=argparse.ArgumentParser(); ap.add_argument('--cell',required=True); ap.add_argument('--baseline-root',type=Path,required=True); ap.add_argument('--candidate-root',type=Path,required=True); ap.add_argument('--difficulty-root',type=Path,required=True); ap.add_argument('--candidate-summary',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 p=json.loads(PROTOCOL.read_text()); f=p['confirmation_split']; expected=f"clean2023_internal_double_cold_salted_{f['split_salt']}_fold{f['dev_fold']}"
 if args.cell!=expected: raise ValueError(f'Only frozen confirmation cell {expected} is allowed')
 cs=json.loads(args.candidate_summary.resolve().read_text());
 if cs.get('model_type')!='rdkitplus_bounded_identity_hidden_residual' or abs(float(cs.get('max_residual_ratio'))-float(p['selected_max_residual_ratio']))>1e-12: raise ValueError('candidate does not match frozen selected cap')
 if float(cs['identity_audit']['max_abs_diff'])>1e-7 or cs['trainable_parameter_names']!=['aux_to_hidden.weight']: raise ValueError('candidate identity/trainable audit failed')
 bp=args.baseline_root.resolve()/args.cell/'query_metrics.csv'; cp=args.candidate_root.resolve()/args.cell/'query_metrics.csv'; dp=args.difficulty_root.resolve()/args.cell/'reaction_slices.csv'
 report=paired_summary(pd.read_csv(bp,dtype={'query_id':str}),pd.read_csv(cp,dtype={'query_id':str}),pd.read_csv(dp,dtype={'reaction_id':str}),candidate_count=185918); g=gate(report)
 out={'protocol':str(PROTOCOL),'protocol_sha256':sha256_file(PROTOCOL),'cell':args.cell,'selected_max_residual_ratio':p['selected_max_residual_ratio'],'outer_labels_used':False,'model_selection_allowed_after_confirmation':False,**report,**g,'decision':'promote_bounded_residual_v3_mainline' if g['pass'] else 'retain_residual_v2_mainline_no_retuning','input_hashes':{'baseline_query_metrics':sha256_file(bp),'candidate_query_metrics':sha256_file(cp),'difficulty_slices':sha256_file(dp),'candidate_summary':sha256_file(args.candidate_summary.resolve())}}
 args.output.resolve().parent.mkdir(parents=True,exist_ok=True); args.output.resolve().write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
