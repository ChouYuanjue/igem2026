from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_REACTION_CENTER_BOUNDED_RESIDUAL_V3.json'
DEFAULT_ROOT=ROOT/'results/cleanroom_internal_reaction_center_bounded_v3'


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def summarize(frame:pd.DataFrame)->dict[str,object]:
    def mean(c:str)->float: return float(pd.to_numeric(frame[c],errors='coerce').mean())
    return {
        'n_queries':int(len(frame)),
        'candidate_count_unique':sorted(map(int,pd.to_numeric(frame['candidate_count'],errors='raise').unique())),
        'mrr':mean('reciprocal_rank'),'map':mean('average_precision'),
        'macro_roc_auc':float(pd.to_numeric(frame['roc_auc'],errors='coerce').dropna().mean()),
        'ndcg_at_10':mean('ndcg_at_10'),'hit_at_10':mean('hit_at_10'),'hit_at_20':mean('hit_at_20'),'hit_at_50':mean('hit_at_50'),
        'median_best_positive_rank':float(pd.to_numeric(frame['best_positive_rank'],errors='coerce').dropna().median()),
        'mean_best_positive_rank_fraction':mean('best_positive_rank_fraction'),
    }


def deltas(base:dict[str,object],cand:dict[str,object])->dict[str,float]:
    keys=['mrr','map','macro_roc_auc','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50','median_best_positive_rank','mean_best_positive_rank_fraction']
    return {k:float(cand[k])-float(base[k]) for k in keys}


def paired_summary(base:pd.DataFrame,cand:pd.DataFrame,difficulty:pd.DataFrame,*,candidate_count:int)->dict[str,object]:
    base=base[base.direction.eq('reaction_to_enzyme')].copy(); cand=cand[cand.direction.eq('reaction_to_enzyme')].copy()
    if base.query_id.duplicated().any() or cand.query_id.duplicated().any() or set(base.query_id)!=set(cand.query_id): raise ValueError('paired R2E support mismatch')
    diff=difficulty[['reaction_id','reaction_similarity_bucket']].rename(columns={'reaction_id':'query_id'}).copy()
    base=base.merge(diff,on='query_id',validate='one_to_one').sort_values('query_id').reset_index(drop=True)
    cand=cand.merge(diff,on='query_id',validate='one_to_one').sort_values('query_id').reset_index(drop=True)
    if not base.query_id.equals(cand.query_id): raise ValueError('paired query order mismatch')
    for frame in (base,cand):
        if set(map(int,pd.to_numeric(frame.candidate_count,errors='raise').unique()))!={candidate_count}: raise ValueError('candidate universe mismatch')
    ab=summarize(base); ac=summarize(cand); hb=summarize(base[base.reaction_similarity_bucket.eq('lt0p3')]); hc=summarize(cand[cand.reaction_similarity_bucket.eq('lt0p3')])
    if int(hb['n_queries'])<=0: raise ValueError('no lt0p3 queries')
    return {'all':{'baseline':ab,'candidate':ac,'delta':deltas(ab,ac)},'lt0p3':{'baseline':hb,'candidate':hc,'delta':deltas(hb,hc)}}


def candidate_gate(folds:list[dict[str,object]], pooled:dict[str,object])->dict[str,object]:
    hard=pooled['lt0p3']; allq=pooled['all']; hb=hard['baseline']; hc=hard['candidate']; ab=allq['baseline']; ac=allq['candidate']
    hard_both=sum(float(f['lt0p3']['candidate']['mrr'])>float(f['lt0p3']['baseline']['mrr']) and float(f['lt0p3']['candidate']['map'])>float(f['lt0p3']['baseline']['map']) for f in folds)
    hard_mrr_d=[float(f['lt0p3']['delta']['mrr']) for f in folds]; hard_map_d=[float(f['lt0p3']['delta']['map']) for f in folds]
    checks={
      'lt0p3_mrr_strict_improve':float(hc['mrr'])>float(hb['mrr']),
      'lt0p3_map_strict_improve':float(hc['map'])>float(hb['map']),
      'lt0p3_ndcg10_no_regress':float(hc['ndcg_at_10'])>=float(hb['ndcg_at_10']),
      'lt0p3_macro_auc_no_regress':float(hc['macro_roc_auc'])>=float(hb['macro_roc_auc']),
      'lt0p3_hit10_no_regress':float(hc['hit_at_10'])>=float(hb['hit_at_10']),
      'lt0p3_hit50_no_regress':float(hc['hit_at_50'])>=float(hb['hit_at_50']),
      'lt0p3_median_rank_decrease':float(hc['median_best_positive_rank'])<float(hb['median_best_positive_rank']),
      'fold_stability_both_mrr_map_improve_2of3':hard_both>=2,
      'fold_stability_mrr_max_regression_le_0p005':min(hard_mrr_d)>=-0.005,
      'fold_stability_map_max_regression_le_0p005':min(hard_map_d)>=-0.005,
      'all_mrr_no_regress':float(ac['mrr'])>=float(ab['mrr']),
      'all_map_no_regress':float(ac['map'])>=float(ab['map']),
      'all_ndcg10_no_regress':float(ac['ndcg_at_10'])>=float(ab['ndcg_at_10']),
      'all_hit10_no_regress':float(ac['hit_at_10'])>=float(ab['hit_at_10']),
      'all_hit20_no_regress':float(ac['hit_at_20'])>=float(ab['hit_at_20']),
      'all_hit50_no_regress':float(ac['hit_at_50'])>=float(ab['hit_at_50']),
    }
    return {'hard_both_improve_folds':int(hard_both),'hard_mrr_fold_deltas':hard_mrr_d,'hard_map_fold_deltas':hard_map_d,'checks':checks,'pass':all(checks.values())}


def tag(cap:float)->str: return str(cap).replace('.','p')


def main()->None:
    ap=argparse.ArgumentParser(description='Apply the frozen bounded reaction-center V3 internal development selection rule.')
    ap.add_argument('--root',type=Path,default=DEFAULT_ROOT); ap.add_argument('--output',type=Path,default=None); args=ap.parse_args()
    protocol=json.loads(PROTOCOL.read_text()); root=args.root.resolve(); split=protocol['development_split']; folds=list(map(int,split['development_folds'])); candidate_count=185918
    cells=[f"clean2023_internal_double_cold_salted_{split['split_salt']}_fold{fold}" for fold in folds]
    candidates=[]
    for cap in map(float,protocol['candidate_family']['max_residual_ratio_candidates']):
        t=tag(cap); fold_reports=[]; base_frames=[]; cand_frames=[]; diff_frames=[]; hashes=[]
        for fold,cell in zip(folds,cells,strict=True):
            bp=root/'eval_base'/cell/'query_metrics.csv'; cp=root/f'eval_cap_{t}'/cell/'query_metrics.csv'; dp=root/'difficulty'/cell/'reaction_slices.csv'; sp=root/f'cap_{t}'/f'fold{fold}'/'summary.json'
            summary=json.loads(sp.read_text())
            if summary.get('model_type')!='rdkitplus_bounded_identity_hidden_residual' or abs(float(summary.get('max_residual_ratio'))-cap)>1e-12: raise ValueError(f'candidate model mismatch cap={cap} fold={fold}')
            if float(summary['identity_audit']['max_abs_diff'])>1e-7 or summary['trainable_parameter_names']!=['aux_to_hidden.weight']: raise ValueError('identity/trainable audit failed')
            b=pd.read_csv(bp,dtype={'query_id':str}); c=pd.read_csv(cp,dtype={'query_id':str}); d=pd.read_csv(dp,dtype={'reaction_id':str})
            fold_reports.append({'fold':fold,**paired_summary(b,c,d,candidate_count=candidate_count),'identity_audit':summary['identity_audit']})
            rb=b[b.direction.eq('reaction_to_enzyme')].copy(); rc=c[c.direction.eq('reaction_to_enzyme')].copy(); rd=d.copy(); rb['query_id']=rb.query_id.astype(str); rc['query_id']=rc.query_id.astype(str); rd['reaction_id']=rd.reaction_id.astype(str)
            base_frames.append(rb); cand_frames.append(rc); diff_frames.append(rd)
            hashes.append({'fold':fold,'base':sha256_file(bp),'candidate':sha256_file(cp),'difficulty':sha256_file(dp),'model_summary':sha256_file(sp)})
        pooled=paired_summary(pd.concat(base_frames,ignore_index=True),pd.concat(cand_frames,ignore_index=True),pd.concat(diff_frames,ignore_index=True).drop_duplicates('reaction_id'),candidate_count=candidate_count)
        gate=candidate_gate(fold_reports,pooled)
        candidates.append({'max_residual_ratio':cap,'tag':t,'folds':fold_reports,'pooled':pooled,'gate':gate,'input_hashes':hashes})
    passing=[x for x in candidates if x['gate']['pass']]
    passing.sort(key=lambda x:(-float(x['pooled']['lt0p3']['delta']['mrr']),-float(x['pooled']['all']['delta']['mrr']),float(x['max_residual_ratio'])))
    selected=passing[0] if passing else None
    result={
      'protocol':'cleanroom_r2e_reaction_center_bounded_residual_v3','protocol_path':str(PROTOCOL),'protocol_sha256':sha256_file(PROTOCOL),
      'outer_labels_used':False,'revealed_outer_metrics_used_for_selection':False,'candidate_count':len(candidates),'development_cells':cells,
      'candidates':candidates,'passing_candidate_count':len(passing),
      'selected_max_residual_ratio':None if selected is None else selected['max_residual_ratio'],
      'pass':selected is not None,
      'decision':'freeze_pre_registered_confirmation' if selected is not None else 'reject_v3_no_new_caps_on_same_split',
      'confirmation_split_if_pass':protocol['future_confirmation_split'],
      'external_evaluation_authorized':False,
    }
    output=args.output.resolve() if args.output else root/'selection'/'summary.json'; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
