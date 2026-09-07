from __future__ import annotations
import json
import numpy as np
import pandas as pd
import torch
from projects.active.terpene_screening.evaluate_enzgfm_native_same_support_v1 import score_dual
from projects.active.terpene_screening.tiger_reactzyme_reaction_similarity_native_v1_common import (
    CONTRACT, PREPARED, RESULT_ROOT, TIGER, DualTowerNative, build_label_matrix, load_combined_features, paper_common_metrics, query_metrics, sha256_file,
)

def main():
    out=RESULT_ROOT/'native_evaluation.json'
    if out.exists(): raise SystemExit(f'refusing to overwrite frozen evaluation: {out}')
    summary_path=RESULT_ROOT/'model/summary.json'; model_path=RESULT_ROOT/'model/model.pt'
    if not summary_path.exists() or not model_path.exists(): raise SystemExit('trained frozen model missing')
    train_summary=json.loads(summary_path.read_text())
    if train_summary['target_native_test_pairs_read'] or train_summary['target_native_test_scores_read'] or train_summary['target_native_test_performance_used_for_training_or_selection']:
        raise AssertionError('training provenance is not clean')
    payload=torch.load(model_path,map_location='cpu',weights_only=False)
    if payload['candidate']!='dual_tower' or payload['config']!={'epochs':8,'batch_size':512,'lr':3e-4,'weight_decay':1e-4,'temperature':.05}:
        raise AssertionError('frozen single-recipe checkpoint drift')
    proteins_df=pd.read_csv(PREPARED/'test_proteins.csv',dtype={'protein_idx':int}).rename(columns={'protein_idx':'protein_row'})
    reactions_df=pd.read_csv(PREPARED/'test_reactions.csv',dtype={'reaction_idx':int})
    pairs=pd.read_csv(PREPARED/'test_pairs.csv',dtype={'protein_idx':int,'reaction_idx':int}).rename(columns={'protein_idx':'protein_row'})
    reaction_all=np.load(PREPARED/'reaction_features.npy',mmap_mode='r'); protein_all=load_combined_features(require_overlay=True)
    reactions=np.asarray(reaction_all[reactions_df.reaction_idx.to_numpy(np.int64)],dtype=np.float32)
    proteins=np.asarray(protein_all[proteins_df.protein_row.to_numpy(np.int64)],dtype=np.float32)
    labels=build_label_matrix(pairs,proteins_df,reactions_df)
    if labels.shape!=(14688,386) or int(labels.sum())!=14689:
        raise AssertionError(f'native support drift: {labels.shape} positives={labels.sum()}')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=DualTowerNative().to(device); model.load_state_dict(payload['state_dict']); model.eval()
    scores=score_dual(model,reactions,proteins,device)
    e2r=query_metrics(scores,labels); r2e=query_metrics(scores.T,labels.T)
    common={'e2r':paper_common_metrics(e2r),'r2e':paper_common_metrics(r2e)}
    delta={d:{k:float(common[d][k]-TIGER[d][k]) for k in TIGER[d]} for d in ('e2r','r2e')}
    result={
      'status':'revealed_previously_known_benchmark_descriptive_only_no_retuning_no_promotion',
      'authoritative_external_baseline':'TIGER','contract':str(CONTRACT),'support':{'protein_candidates':14688,'reaction_candidates':386,'positive_pairs':14689},
      'catalyst':{'e2r':e2r,'r2e':r2e},'tiger_common_metrics':TIGER,'catalyst_common_metrics':common,'direct_common_metric_deltas':delta,
      'metric_semantics':{'author_avg_positive_rr':'Directly comparable to the ReactZyme/TIGER paper column labeled MRR; not standard best-positive MRR.','best_mrr':'Catalyst-only standard first-positive reciprocal rank; no direct TIGER delta.','map':'Catalyst-only; no direct TIGER delta.'},
      'benchmark_was_already_revealed_before_protocol':True,'test_performance_used_for_model_selection':False,'post_reveal_retuning_allowed':False,'promotion_evidence_allowed':False,
      'model_sha256':sha256_file(model_path)
    }
    np.save(RESULT_ROOT/'native_scores.npy',scores.astype(np.float32)); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
