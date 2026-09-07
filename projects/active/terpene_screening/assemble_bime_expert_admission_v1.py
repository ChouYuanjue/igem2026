import json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def J(p): return json.loads((ROOT / p).read_text())
def subset(d, keys): return {k:d[k] for k in keys}
MET=['mrr','map','ndcg_at_10','hit_at_10','hit_at_20','hit_at_50']
r2dev=J('results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json')
r2ext=J('results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/summary.json')
e2sel=J('results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selection_result.json')
e2ext=J('results/clipzyme_native_extension_v1/e2r_strict650_clipzyme_v4_fair_v1/summary.json')
r2ret=J('results/bime_rank_unified_v1/r2e_seed_context_retention_v1/summary.json')
e2ret=J('results/bime_rank_unified_v1/e2r_seed_context_retention_v1/summary.json')
multi=J('results/bime_rank_unified_v1/multiseed_scaling_v1/summary.json')
hdev=J('results/bime_rank_unified_v1/r2e_homology_context_v1/development_result.json')
hret=J('results/bime_rank_unified_v1/r2e_homology_context_retention_v1/summary.json')
recdev=J('results/bime_rank_unified_v1/r2e_reciprocal_consistency_v1/development_result.json')
recext=J('results/bime_rank_unified_v1/r2e_reciprocal_external_confirmation_v1/summary.json')
cage=J('results/bime_rank_unified_v1/tps_cage_top20_expert_v1/development_result.json')
cageprep=J('results/bime_rank_unified_v1/tps_cage_top20_expert_v1/prepare_summary.json')
e2sm=J('results/bime_rank_unified_v1/e2r_runtime_smoke_v2/summary.json')
r2sm=J('results/bime_rank_unified_v1/r2e_runtime_smoke_v2/summary.json')
mdiff=J('results/bime_rank_unified_v1/promotion_audit_v1/manifest_diff.json')
equiv=J('results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_manifest_equivalence.json')
psmoke=J('results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_smoke/summary.json')
gate=J('results/bime_rank_unified_v1/promotion_audit_v1/promotion_gate.json')
prod1=J('projects/active/terpene_screening/BIME_RANK_PRODUCTION_V1_RESULT.json')
prod2=J('projects/active/terpene_screening/BIME_RANK_PRODUCTION_V2_RESULT.json')
qman=J('results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1/manifest.json')
rman=J('results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/manifest.json')
# helper transforms
r2_old={m:r2ext['paired_bootstrap_50000'][m]['old_bime'] for m in MET}
r2_clip={m:r2ext['paired_bootstrap_50000'][m]['clipzyme'] for m in MET}
s=e2sel['selected_summary']
e2base={k:s['incumbent_v3_'+src] for k,src in [('mrr','mrr'),('map','map'),('auc','auc'),('ndcg10','ndcg10'),('hit10','hit10'),('hit20','hit20'),('hit50','hit50')]}
e2cand={k:s['candidate_'+src] for k,src in [('mrr','mrr'),('map','map'),('auc','auc'),('ndcg10','ndcg10'),('hit10','hit10'),('hit20','hit20'),('hit50','hit50')]}
e2delta={k:s['delta_vs_v3_'+src] for k,src in [('mrr','mrr'),('map','map'),('auc','auc'),('ndcg10','ndcg10'),('hit10','hit10'),('hit20','hit20'),('hit50','hit50')]}
ctx1r=multi['r2e']['metrics']['context']; ctx1e=multi['e2r']['metrics']['context']
payload={
 'schema':'bime_rank_expert_admission_v1','method':'BiME-Rank','status':'promoted_default_production',
 'policy':{
  'selection':'expert admission is decided on internal clean development data before external confirmation',
  'external_retention':'frozen external/temporal-double-cold results may veto an internally admitted expert but are never used for retuning',
  'missing_expert':'absence is represented as unavailable; it is never converted into a low score',
  'fallback':'availability-aware experts must reproduce the incumbent route exactly when unavailable',
  'candidate_manifest':'configs/production_routes/bime_rank_candidate_v1.yaml','default_production_manifest':'configs/production_routes/terpene_v1.yaml'},
 'experts':{},
 'candidate_runtime':{
  'r2e':'Availability-aware CLIPZyme structural signal on the frozen R2E BiME-Rank prefix; exact base LambdaRank fallback when structure is unavailable.',
  'e2r':'Availability-aware CLIPZyme structural signal on anchored E2R BiME-Rank; exact non-structural anchored fallback when structure is unavailable.',
  'context':'Bidirectional known-positive seed-context is production-admitted; homology context is excluded from current BiME ranking after failed frozen temporal retention.',
  'excluded':['enzymecage_top20_structure','homology_context','reciprocal_consistency']},
}
payload['experts']['r2e_clipzyme_structure']={
 'direction':'reaction_to_enzyme','status':'promoted_production','role':'availability_aware_structural_retrieval_signal',
 'internal_evidence':'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/development_result.json',
 'internal_baseline':r2dev['old'],'internal_candidate':r2dev['new'],'internal_delta':r2dev['delta'],
 'strict_external_evidence':'results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/summary.json',
 'strict_external_metrics':subset(r2ext['metrics'],MET),
 'strict_external_vs_old_bime':r2_old,'strict_external_vs_clipzyme':r2_clip,
 'ranker':'results/bime_rank_unified_v1/r2e_clipzyme_expert_v1/selected/ranker.json','ranker_sha256':r2ext['structural_ranker_sha256'],
 'runtime':'projects/active/terpene_screening/bime_rank_r2e_runtime.py','fallback':'exact frozen R2E LambdaRank when reaction lacks CLIPZyme support',
 'runtime_smoke':'results/bime_rank_unified_v1/r2e_runtime_smoke_v2/summary.json'}
payload['experts']['e2r_clipzyme_structure']={
 'direction':'enzyme_to_reaction','status':'promoted_production','role':'availability_aware_structural_retrieval_signal',
 'internal_evidence':'results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selection_result.json',
 'internal_baseline_v3':e2base,'internal_candidate_v4':e2cand,'internal_delta_vs_v3':e2delta,
 'strict_external_evidence':'results/clipzyme_native_extension_v1/e2r_strict650_clipzyme_v4_fair_v1/summary.json',
 'strict_external_models':e2ext['models'],
 'strict_external_v4_minus_clipzyme':e2ext['paired_bootstrap_50000']['v4_minus_clipzyme'],
 'strict_external_v4_minus_v3':e2ext['paired_bootstrap_50000']['v4_minus_current_v3'],
 'ranker':'results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/selected/ranker.json',
 'ranker_sha256':e2ext['fairness']['v4_ranker_sha256'],
 'runtime':'projects/active/terpene_screening/bime_rank_e2r_runtime.py',
 'fallback':'exact Anchored LambdaMART V3 when protein lacks CLIPZyme representation',
 'runtime_smoke':'results/bime_rank_unified_v1/e2r_runtime_smoke_v2/summary.json',
 'protein_query_asset':'results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1',
 'protein_query_manifest_sha256':hashlib.sha256((ROOT / 'results/bime_rank_unified_v1/clipzyme_e2r_query_asset_v1/manifest.json').read_bytes()).hexdigest(),
 'reaction_asset':'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1',
 'reaction_manifest_sha256':hashlib.sha256((ROOT / 'results/clipzyme_native_extension_v1/full_hplus_candidate_reactions/clipzyme_embeddings_gpu_v1/manifest.json').read_bytes()).hexdigest(),
 'runtime_retention':'results/bime_rank_unified_v1/promotion_audit_v1/e2r_runtime_retention.json','runtime_retention_status':J('results/bime_rank_unified_v1/promotion_audit_v1/e2r_runtime_retention.json')['status'],
 'query_support_count':qman['total_supported_count'],'reaction_support_count':rman['clipzyme_supported_count']}
payload['experts']['seed_context']={
 'directions':['reaction_to_enzyme','enzyme_to_reaction'],'status':'promoted_production_conditional_context','role':'known-positive conditional second-stage reranking on the same frozen zero-shot BiME-Rank order',
 'invariant':'zero-shot BiME order is formed first; context only activates when a valid seed exists; seed IDs are masked; tail remains the zero-shot BiME order',
 'r2e':{'internal_evidence':'results/bime_rank_unified_v1/r2e_seed_context_v1/development_result.json','strict_temporal_retention':'results/bime_rank_unified_v1/r2e_seed_context_retention_v1/summary.json','ranker':'results/bime_rank_unified_v1/r2e_seed_context_v1/selected/ranker.json','ranker_sha256':r2ret['seed_ranker_sha256'],'retention_queries':r2ret['eligible_queries'],'retention_trials':r2ret['trials'],'metrics':r2ret['metrics']['context'],'delta_vs_previous_production_fewshot':r2ret['delta_context_vs_legacy']},
 'e2r':{'internal_evidence':'results/bime_rank_unified_v1/e2r_seed_context_v1/development_result.json','strict_temporal_retention':'results/bime_rank_unified_v1/e2r_seed_context_retention_v1/summary.json','ranker':'results/bime_rank_unified_v1/e2r_seed_context_v1/selected/ranker.json','ranker_sha256':e2ret['seed_ranker_sha256'],'retention_queries':e2ret['eligible_queries'],'retention_trials':e2ret['trials'],'metrics':e2ret['metrics']['context'],'delta_vs_zero_shot':e2ret['delta_context_vs_zero_shot']},
 'external_metrics_used_for_selection':False,'external_metrics_used_for_retuning':False,
 'claim_boundary':'conditional known-positive retrieval only; these metrics are not zero-shot claims',
 'seed_set_semantics':{'r2e':'max cosine over all provided verified protein seeds','e2r':'mean across frozen reaction experts of max cosine over all provided verified reaction seeds'},
 'multi_seed_scaling':{'evidence':'results/bime_rank_unified_v1/multiseed_scaling_v1/summary.json','design':'same query and same hidden target; nested 1/2/3/5 seed sets; one-seed-trained cross-fit ranker frozen; no multi-seed retuning','r2e_common_cohort_queries':ctx1r['1']['queries'],'e2r_common_cohort_queries':ctx1e['1']['queries'],'r2e_mrr_1_to_5':[ctx1r['1']['mrr'],ctx1r['5']['mrr']],'r2e_hit50_1_to_5':[ctx1r['1']['hit_at_50'],ctx1r['5']['hit_at_50']],'e2r_mrr_1_to_5':[ctx1e['1']['mrr'],ctx1e['5']['mrr']],'e2r_hit10_1_to_5':[ctx1e['1']['hit_at_10'],ctx1e['5']['hit_at_10']]}}
payload['experts']['homology_context']={
 'direction':'reaction_to_enzyme','status':'rejected_external_retention','role':'analysis_and_legacy_context_only_not_current_bime_ranking_expert',
 'candidate_recall_diagnostic':'results/bime_rank_unified_v1/r2e_homology_admission_v1/candidate_recall_diagnostic.json','internal_evidence':'results/bime_rank_unified_v1/r2e_homology_context_v1/development_result.json',
 'internal_baseline':hdev['base_structural_bime'],'internal_candidate':hdev['homology_context'],'internal_delta':hdev['delta'],
 'frozen_ranker_sha256':hret['homology_ranker_sha256'],'external_evidence':'results/bime_rank_unified_v1/r2e_homology_context_retention_v1/summary.json',
 'external_baseline':subset(hret['base_structural_bime'],MET),'external_candidate':subset(hret['homology_context'],MET),'external_delta':hret['delta'],
 'reason':'train-only homology context showed internal complementarity but materially degraded frozen strict temporal MRR/MAP/NDCG; external labels were veto-only and no retuning was allowed'}
payload['experts']['reciprocal_consistency']={
 'direction':'reaction_to_enzyme','status':'rejected_external_retention','role':'ablation_only','internal_evidence':'results/bime_rank_unified_v1/r2e_reciprocal_consistency_v1/development_result.json',
 'internal_baseline':recdev['old'],'internal_candidate':recdev['new'],'internal_delta':recdev['delta'],'external_evidence':'results/bime_rank_unified_v1/r2e_reciprocal_external_confirmation_v1/summary.json',
 'external_vs_structural_bime':recext['paired_bootstrap_50000_vs_structural_bime'],
 'reason':'internal gains did not transport to frozen strict temporal/double-cold confirmation; no external retuning allowed'}
payload['experts']['enzymecage_top20_structure']={
 'direction':'reaction_to_enzyme_tps_conditional','status':'rejected_internal_oof','role':'external_baseline_and_structure_diagnostic_only','evidence':'results/bime_rank_unified_v1/tps_cage_top20_expert_v1/development_result.json',
 'same_capacity_baseline':cage['same_capacity_baseline_ranker'],'candidate':cage['cage_top20_expert'],'delta':cage['delta_vs_same_capacity'],
 'coverage':f"{cageprep['pairs']}/{cageprep['pairs']} fixed Catalyst Top{cageprep['topk']} pairs on {cageprep['queries']} TPS queries had official EnzymeCAGE scores",
 'reason':'fixed five-fold OOF showed no robust benefit and negative MRR/MAP/NDCG/Hit10 deltas'}
payload['staging']={
 'e2r_smoke':e2sm,'r2e_smoke':r2sm,'manifest_semantic_diff_status':mdiff['status'],'manifest_semantic_diff_count':mdiff['semantic_diff_count'],'unexpected_manifest_diffs':len(mdiff['unexpected_diffs']),
 'default_production_overwritten':prod1['status']=='promoted_and_verified','e2r_runtime_strict_bit_diagnostic':'results/bime_rank_unified_v1/promotion_audit_v1/e2r_runtime_equivalence.json','e2r_runtime_retention':'results/bime_rank_unified_v1/promotion_audit_v1/e2r_runtime_retention.json','post_promotion_smoke':'results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_smoke/summary.json','post_promotion_smoke_status':psmoke['status'],'post_promotion_manifest_equivalence_status':equiv['status'],'post_promotion_regression_passed':prod1['post_promotion_regression']['passed'],'post_promotion_regression_failed':prod1['post_promotion_regression']['failed'],'post_promotion_result':'projects/active/terpene_screening/BIME_RANK_PRODUCTION_V1_RESULT.json','asset_scope':'historical first promotion audit; current production uses current_production_result'}
payload['production']={'manifest':equiv['production_manifest'],'route_version':equiv['production_route_version'],'manifest_sha256':equiv['production_sha256'],'rollback_manifest':equiv['rollback_manifest'],'rollback_sha256':equiv['rollback_sha256'],'pre_promotion_manifest_sha256':gate['pre_promotion']['default_manifest_sha256'],'promotion_gate':'results/bime_rank_unified_v1/promotion_audit_v1/promotion_gate.json','post_promotion_manifest_equivalence':'results/bime_rank_unified_v1/promotion_audit_v1/post_promotion_manifest_equivalence.json'}
payload['current_production_v2']={'route_version':prod2['route_version'],'bidirectional_seed_context_promoted':bool(prod2['seed_context']['r2e_retention_passed'] and prod2['seed_context']['e2r_retention_passed']),'structure_experts_promoted':True,'homology_context_deployed':'homology_context' not in prod2['rejected_not_deployed'],'reciprocal_consistency_deployed':'reciprocal_consistency' not in prod2['rejected_not_deployed'],'enzymecage_as_expert_deployed':'enzymecage_top20_structure' not in prod2['rejected_not_deployed'],'final_regression_passed':prod2['final_regression']['passed'],'final_regression_failed':prod2['final_regression']['failed']}
payload['current_production_result']='projects/active/terpene_screening/BIME_RANK_PRODUCTION_V2_RESULT.json'


DEFAULT_OUTPUT = ROOT / "projects/active/terpene_screening/BIME_RANK_EXPERT_ADMISSION_V1.json"


def assembled_payload() -> dict:
    return payload


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Deterministically assemble the BiME-Rank expert-admission aggregate from retained component evidence.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-existing", action="store_true", help="Require assembled semantic content to equal the existing output without modifying it.")
    args = parser.parse_args()
    target = args.output if args.output.is_absolute() else ROOT / args.output
    try:
        display_output = str(target.relative_to(ROOT))
    except ValueError:
        display_output = str(target)
    value = assembled_payload()
    if args.verify_existing:
        if not target.is_file():
            raise FileNotFoundError(target)
        existing = json.loads(target.read_text())
        if existing != value:
            raise SystemExit(f"assembled expert-admission payload differs from {target}")
        print(json.dumps({"status":"match","output":display_output}, indent=2))
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps({"status":"written","output":display_output}, indent=2))


if __name__ == "__main__":
    main()
