from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json'
BUILDER=ROOT/'projects/active/terpene_screening/prepare_rhea_snapshot_delta_external_benchmark_v2.py'
BROAD=ROOT/'projects/active/terpene_screening/evaluate_broad_rhea_benchmark.py'
FINAL=ROOT/'projects/active/terpene_screening/evaluate_rhea128_to141_external_v2.py'
BASE_FEATURES=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'
CENTER_FEATURES=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1'
BASE_MODEL=ROOT/'results/rhea128_to141_external_v1/models/base'
CANDIDATE_MODEL=ROOT/'results/rhea128_to141_external_v1/models/residual'
DEFAULT_OUTPUT=ROOT/'results/rhea128_to141_external_v2'

def support_decision(manifest:dict,protocol:dict)->dict:
    audit=manifest['audit']; rule=protocol['minimum_support_rule']
    nq=int(audit['test_query_reactions']); npairs=int(audit['test_pairs'])
    met=nq>=int(rule['min_query_reactions']) and npairs>=int(rule['min_test_pairs'])
    return {'observed_query_reactions':nq,'required_query_reactions':int(rule['min_query_reactions']),'observed_test_pairs':npairs,'required_test_pairs':int(rule['min_test_pairs']),'minimum_support_met':met,'action':'score_once_with_frozen_models' if met else 'underpowered_stop_without_model_scoring'}

def run(cmd:list[str])->None:
    print('+',' '.join(cmd),flush=True); subprocess.run(cmd,check=True,cwd=ROOT)

def main()->None:
    ap=argparse.ArgumentParser(description='One-way frozen Rhea128→141 V2 reveal: materialize support, stop if underpowered, otherwise score fixed models exactly once.')
    ap.add_argument('--release128-sprot',type=Path,required=True); ap.add_argument('--release141-sprot',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT); ap.add_argument('--device',default='cuda')
    args=ap.parse_args(); protocol=json.loads(PROTOCOL.read_text()); out=args.output_root.resolve(); cell=protocol['benchmark_cell']
    run([sys.executable,str(BUILDER),'--release128-sprot',str(args.release128_sprot.resolve()),'--release141-sprot',str(args.release141_sprot.resolve()),'--output-root',str(out)])
    manifest_path=out/cell/'manifest.json'; manifest=json.loads(manifest_path.read_text()); decision=support_decision(manifest,protocol)
    decision.update({'protocol':str(PROTOCOL),'benchmark_cell':cell,'model_performance_read_before_support_gate':False})
    (out/'support_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(json.dumps(decision,indent=2),flush=True)
    if not decision['minimum_support_met']:
        (out/'external_result.json').write_text(json.dumps({'status':'underpowered_external_descriptive','pass':False,'support_decision':decision,'model_scores_materialized':False,'post_reveal_retuning_allowed':False},indent=2)+'\n')
        return
    common=['--cell',cell,'--benchmark-root',str(out),'--e2r-model-dir',str(BASE_MODEL),'--e2r-reaction-feature-dir',str(BASE_FEATURES),'--max-e2r-queries','1','--device',args.device]
    eval_base=out/'eval_base'; eval_candidate=out/'eval_candidate'
    run([sys.executable,str(BROAD),*common,'--r2e-model-dir',str(BASE_MODEL),'--r2e-reaction-feature-dir',str(BASE_FEATURES),'--output-dir',str(eval_base)])
    run([sys.executable,str(BROAD),*common,'--r2e-model-dir',str(CANDIDATE_MODEL),'--r2e-reaction-feature-dir',str(CENTER_FEATURES),'--output-dir',str(eval_candidate)])
    run([sys.executable,str(FINAL),'--benchmark-root',str(out),'--baseline-root',str(eval_base),'--candidate-root',str(eval_candidate),'--output',str(out/'external_result.json')])
if __name__=='__main__': main()
