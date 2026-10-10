"""Exact released CAGE gate + NEW 5216-only R2E graph/functional rerank.

Preserves the original author-gated candidate pool, candidate external
ESM-C materialization, and all filtering semantics of the official hybrid
evaluator; only the R2E graph authority and EnzGFM scaling are refit on
the single 5216-positive development source. Exploratory post-hoc test.
"""
from __future__ import annotations
import json,pickle
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime
from reproducibility.bime_rank.scripts import evaluate_bridge_layered_v4_cage_gate_bridge as original

def main():
 dest=ROOT/'results/bridge_r2e_single_validation_cage_gate_v1'
 dest.mkdir(parents=True,exist_ok=True)
 if (dest/'r2e_edge_metrics.csv.gz').exists():raise RuntimeError('Refuse recomputing finalized gated rerank')
 original.OUT=dest
 rt=FinalBridgeRuntime(device='cuda')
 gate=ROOT/'results/bridge_single_validation_v1/r2e_relation_gate.exploratory.pkl'
 with gate.open('rb') as fp:model=pickle.load(fp)
 assert model['direction']=='r2e'
 rt.relation_gate['directions']['r2e']=model
 from projects.active.bridge.runtime.final_system import R2E_FUNCTIONAL_VALIDATION_SCALE
 assert R2E_FUNCTIONAL_VALIDATION_SCALE==1.25
 print('RUN_CAGE_FIXED_POOL_5216_TRAINED_R2E_GATE_BETA1p25',flush=True)
 outcome=original.evaluate_r2e(rt)
 record={'schema':'BRIDGE_R2E_CAGE_NATIVE_FIXED_GATE_REUNIFIED_VALIDATION_V1',
        'source_only_relation_validation_edges':5216,
        'legacy_16108_used_for_fit':False,
        'functional_score_multiplier':1.25,
        'same_raw_unseen_pairs':23773,
        'fixed_final_test_pairs_after_protein_query_split':21505,
        'original_cage_pool_frozen':True,
        'classification':'exploratory posthoc rescoring; already viewed final test cannot be claimed independent',
        'result':outcome}
 (dest/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
 print('CAGE_NATIVE_R2E_GATE_NEW_RECALIBRATION_DONE',json.dumps(outcome['metrics']),flush=True)
if __name__=='__main__':main()
