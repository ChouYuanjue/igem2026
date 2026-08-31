from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
from projects.active.terpene_screening.evaluate_rhea128_to141_external import evaluate
ROOT=Path(__file__).resolve().parents[3]
PROTOCOL=ROOT/'projects/active/terpene_screening/CLEANROOM_R2E_RHEA128_TO141_EXTERNAL_V2.json'
def sha(path:Path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--benchmark-root',type=Path,required=True); ap.add_argument('--baseline-root',type=Path,required=True); ap.add_argument('--candidate-root',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 p=json.loads(PROTOCOL.read_text()); cell=p['benchmark_cell']; mp=a.benchmark_root.resolve()/cell/'manifest.json'; bp=a.baseline_root.resolve()/cell/'query_metrics.csv'; cp=a.candidate_root.resolve()/cell/'query_metrics.csv'
 m=json.loads(mp.read_text());
 if m.get('name')!=cell or m.get('identifier_semantics','').startswith('direction-specific RHEA_ID') is False: raise ValueError('benchmark manifest is not frozen V2 semantics')
 r=evaluate(m,pd.read_csv(bp,dtype={'query_id':str}),pd.read_csv(cp,dtype={'query_id':str}),min_queries=p['minimum_support_rule']['min_query_reactions'],min_pairs=p['minimum_support_rule']['min_test_pairs'],candidate_count=p['fixed_support']['expected_protein_candidates'])
 r.update({'protocol':str(PROTOCOL),'protocol_sha256':sha(PROTOCOL),'benchmark_cell':cell,'benchmark_manifest_sha256':sha(mp),'baseline_query_metrics_sha256':sha(bp),'candidate_query_metrics_sha256':sha(cp),'fresh_external_snapshot':True,'identifier_semantics':'RHEA_ID + alias-mapped protein'})
 a.output.resolve().parent.mkdir(parents=True,exist_ok=True); a.output.resolve().write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
