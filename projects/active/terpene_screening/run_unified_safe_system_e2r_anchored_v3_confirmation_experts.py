from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from projects.active.terpene_screening.run_unified_safe_system_e2r_anchored_v3_experts import TRAIN,ASSOC,BASE_SCHEMA,BASE_RXN,RDKIT,PROTEINS,ROOT

PROTOCOL=ROOT/'projects/active/terpene_screening/UNIFIED_SAFE_SYSTEM_E2R_ANCHORED_LAMBDAMART_V3_CONFIRMATION.json'
OUT=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_confirmation/experts'

def protocol(): return json.loads(PROTOCOL.read_text())
def complete(path:Path,salt:str,folds:int)->bool:
 if not path.is_file(): return False
 d=json.loads(path.read_text()); return d.get('split_salt')==salt and int(d.get('folds',0))==folds and 'common_ir_e2r' in (d.get('dev_metrics') or {})

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--expert',choices=[*PROTEINS,'all'],default='all'); a=ap.parse_args()
 p=protocol(); split=p['confirmation_split']; salt=str(split['split_salt']); folds=int(split['folds']); dev_fold=int(split['dev_fold']); names=list(PROTEINS) if a.expert=='all' else [a.expert]
 for name in names:
  out=OUT/name/f'fold{dev_fold}'; summary=out/'summary.json'
  if complete(summary,salt,folds): print('SKIP',name,dev_fold,flush=True); continue
  rxn=RDKIT if name=='rdkitplus' else BASE_RXN; schema=RDKIT if name=='rdkitplus' else BASE_SCHEMA
  cmd=[str(ROOT/'.venv/bin/python'),str(TRAIN),'--associations-csv',str(ASSOC),'--schema-dir',str(schema),'--reaction-feature-dir',str(rxn),'--protein-feature-dir',str(PROTEINS[name]),'--output-dir',str(out),'--dev-fold',str(dev_fold),'--folds',str(folds),'--split-salt',salt,'--epochs','8','--steps-per-epoch','60','--reaction-batch-size','64','--protein-batch-size','48','--neighbor-k','32','--dev-neighbor-reactions','10','--hard-negatives','80','--random-negatives','8','--hard-negative-ramp-epochs','0','--hidden-dim','768','--embedding-dim','320','--dropout','0.1','--learning-rate','3e-4','--weight-decay','1e-4','--temperature','0.035','--topk','10','--topk-weight','1.0','--margin','0.12','--r2e-weight','0.70','--reaction-novelty-repeat','0','--seed','20260723','--device',a.device]
  print('+',' '.join(cmd),flush=True); subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__': main()
