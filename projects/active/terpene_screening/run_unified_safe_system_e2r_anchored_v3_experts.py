from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
TRAIN=ROOT/'projects/active/terpene_screening/train_cleanroom_rhea_retriever.py'
ASSOC=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
BASE_SCHEMA=ROOT/'results/terpene_production_models/marts_adapted_drfp_pu'
BASE_RXN=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1'
RDKIT=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'
PROTEINS={
 'enzgfm':ROOT/'data/external/enzgfm_current/clean2023_650m_mean',
 'esmc':ROOT/'data/catalyst_candidate_universes/general_merged/proteins',
 'equalblock':ROOT/'data/external/enzgfm_current/clean2023_esmc_enzgfm_equalblock',
 'rdkitplus':ROOT/'data/external/enzgfm_current/clean2023_esmc_enzgfm_equalblock',
}
DEV_SALT='e2r_anchored_lambdamart_v3_dev_20260902_c'
OUT=ROOT/'results/unified_safe_system_v1/e2r_anchored_lambdamart_v3_dev/experts'

def complete(path:Path)->bool:
 if not path.is_file(): return False
 d=json.loads(path.read_text()); return d.get('split_salt')==DEV_SALT and int(d.get('folds',0))==5 and 'common_ir_e2r' in (d.get('dev_metrics') or {})

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--expert',choices=[*PROTEINS,'all'],default='all'); ap.add_argument('--fold',type=int,choices=[0,1,2]); a=ap.parse_args()
 names=list(PROTEINS) if a.expert=='all' else [a.expert]; folds=[0,1,2] if a.fold is None else [a.fold]
 for name in names:
  for fold in folds:
   out=OUT/name/f'fold{fold}'; summary=out/'summary.json'
   if complete(summary): print('SKIP',name,fold,flush=True); continue
   rxn=RDKIT if name=='rdkitplus' else BASE_RXN; schema=RDKIT if name=='rdkitplus' else BASE_SCHEMA
   cmd=[str(ROOT/'.venv/bin/python'),str(TRAIN),'--associations-csv',str(ASSOC),'--schema-dir',str(schema),'--reaction-feature-dir',str(rxn),'--protein-feature-dir',str(PROTEINS[name]),'--output-dir',str(out),'--dev-fold',str(fold),'--folds','5','--split-salt',DEV_SALT,'--epochs','8','--steps-per-epoch','60','--reaction-batch-size','64','--protein-batch-size','48','--neighbor-k','32','--dev-neighbor-reactions','10','--hard-negatives','80','--random-negatives','8','--hard-negative-ramp-epochs','0','--hidden-dim','768','--embedding-dim','320','--dropout','0.1','--learning-rate','3e-4','--weight-decay','1e-4','--temperature','0.035','--topk','10','--topk-weight','1.0','--margin','0.12','--r2e-weight','0.70','--reaction-novelty-repeat','0','--seed','20260723','--device',a.device]
   print('+',' '.join(cmd),flush=True); subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__': main()
