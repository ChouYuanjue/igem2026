from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.train_cleanroom_rhea_retriever import split_double_cold
SRC=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
OUT=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation/split'
SALT='reactzyme_native_bag_confirm_v1_20260901_a'; FOLDS=7; DEV=6

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pairs=pd.read_csv(SRC,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates()
 train,dev=split_double_cold(pairs,dev_fold=DEV,folds=FOLDS,salt=SALT)
 assert not(set(train.protein_id)&set(dev.protein_id)); assert not(set(train.reaction_id)&set(dev.reaction_id))
 OUT.mkdir(parents=True,exist_ok=True); train.to_csv(OUT/'training_pairs.csv',index=False); dev.to_csv(OUT/'dev_pairs.csv',index=False)
 manifest={'status':'materialized','source':str(SRC),'source_sha256':sha(SRC),'salt':SALT,'fold_count':FOLDS,'confirmation_fold':DEV,'train_pairs':len(train),'dev_pairs':len(dev),'train_proteins':train.protein_id.nunique(),'dev_proteins':dev.protein_id.nunique(),'train_reactions':train.reaction_id.nunique(),'dev_reactions':dev.reaction_id.nunique(),'protein_overlap':0,'reaction_overlap':0,'training_pairs_sha256':sha(OUT/'training_pairs.csv'),'dev_pairs_sha256':sha(OUT/'dev_pairs.csv')}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
