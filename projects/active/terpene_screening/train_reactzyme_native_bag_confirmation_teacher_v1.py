from __future__ import annotations
import hashlib,json,sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd, torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.rank_open_world import load_feature_schema,load_protein_library,load_registered_reaction_feature_library
from projects.active.terpene_screening.train_cleanroom_rhea_retriever import train_cleanroom
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig
SPLIT=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation/split'
OUT=ROOT/'results/reactzyme_native_bag_adapter_v1_confirmation/teacher'
FEATURE=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'
PROT=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
SCHEMA=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 train_path=SPLIT/'training_pairs.csv'; train=pd.read_csv(train_path,dtype=str).fillna('')[['protein_id','reaction_id']].drop_duplicates()
 schema=load_feature_schema(SCHEMA); pf,pids=load_protein_library(PROT); rf,rids=load_registered_reaction_feature_library(FEATURE,schema)
 pset,rset=set(pids),set(rids); train=train[train.protein_id.isin(pset)&train.reaction_id.isin(rset)].reset_index(drop=True)
 cfg=ModelConfig(protein_input_dim=pf.shape[1],reaction_input_dim=rf.shape[1],hidden_dim=768,embedding_dim=320,dropout=0.1)
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model,hist,neighbors,novelty=train_cleanroom(pf,pids,rf,rids,train,config=cfg,epochs=8,steps_per_epoch=60,reaction_batch_size=64,protein_batch_size=48,hard_negatives=80,random_negatives=8,hard_negative_start=0,hard_negative_ramp_epochs=0,neighbor_k=32,learning_rate=3e-4,weight_decay=1e-4,temperature=0.035,topk=10,topk_weight=1.0,margin=0.12,r2e_weight=0.98,reaction_novelty_threshold=0.7,reaction_novelty_repeat=0,seed=20260723,device=device,neighbor_queries=set())
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'models').mkdir(exist_ok=True)
 ck=OUT/'models/production_seed20260723.pt'; torch.save({'model_type':'dual_tower','model_state_dict':model.state_dict(),'model_config':asdict(cfg),'seed':20260723,'training_pairs_sha256':sha(train_path),'confirmation_dev_read':False},ck)
 pd.DataFrame(hist).to_csv(OUT/'training_history.csv',index=False)
 summary={'status':'trained_train_only','training_pairs':len(train),'train_proteins':train.protein_id.nunique(),'train_reactions':train.reaction_id.nunique(),'training_pairs_sha256':sha(train_path),'confirmation_dev_path_opened':False,'confirmation_dev_labels_read':False,'confirmation_dev_metadata_read':False,'neighbor_queries_from_dev':False,'model_config':asdict(cfg),'recipe':{'epochs':8,'steps_per_epoch':60,'reaction_batch_size':64,'protein_batch_size':48,'neighbor_k':32,'hard_negatives':80,'random_negatives':8,'learning_rate':3e-4,'weight_decay':1e-4,'temperature':0.035,'topk':10,'topk_weight':1.0,'margin':0.12,'r2e_weight':0.98,'reaction_novelty_repeat':0,'seed':20260723},'checkpoint':str(ck),'checkpoint_sha256':sha(ck)}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
