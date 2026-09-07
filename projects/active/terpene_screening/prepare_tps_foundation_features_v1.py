from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.terpene_screening.build_enzgfm_protein_features import load_runtime, truncate_middle

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/'data/terpene_marts_adaptation_confirmatory20260726'
GENERAL_SEQ=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
GENERAL_ENZGFM=ROOT/'data/external/enzgfm_current/general_merged_650m_mean_v1'
OUT=ROOT/'data/terpene_tps_foundation_v1'


def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''): h.update(c)
 return h.hexdigest()

def norm(x):
 x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x,axis=1,keepdims=True); n[n==0]=1; return x/n

def write_lib(path:Path,ids:list[str],matrix:np.ndarray,manifest:dict):
 path.mkdir(parents=True,exist_ok=True); pd.DataFrame({'row':np.arange(len(ids)),'Entry':ids}).to_csv(path/'entries.csv',index=False); np.save(path/'embeddings.npy',np.asarray(matrix,dtype=np.float32)); (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def encode_missing(seqs:list[str],device:str)->np.ndarray:
 if not seqs: return np.empty((0,2048),dtype=np.float32)
 model,tokenizer,config=load_runtime(ROOT/'external/enzgfm_reference',ROOT/'external_models/enzgfm/EnzGFM_650M',torch.device(device))
 texts=[truncate_middle(s,1000) for s in seqs]
 enc=tokenizer(texts,return_tensors='pt',truncation=True,max_length=1002,padding=True); enc={k:v.to(device) for k,v in enc.items()}
 with torch.no_grad():
  states=model(**enc,output_hidden_states=False,return_dict=True).last_hidden_state.float(); mask=enc['attention_mask'].unsqueeze(-1).float(); pooled=((states*mask).sum(1)/mask.sum(1)).cpu().numpy().astype(np.float32)
 assert pooled.shape==(len(seqs),int(config.hidden_size)); return pooled

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
 p=pd.read_csv(CACHE/'protein_entities.csv',dtype=str).fillna('').reset_index(drop=True); ids=p.protein_id.astype(str).tolist(); seqs=p.sequence.astype(str).tolist(); esmc=np.load(CACHE/'protein_features.npy').astype(np.float32); assert len(p)==len(esmc)
 write_lib(out/'esmc',ids,esmc,{'version':'tps-esmc-from-frozen-marts-cache-v1','source':str(CACHE/'protein_features.npy'),'source_sha256':sha(CACHE/'protein_features.npy'),'protein_count':len(ids),'feature_dimension':esmc.shape[1],'labels_used':False})
 # Reuse exact-sequence EnzGFM rows from the immutable general library, encode only missing sequences.
 ge=pd.read_csv(GENERAL_ENZGFM/'entries.csv',dtype={'Entry':str}).sort_values('row').reset_index(drop=True); gx=np.load(GENERAL_ENZGFM/'embeddings.npy',mmap_mode='r'); gs=pd.read_csv(GENERAL_SEQ,sep='\t',dtype=str).fillna(''); id2seq=dict(zip(gs.protein_id.astype(str),gs.sequence.astype(str))); seq2row={}
 for i,eid in enumerate(ge.Entry.astype(str)):
  s=id2seq.get(eid,'');
  if s and s not in seq2row: seq2row[s]=i
 enz=np.zeros((len(ids),gx.shape[1]),dtype=np.float32); missing=[]
 for i,s in enumerate(seqs):
  r=seq2row.get(s)
  if r is None: missing.append(i)
  else: enz[i]=np.asarray(gx[r],dtype=np.float32)
 miss_vec=encode_missing([seqs[i] for i in missing],a.device)
 for j,i in enumerate(missing): enz[i]=miss_vec[j]
 write_lib(out/'enzgfm',ids,enz,{'version':'tps-enzgfm650m-exact-sequence-reuse-plus-missing-v1','general_source':str(GENERAL_ENZGFM),'protein_count':len(ids),'feature_dimension':enz.shape[1],'reused_exact_sequence_rows':len(ids)-len(missing),'encoded_missing_rows':len(missing),'missing_protein_ids':[ids[i] for i in missing],'labels_used':False})
 eq=np.concatenate([norm(esmc),norm(enz)],axis=1).astype(np.float32); write_lib(out/'equalblock',ids,eq,{'version':'tps-esmc-enzgfm-equalblock-v1','protein_count':len(ids),'feature_dimension':eq.shape[1],'base_dimension':esmc.shape[1],'aux_dimension':enz.shape[1],'combination':'independent L2 normalize then concatenate','labels_used':False})
 # Preserve the exact historical TPS 2115-d base; build RDKit+ as an append-only block.
 r=pd.read_csv(CACHE/'reaction_entities.csv',dtype=str).fillna('').reset_index(drop=True); rx=np.load(CACHE/'reaction_features.npy').astype(np.float32); assert len(r)==len(rx)
 base=out/'reaction_2115'; base.mkdir(exist_ok=True); pd.DataFrame({'row':np.arange(len(r)),'reaction_id':r.reaction_id.astype(str)}).to_csv(base/'entries.csv',index=False); np.save(base/'reaction_feature_matrix.npy',rx)
 schema=json.loads((ROOT/'results/terpene_production_models/marts_adapted_drfp_pu/feature_schema.json').read_text()); schema['reaction_ids']=r.reaction_id.astype(str).tolist(); schema['reaction_feature_dimension']=int(rx.shape[1]); (base/'feature_schema.json').write_text(json.dumps(schema,indent=2)+'\n'); manifest={'version':'tps-frozen-2115-reaction-features-v1','feature_dimension':rx.shape[1],'reaction_count':len(r),'source':str(CACHE/'reaction_features.npy'),'source_sha256':sha(CACHE/'reaction_features.npy'),'contract':{'reaction_feature_dimension':int(rx.shape[1]),'drfp_dimension':int(schema.get('drfp_dimension',2048))},'labels_used':False}; (base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 reactions=out/'reactions.csv'; r[['reaction_id','reaction_smiles']].to_csv(reactions,index=False)
 rd=out/'reaction_3139_rdkitplus'
 subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'projects/active/terpene_screening/build_rdkitplus_augmented_reaction_features.py'),'--reactions',str(reactions),'--base-feature-dir',str(base),'--base-schema-dir',str(base),'--output-dir',str(rd)],cwd=ROOT,check=True)
 aug=np.load(rd/'reaction_feature_matrix.npy',mmap_mode='r'); assert aug.shape==(len(r),3139); assert np.array_equal(np.asarray(aug[:,:2115]),rx)
 summary={'status':'ready','protein_count':len(ids),'reaction_count':len(r),'protein_features':{'esmc':1152,'enzgfm':2048,'equalblock':3200},'reaction_features':{'base':2115,'rdkitplus':3139},'enzgfm_reused':len(ids)-len(missing),'enzgfm_encoded_missing':len(missing),'rdkitplus_base_exact_parity':True,'labels_used':False}; (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
