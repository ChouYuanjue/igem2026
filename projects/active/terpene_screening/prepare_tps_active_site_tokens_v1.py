from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.extract_esmc_motif_context_embeddings import select_motif_contexts
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature
from projects.active.terpene_screening.rank_open_world import load_esmc_model_cached

PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
REACTIONS=ROOT/'data/terpene_marts_adaptation/reaction_entities.csv'
GENERAL_RXN=ROOT/'data/catalyst_candidate_universes/general_merged/reactions.csv'
MAPPED=ROOT/'data/external/rxnmapper_current/general_merged_v1/mapped_reactions.csv'
OUT=ROOT/'data/terpene_tps_active_site_xattn_v1'
MOTIF_SLOTS=('ddxxd','nse_dte','dxdd','qw1','qw2')
WINDOW=12
SLOT_WIDTH=2*WINDOW+1
PROTEIN_BUDGET=1+len(MOTIF_SLOTS)*SLOT_WIDTH
REACTION_SIDE_BUDGET=52
REACTION_BUDGET=104
ATOM_DIM=23

def sha256_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def stable_mapped_reactions()->tuple[dict[str,str],dict[str,object]]:
 rx=pd.read_csv(REACTIONS,dtype=str).fillna('')
 general=pd.read_csv(GENERAL_RXN,dtype=str).fillna('')
 general['signature']=general.reaction_smiles.map(reaction_signature)
 mp=pd.read_csv(MAPPED,dtype=str).fillna(''); mp['success_b']=mp.success.astype(str).str.lower().eq('true'); mp['confidence_n']=pd.to_numeric(mp.confidence,errors='coerce').fillna(-1.0)
 good=mp[mp.success_b & mp.mapped_rxn.ne('')].copy()
 joined=general.merge(good[['reaction_id','mapped_rxn','confidence_n']],on='reaction_id',how='inner')
 by={}
 audits=[]
 for row in rx.itertuples(index=False):
  z=joined[joined.signature.eq(str(row.reaction_signature))].sort_values(['confidence_n','reaction_id'],ascending=[False,True])
  if z.empty: continue
  best=z.iloc[0]; by[str(row.reaction_id)]=str(best.mapped_rxn)
  audits.append({'reaction_id':str(row.reaction_id),'source_reaction_id':str(best.reaction_id),'mapping_confidence':float(best.confidence_n),'matching_mapped_candidates':int(len(z))})
 return by,{'mapped_count':len(by),'audit':audits}

def changed_map_ids(mapped:str)->set[int]:
 left,right=mapped.split('>>')
 a=Chem.MolFromSmiles(left); b=Chem.MolFromSmiles(right)
 if a is None or b is None: raise ValueError('bad mapped reaction')
 def amap(m): return {int(x.GetAtomMapNum()):x.GetIdx() for x in m.GetAtoms() if x.GetAtomMapNum()>0}
 def bonds(m):
  out={}
  for bond in m.GetBonds():
   x=m.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(); y=m.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum()
   if x>0 and y>0: out[tuple(sorted((int(x),int(y))))]=float(bond.GetBondTypeAsDouble())
  return out
 def state(x): return (x.GetAtomicNum(),x.GetFormalCharge(),x.GetTotalNumHs(),int(x.GetIsAromatic()),int(x.GetChiralTag()))
 am,bm=amap(a),amap(b); ab,bb=bonds(a),bonds(b); changed=set()
 for pair in set(ab)|set(bb):
  if ab.get(pair,0.0)!=bb.get(pair,0.0): changed.update(pair)
 for k in set(am)&set(bm):
  if state(a.GetAtomWithIdx(am[k]))!=state(b.GetAtomWithIdx(bm[k])): changed.add(k)
 return changed

def atom_vector(atom:Chem.Atom,*,product:bool,changed:bool)->np.ndarray:
 v=[]
 v += [atom.GetAtomicNum()/100.0, min(atom.GetDegree(),6)/6.0, max(-4,min(4,atom.GetFormalCharge()))/4.0, min(atom.GetTotalNumHs(),4)/4.0]
 v += [float(atom.GetIsAromatic()),float(atom.IsInRing()),int(atom.GetChiralTag())/4.0,float(product),float(changed),float(atom.GetAtomMapNum()>0)]
 hyb=str(atom.GetHybridization()); hs=['SP','SP2','SP3','SP3D','SP3D2']; v += [float(hyb==x) for x in hs]+[float(hyb not in hs)]
 z=atom.GetAtomicNum(); classes=[z==6,z==7,z==8,z==15,z==16,z in {9,17,35,53},z not in {6,7,8,9,15,16,17,35,53}]; v += [float(x) for x in classes]
 x=np.asarray(v,dtype=np.float32); assert x.shape==(ATOM_DIM,); return x

def build_reaction_tokens(out:Path)->dict:
 out.mkdir(parents=True,exist_ok=True); rx=pd.read_csv(REACTIONS,dtype=str).fillna(''); mapped,ma=stable_mapped_reactions()
 feats=np.zeros((len(rx),REACTION_BUDGET,ATOM_DIM),dtype=np.float16); mask=np.zeros((len(rx),REACTION_BUDGET),dtype=bool); amap_out=np.zeros((len(rx),REACTION_BUDGET),dtype=np.int16); side=np.zeros((len(rx),REACTION_BUDGET),dtype=np.int8); changed_out=np.zeros((len(rx),REACTION_BUDGET),dtype=bool); audit=[]
 for i,row in enumerate(rx.itertuples(index=False)):
  rid=str(row.reaction_id); m=mapped.get(rid)
  if not m: raise RuntimeError(f'missing mapped reaction {rid}')
  changed=changed_map_ids(m); left,right=m.split('>>'); mols=[Chem.MolFromSmiles(left),Chem.MolFromSmiles(right)]; counts=[]
  for s,mol in enumerate(mols):
   assert mol is not None
   atoms=sorted(list(mol.GetAtoms()),key=lambda a:(a.GetAtomMapNum()<=0,a.GetAtomMapNum() if a.GetAtomMapNum()>0 else a.GetIdx()))
   if len(atoms)>REACTION_SIDE_BUDGET: raise RuntimeError(f'{rid} side{s} atoms={len(atoms)}>52')
   counts.append(len(atoms)); base=s*REACTION_SIDE_BUDGET
   for j,a in enumerate(atoms):
    k=base+j; ischg=int(a.GetAtomMapNum()) in changed if a.GetAtomMapNum()>0 else False
    feats[i,k]=atom_vector(a,product=bool(s),changed=ischg); mask[i,k]=True; amap_out[i,k]=int(a.GetAtomMapNum()); side[i,k]=s; changed_out[i,k]=ischg
  audit.append({'row':i,'reaction_id':rid,'reactant_atoms':counts[0],'product_atoms':counts[1],'changed_map_count':len(changed),'token_count':sum(counts)})
 np.save(out/'reaction_atom_features.npy',feats); np.save(out/'reaction_atom_mask.npy',mask); np.save(out/'reaction_atom_map.npy',amap_out); np.save(out/'reaction_atom_side.npy',side); np.save(out/'reaction_atom_changed.npy',changed_out); pd.DataFrame({'row':range(len(rx)),'reaction_id':rx.reaction_id}).to_csv(out/'reaction_entries.csv',index=False); pd.DataFrame(audit).to_csv(out/'reaction_token_audit.csv',index=False); pd.DataFrame(ma['audit']).to_csv(out/'reaction_mapping_audit.csv',index=False)
 result={'reaction_count':len(rx),'mapped_count':ma['mapped_count'],'token_budget':REACTION_BUDGET,'side_budget':REACTION_SIDE_BUDGET,'atom_feature_dim':ATOM_DIM,'max_reactant_atoms':max(x['reactant_atoms'] for x in audit),'max_product_atoms':max(x['product_atoms'] for x in audit),'changed_reactions':sum(x['changed_map_count']>0 for x in audit),'labels_used':False}
 (out/'reaction_tokens_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def motif_centers_for_slots(seq:str)->list[int|None]:
 c=select_motif_contexts(seq)
 def one(name):
  z=list(c[name]['selected_positions']); return int(z[0]) if z else None
 q=sorted(map(int,c['qw']['selected_positions']))[:2]
 return [one('ddxxd'),one('nse_dte'),one('dxdd'),q[0] if len(q)>0 else None,q[1] if len(q)>1 else None]

def build_protein_tokens(out:Path,device:str,max_batch_tokens:int=3000,max_batch_size:int=8)->dict:
 out.mkdir(parents=True,exist_ok=True); p=pd.read_csv(PROTEINS,dtype=str).fillna('').sort_values('protein_id').reset_index(drop=True); items=list(p[['protein_id','sequence']].itertuples(index=False,name=None))
 model=load_esmc_model_cached('esmc_600m',device); pad=model.tokenizer.pad_token_id; bos=model.tokenizer.bos_token_id; eos=model.tokenizer.eos_token_id
 feats=np.zeros((len(p),PROTEIN_BUDGET,1152),dtype=np.float16); mask=np.zeros((len(p),PROTEIN_BUDGET),dtype=bool); token_type=np.zeros((len(p),PROTEIN_BUDGET),dtype=np.int8); relpos=np.zeros((len(p),PROTEIN_BUDGET),dtype=np.int8); audit=[]
 batches=[]; cur=[]; mx=0
 for x in sorted(items,key=lambda z:(len(z[1]),z[0])):
  proposed=max(mx,len(x[1])+2); n=len(cur)+1
  if cur and (n>max_batch_size or proposed*n>max_batch_tokens): batches.append(cur); cur=[]; mx=0
  cur.append(x); mx=max(mx,len(x[1])+2)
 if cur: batches.append(cur)
 row_by={x:i for i,x in enumerate(p.protein_id.astype(str))}
 for bi,batch in enumerate(batches):
  seqs=[x[1] for x in batch]; tokens=model._tokenize(seqs); dev=next(model.parameters()).device; ac=torch.autocast(device_type=dev.type,dtype=torch.bfloat16) if dev.type=='cuda' else contextlib.nullcontext()
  with torch.no_grad(),ac:
   emb=model.embed(tokens); emb,_,_=model.transformer(emb,sequence_id=tokens.eq(pad))
  rmask=tokens.ne(pad)
  if bos is not None: rmask &= tokens.ne(bos)
  if eos is not None: rmask &= tokens.ne(eos)
  for j,(pid,seq) in enumerate(batch):
   residues=emb[j,rmask[j]].float();
   if len(residues)!=len(seq): raise RuntimeError(f'{pid} residue mismatch {len(residues)}!={len(seq)}')
   i=row_by[pid]; feats[i,0]=residues.mean(0).cpu().numpy().astype(np.float16); mask[i,0]=True; token_type[i,0]=0
   centers=motif_centers_for_slots(seq); available=0
   for slot,center in enumerate(centers,1):
    base=1+(slot-1)*SLOT_WIDTH; token_type[i,base:base+SLOT_WIDTH]=slot; relpos[i,base:base+SLOT_WIDTH]=np.arange(-WINDOW,WINDOW+1,dtype=np.int8)
    if center is None: continue
    available+=1
    for pos_rel in range(-WINDOW,WINDOW+1):
     pos=center+pos_rel; k=base+(pos_rel+WINDOW)
     if 0<=pos<len(seq): feats[i,k]=residues[pos].cpu().numpy().astype(np.float16); mask[i,k]=True
   audit.append({'row':i,'protein_id':pid,'length':len(seq),'ddxxd_center':centers[0] if centers[0] is not None else '', 'nse_dte_center':centers[1] if centers[1] is not None else '', 'dxdd_center':centers[2] if centers[2] is not None else '', 'qw1_center':centers[3] if centers[3] is not None else '', 'qw2_center':centers[4] if centers[4] is not None else '', 'available_motif_slots':available,'valid_tokens':int(mask[i].sum())})
  del tokens,emb,rmask
  if dev.type=='cuda': torch.cuda.empty_cache()
  if bi%20==0: print(json.dumps({'batch':bi,'of':len(batches),'proteins_done':sum(len(x) for x in batches[:bi+1])}),flush=True)
 np.save(out/'protein_tokens.npy',feats); np.save(out/'protein_token_mask.npy',mask); np.save(out/'protein_token_type.npy',token_type); np.save(out/'protein_token_relpos.npy',relpos); pd.DataFrame({'row':range(len(p)),'protein_id':p.protein_id}).to_csv(out/'protein_entries.csv',index=False); pd.DataFrame(audit).sort_values('row').to_csv(out/'protein_token_audit.csv',index=False)
 result={'protein_count':len(p),'token_budget':PROTEIN_BUDGET,'embedding_dim':1152,'window_radius':WINDOW,'slot_width':SLOT_WIDTH,'motif_slots':list(MOTIF_SLOTS),'all_global_present':bool(mask[:,0].all()),'proteins_with_any_motif':int((mask[:,1:].sum(1)>0).sum()),'labels_used':False,'storage_dtype':'float16'}
 (out/'protein_tokens_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['reaction','protein','all']); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); out=a.output_dir.resolve()
 if a.action in {'reaction','all'}: print(json.dumps(build_reaction_tokens(out),indent=2))
 if a.action in {'protein','all'}: print(json.dumps(build_protein_tokens(out,a.device),indent=2))
if __name__=='__main__': main()
