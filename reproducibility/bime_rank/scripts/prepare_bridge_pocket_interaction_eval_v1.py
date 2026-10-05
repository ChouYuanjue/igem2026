from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd,torch,yaml
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_broad_shared_pool_v1 import aliases,cage_supported_aliases

TARGETS=ROOT/'results/bridge_relation_unseen_max_v3/targets.csv'
TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
META=ROOT/'data/catalyst_candidate_universes/general_merged/protein_metadata.csv'
SEQUENCES=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reactions.csv'
REACTION_ASSET=ROOT/'results/enzymecage_reaction_family_response_v1/full_outer_assets'
FOLDS=ROOT/'results/cleanroom_internal_full_candidate_benchmarks_v1'
OUT=ROOT/'results/bridge_pocket_interaction_eval_v1'
TOPK=1000;MAX_SCOREABLE=32;MIN_SCOREABLE=8

def write_config(out,pairs):
 cfg=yaml.safe_load((REACTION_ASSET/'probe.yaml').read_text());cfg['data_path']=str(pairs.resolve());cfg['batch_size']=256;(out/'infer.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))

def select_scoreable(index,score,cmap,limit=MAX_SCOREABLE):
 vals,inds=torch.topk(score,k=TOPK,largest=True,sorted=True);inds=inds.cpu().numpy();vals=vals.cpu().numpy();selected=[]
 for rank,(ri,bv) in enumerate(zip(inds,vals),1):
  pid=index.protein_ids[int(ri)];uid=cmap.get(pid)
  if uid:selected.append((pid,uid,rank,float(bv)))
  if len(selected)>=limit:break
 return selected

def prepare_internal(fold,index,cmap,seq,valid_rx,smiles):
 d=FOLDS/f'clean2023_internal_double_cold_fold{fold}';train=pd.read_csv(d/'train_pairs.csv',dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);test=pd.read_csv(d/'test_pairs.csv',dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);positives=test.groupby('reaction_id').protein_id.apply(lambda x:set(x.astype(str))).to_dict();known=train.groupby('reaction_id').protein_id.apply(lambda x:set(x.astype(str))).to_dict();queries=sorted(positives);rows=[];audit=[];pidrow=index.protein_index
 for st in range(0,len(queries),128):
  qs=[q for q in queries[st:st+128] if q in index.reaction_index]
  if not qs:continue
  qr=torch.as_tensor([index.reaction_index[q] for q in qs],dtype=torch.long,device=index.device)
  with torch.no_grad():scores=(index.reaction_embeddings.index_select(0,qr)@index.protein_embeddings.T).float()
  for j,q in enumerate(qs):
   s=scores[j].clone();kr=[pidrow[p] for p in known.get(q,set()) if p in pidrow]
   if kr:s[torch.as_tensor(kr,dtype=torch.long,device=index.device)]=-torch.inf
   selected=select_scoreable(index,s,cmap);ok=q in valid_rx and q in smiles;audit.append({'fold':fold,'reaction_id':q,'scoreable_candidates':len(selected),'active':int(ok and len(selected)>=MIN_SCOREABLE),'target_count':len(positives[q])})
   if not ok or len(selected)<MIN_SCOREABLE:continue
   for pid,uid,rank,bv in selected:rows.append({'fold':fold,'reaction_id':q,'protein_id':pid,'CANO_RXN_SMILES':smiles[q],'UniprotID':uid,'sequence':seq[pid],'Label':int(pid in positives[q]),'broad_score':bv,'broad_rank_top1000':rank})
 out=OUT/f'fold{fold}';out.mkdir(parents=True,exist_ok=True);pairs=pd.DataFrame(rows);pairs.to_csv(out/'pairs.csv',index=False);pd.DataFrame(audit).to_csv(out/'query_audit.csv',index=False);write_config(out,out/'pairs.csv');return {'fold':fold,'test_relations':len(test),'queries':len(queries),'active_queries':int(pd.DataFrame(audit).active.sum()),'scoring_pairs':len(pairs)}

def prepare_outer(index,cmap,seq,valid_rx,smiles):
 targets=pd.read_csv(TARGETS,dtype=str).fillna('');positives=targets.groupby('reaction_id').protein_id.apply(lambda x:set(x.astype(str))).to_dict();queries=sorted(positives);train=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);known=train.groupby('reaction_id').protein_id.apply(lambda x:set(x.astype(str))).to_dict();rows=[];audit=[];pidrow=index.protein_index
 for st in range(0,len(queries),128):
  qs=[q for q in queries[st:st+128] if q in index.reaction_index]
  if not qs:continue
  qr=torch.as_tensor([index.reaction_index[q] for q in qs],dtype=torch.long,device=index.device)
  with torch.no_grad():scores=(index.reaction_embeddings.index_select(0,qr)@index.protein_embeddings.T).float()
  for j,q in enumerate(qs):
   s=scores[j].clone();kr=[pidrow[p] for p in known.get(q,set()) if p in pidrow]
   if kr:s[torch.as_tensor(kr,dtype=torch.long,device=index.device)]=-torch.inf
   selected=select_scoreable(index,s,cmap);ok=q in valid_rx and q in smiles;audit.append({'reaction_id':q,'scoreable_candidates':len(selected),'active':int(ok and len(selected)>=MIN_SCOREABLE),'target_count':len(positives[q])})
   if not ok or len(selected)<MIN_SCOREABLE:continue
   for pid,uid,rank,bv in selected:rows.append({'reaction_id':q,'protein_id':pid,'CANO_RXN_SMILES':smiles[q],'UniprotID':uid,'sequence':seq[pid],'Label':0,'broad_score':bv,'broad_rank_top1000':rank})
 out=OUT/'outer_max';out.mkdir(parents=True,exist_ok=True);pairs=pd.DataFrame(rows);pairs.to_csv(out/'pairs.csv',index=False);pd.DataFrame(audit).to_csv(out/'query_audit.csv',index=False);write_config(out,out/'pairs.csv');af=pd.DataFrame(audit);return {'queries':len(queries),'active_queries':int(af.active.sum()),'active_fraction':float(af.active.mean()),'scoring_pairs':len(pairs)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);seqf=pd.read_csv(SEQUENCES,sep='\t',dtype=str).fillna('');seq=dict(zip(seqf.protein_id.astype(str),seqf.sequence.astype(str)));meta=pd.read_csv(META,dtype=str).fillna('');cmap=cage_supported_aliases(aliases(meta),seq)[0];valid_rx=set(pd.read_csv(REACTION_ASSET/'probe_strict.csv',usecols=['reaction_id'],dtype=str).reaction_id.astype(str));rx=pd.read_csv(REACTIONS,dtype=str).fillna('');smiles=dict(zip(rx.reaction_id.astype(str),rx.reaction_smiles.astype(str)));index=FibreCandidateIndex(device=a.device);result={'schema':'bridge-pocket-interaction-eval-prep-v1','candidate_policy':'masked Broad Top1000 then first 32 CAGE-native-scoreable candidates; labels used only after candidate selection','internal':[prepare_internal(f,index,cmap,seq,valid_rx,smiles) for f in (0,1,2)],'outer':prepare_outer(index,cmap,seq,valid_rx,smiles)};(OUT/'prepare_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
