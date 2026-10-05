from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd,torch,yaml
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex
from reproducibility.bime_rank.scripts.evaluate_fibre_cage_broad_shared_pool_v1 import aliases,cage_supported_aliases

TRAIN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
META=ROOT/'data/catalyst_candidate_universes/general_merged/protein_metadata.csv'
SEQUENCES=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
REACTIONS=ROOT/'data/catalyst_candidate_universes/general_merged/reactions.csv'
REACTION_ASSET=ROOT/'results/enzymecage_reaction_family_response_v1/full_outer_assets'
FOLDS=ROOT/'results/cleanroom_internal_full_candidate_benchmarks_v1'
OUT=ROOT/'results/bridge_pocket_interaction_train_v2'
NEG_PER_QUERY=32

def fold_membership():
 m=defaultdict(set)
 for f in (0,1,2):
  x=pd.read_csv(FOLDS/f'clean2023_internal_double_cold_fold{f}/test_pairs.csv',dtype=str).fillna('')
  for p,r in x[['protein_id','reaction_id']].itertuples(index=False):m[(str(p),str(r))].add(f)
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');a=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 pos=pd.read_csv(TRAIN,dtype=str).fillna('').drop_duplicates(['protein_id','reaction_id']);positive_by_r=pos.groupby('reaction_id').protein_id.apply(lambda s:set(map(str,s))).to_dict();fm=fold_membership()
 seqf=pd.read_csv(SEQUENCES,sep='\t',dtype=str).fillna('');seq=dict(zip(seqf.protein_id.astype(str),seqf.sequence.astype(str)));meta=pd.read_csv(META,dtype=str).fillna('');cmap=cage_supported_aliases(aliases(meta),seq)[0];supported=set(cmap)
 valid_rx=set(pd.read_csv(REACTION_ASSET/'probe_strict.csv',usecols=['reaction_id'],dtype=str).reaction_id.astype(str));rx=pd.read_csv(REACTIONS,dtype=str).fillna('');smiles=dict(zip(rx.reaction_id.astype(str),rx.reaction_smiles.astype(str)))
 index=FibreCandidateIndex(device=a.device);pidrow=index.protein_index;queries=sorted(set(pos.reaction_id)&valid_rx&set(smiles)&set(index.reaction_index));rows=[];audit=[]
 for st in range(0,len(queries),128):
  qs=queries[st:st+128];qr=torch.as_tensor([index.reaction_index[q] for q in qs],dtype=torch.long,device=index.device)
  with torch.no_grad():score=(index.reaction_embeddings.index_select(0,qr)@index.protein_embeddings.T).float()
  for j,q in enumerate(qs):
   s=score[j];known=positive_by_r.get(q,set());supported_pos=[p for p in known if p in supported and p in pidrow]
   # all supported positives, regardless of Broad rank
   for p in sorted(supported_pos):
    ri=pidrow[p];sc=float(s[ri].item());folds=','.join(map(str,sorted(fm.get((p,q),set()))))
    rows.append({'reaction_id':q,'protein_id':p,'CANO_RXN_SMILES':smiles[q],'UniprotID':cmap[p],'sequence':seq[p],'Label':1,'broad_score':sc,'broad_rank_top1000':0,'heldout_folds':folds})
   # Broad-hard scoreable negatives, labels selected only from clean2023.
   vals,inds=torch.topk(s,k=min(5000,len(index.protein_ids)),largest=True,sorted=True);inds=inds.cpu().numpy();vals=vals.cpu().numpy();neg=[]
   for rank,(ri,bv) in enumerate(zip(inds,vals),1):
    p=index.protein_ids[int(ri)]
    if p in known or p not in supported:continue
    neg.append((p,rank,float(bv)))
    if len(neg)>=NEG_PER_QUERY:break
   for p,rank,bv in neg:rows.append({'reaction_id':q,'protein_id':p,'CANO_RXN_SMILES':smiles[q],'UniprotID':cmap[p],'sequence':seq[p],'Label':0,'broad_score':bv,'broad_rank_top1000':rank,'heldout_folds':''})
   audit.append({'reaction_id':q,'positive_rows':len(supported_pos),'negative_rows':len(neg)})
  print('train-pair-prep',min(st+128,len(queries)),'/',len(queries),flush=True)
 frame=pd.DataFrame(rows).drop_duplicates(['reaction_id','protein_id']);frame.to_csv(OUT/'pairs.csv',index=False);pd.DataFrame(audit).to_csv(OUT/'query_audit.csv',index=False)
 cfg=yaml.safe_load((REACTION_ASSET/'probe.yaml').read_text());cfg['data_path']=str((OUT/'pairs.csv').resolve());cfg['batch_size']=256;(OUT/'infer.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 summary={'schema':'bridge-pocket-interaction-train-v2-prep','positive_source':'all clean2023 relations with native pocket support','negative_source':f'Broad-hard CAGE-scoreable candidates absent from clean2023 positives, top {NEG_PER_QUERY} per reaction','queries':int(frame.reaction_id.nunique()),'rows':len(frame),'positive_rows':int((frame.Label==1).sum()),'negative_rows':int((frame.Label==0).sum()),'heldout_fold_membership_recorded_on_positive_relations':True,'outer_relations_used':False};(OUT/'prepare_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
