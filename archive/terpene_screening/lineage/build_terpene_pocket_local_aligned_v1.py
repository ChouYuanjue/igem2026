from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3]
PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
CAND=ROOT/'data/terpene_p2rank_current_v1/candidates.csv'
MAN=ROOT/'results/terpene_p2rank_current_v1/p2rank_pocket_manifest.csv'
OLD=ROOT/'data/terpene_embeddings/esmc600m_pocket_local'
NEW=ROOT/'data/terpene_embeddings/esmc600m_pocket_local_external_fill_v1'
OUT=ROOT/'data/terpene_pocket_local_aligned_v1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(d):
 e=pd.read_csv(d/'entries.csv',dtype=str).fillna(''); e['row']=pd.to_numeric(e.row).astype(int)
 m=np.load(d/'embeddings.npy',mmap_mode='r'); return e,m
def main():
 p=pd.read_csv(PROTEINS,dtype=str).fillna('')
 c=pd.read_csv(CAND,dtype=str).fillna('')
 man=pd.read_csv(MAN,dtype=str).fillna('')
 active=c[['UniprotID','protein_id']].merge(man[['UniprotID','status','pocket_residues']],on='UniprotID',how='inner',validate='one_to_one')
 active=active[active.status.eq('ok') & active.pocket_residues.ne('')].copy()
 if active.protein_id.duplicated().any(): raise RuntimeError('current P2Rank protein mapping nonunique')
 uid_to_pid=dict(zip(active.UniprotID.astype(str),active.protein_id.astype(str)))
 alias_to_pid={}
 for pr in p.itertuples(index=False):
  for token in re.split(r'[;,|\s]+',str(pr.aliases)):
   if token: alias_to_pid[token]=str(pr.protein_id)
 active_pids=set(active.protein_id.astype(str))
 olde,oldm=load(OLD); newe,newm=load(NEW)
 source={}; duplicates=[]
 for label,e,m in [('historical_current',olde,oldm),('completed_external',newe,newm)]:
  for r in e.itertuples(index=False):
   pid=uid_to_pid.get(str(r.Entry))
   if pid is None and label=='historical_current':
    candidate=alias_to_pid.get(str(r.Entry))
    pid=candidate if candidate in active_pids else None
   if pid is None: continue
   vec=np.asarray(m[int(r.row)],dtype=np.float32)
   if pid in source:
    prev=source[pid][1]; cos=float(np.dot(prev,vec)/(max(np.linalg.norm(prev)*np.linalg.norm(vec),1e-12)))
    duplicates.append((pid,source[pid][0],label,cos))
    if cos < 0.999: raise RuntimeError(f'conflicting pocket embedding for {pid}: {cos}')
    continue
   source[pid]=(label,vec,str(r.Entry))
 ids=p.protein_id.astype(str).tolist(); aligned=np.zeros((len(ids),oldm.shape[1]),dtype=np.float32); avail=np.zeros(len(ids),bool); src=[]
 for i,pid in enumerate(ids):
  if pid in source:
   label,vec,entry=source[pid]; aligned[i]=vec; avail[i]=True; src.append({'row':i,'protein_id':pid,'Entry':entry,'source':label})
 expected=set(active.protein_id.astype(str)); observed=set(source)
 if observed != expected:
  raise RuntimeError(f'active pocket embedding coverage mismatch missing={len(expected-observed)} extra={len(observed-expected)}')
 OUT.mkdir(parents=True,exist_ok=True); np.save(OUT/'embeddings.npy',aligned); np.save(OUT/'available.npy',avail)
 pd.DataFrame(src).to_csv(OUT/'entries.csv',index=False)
 summary={
  'version':'terpene-pocket-local-aligned-v1','protein_count':len(ids),'embedding_dimension':int(aligned.shape[1]),
  'available_count':int(avail.sum()),'available_fraction':float(avail.mean()),
  'historical_current_count':sum(v[0]=='historical_current' for v in source.values()),
  'completed_external_count':sum(v[0]=='completed_external' for v in source.values()),
  'duplicate_mapped_count':len(duplicates),
  'semantic_rule':'ESM-C mean embedding of merged +/-4-aa P2Rank top1 pocket-residue windows; missing pocket is missing observation',
  'labels_used':False,
  'sha256':{'protein_entities':sha(PROTEINS),'p2rank_candidates':sha(CAND),'p2rank_manifest':sha(MAN),'old_embeddings':sha(OLD/'embeddings.npy'),'new_embeddings':sha(NEW/'embeddings.npy')},
 }
 (OUT/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
