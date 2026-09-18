from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
AUDIT=ROOT/'data/terpene_pocket_local_aligned_v1/current_semantic_reuse_audit.csv'
OLD=ROOT/'data/terpene_embeddings/esmc600m_pocket_local'
FILL=ROOT/'data/terpene_embeddings/esmc600m_pocket_local_external_fill_v1'
GAP=ROOT/'data/terpene_embeddings/esmc600m_pocket_local_current_gap_v1'
OUT=ROOT/'data/terpene_pocket_local_aligned_v2'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load(directory: Path) -> tuple[dict[str,np.ndarray],int]:
    entries=pd.read_csv(directory/'entries.csv',dtype=str).fillna('')
    entries['row']=pd.to_numeric(entries['row']).astype(int)
    matrix=np.load(directory/'embeddings.npy',mmap_mode='r')
    return {str(r.Entry):np.asarray(matrix[int(r.row)],dtype=np.float32) for r in entries.itertuples(index=False)}, int(matrix.shape[1])


def main() -> None:
    proteins=pd.read_csv(PROTEINS,dtype=str).fillna('')
    audit=pd.read_csv(AUDIT,dtype=str).fillna('')
    for c in ['exact_old_matches','old_mismatches','old_observation_count']:
        audit[c]=pd.to_numeric(audit[c]).astype(int)
    old,dim0=load(OLD); fill,dim1=load(FILL); gap,dim2=load(GAP)
    if len({dim0,dim1,dim2})!=1: raise RuntimeError('embedding dimensions differ')
    dim=dim0
    row_by_pid={str(r.protein_id):r for r in audit.itertuples(index=False)}
    aligned=np.zeros((len(proteins),dim),dtype=np.float32)
    available=np.zeros(len(proteins),dtype=bool)
    provenance=[]
    stale=[]
    for i,pr in enumerate(proteins.itertuples(index=False)):
        pid=str(pr.protein_id)
        r=row_by_pid.get(pid)
        if r is None:
            provenance.append({'row':i,'protein_id':pid,'Entry':'','source':'missing_no_current_pocket'})
            continue
        current_uid=str(r.current_uid)
        matching=json.loads(str(r.matching_uids) or '[]')
        mismatching=json.loads(str(r.mismatching_uids) or '[]')
        stale.extend({'protein_id':pid,'current_uid':current_uid,'stale_uid':u} for u in mismatching)
        vec=None; source=''; chosen=''
        # Reuse is legal only when the historical snippet is byte-identical to the current P2Rank-derived snippet.
        if matching:
            chosen=current_uid if current_uid in matching and current_uid in old else next((u for u in matching if u in old),'')
            if not chosen: raise RuntimeError(f'exact old match has no vector: {pid} {matching}')
            vec=old[chosen]; source='historical_exact_semantic_reuse'
        elif current_uid in fill:
            chosen=current_uid; vec=fill[current_uid]; source='external_fill_current_semantics'
        elif current_uid in gap:
            chosen=current_uid; vec=gap[current_uid]; source='current_gap_recompute'
        else:
            raise RuntimeError(f'current pocket lacks canonical embedding: {pid} {current_uid}')
        aligned[i]=vec; available[i]=True
        provenance.append({'row':i,'protein_id':pid,'Entry':chosen,'current_uid':current_uid,'source':source})
    if int(available.sum())!=1287: raise RuntimeError(f'expected 1287 available, got {available.sum()}')
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(OUT/'embeddings.npy',aligned)
    np.save(OUT/'available.npy',available)
    pd.DataFrame(provenance).to_csv(OUT/'entries.csv',index=False)
    pd.DataFrame(stale).to_csv(OUT/'excluded_stale_alias_observations.csv',index=False)
    src=pd.DataFrame(provenance).source.value_counts().to_dict()
    manifest={
      'version':'terpene-pocket-local-aligned-v2',
      'semantic_role':'current-P2Rank-backed pocket-local sequence coordinate on the fixed 1,421-protein terpene manifold',
      'protein_count':int(len(proteins)),'available_count':int(available.sum()),'dimension':int(dim),
      'source_counts':{str(k):int(v) for k,v in src.items()},
      'snippet_rule':'for each current P2Rank top-1 pocket residue take +/-4 aa; merge overlapping/adjacent windows; join disjoint windows with X',
      'reuse_rule':'historical vectors are reusable only when their historical snippet is byte-identical to the snippet regenerated from the current canonical sequence and current P2Rank residues',
      'stale_alias_observation_count':int(len(stale)),
      'missing_is_negative':False,'labels_used':False,
      'sha256':{
        'protein_entities':sha(PROTEINS),'semantic_audit':sha(AUDIT),
        'historical_embeddings':sha(OLD/'embeddings.npy'),'external_fill_embeddings':sha(FILL/'embeddings.npy'),'gap_embeddings':sha(GAP/'embeddings.npy')
      }
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
