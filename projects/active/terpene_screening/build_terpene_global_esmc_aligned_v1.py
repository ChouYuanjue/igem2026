from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
PROTEINS=ROOT/'data/terpene_marts_adaptation/protein_entities.csv'
OLD=ROOT/'data/terpene_embeddings/esmc600m_global_combined'
FILL=ROOT/'data/terpene_embeddings/esmc600m_global_fill_v1'
OUT=ROOT/'data/terpene_global_esmc_aligned_v1'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def load(directory: Path):
    e=pd.read_csv(directory/'entries.csv',dtype=str).fillna('')
    e['row']=pd.to_numeric(e['row']).astype(int)
    x=np.load(directory/'embeddings.npy',mmap_mode='r')
    return e,x


def main() -> None:
    p=pd.read_csv(PROTEINS,dtype=str).fillna('')
    if p.protein_id.duplicated().any(): raise RuntimeError('protein_id must be unique')
    alias={}
    for r in p.itertuples(index=False):
        for a in re.split(r'[;,|\s]+',str(r.aliases)):
            if a:
                previous=alias.get(a)
                if previous is not None and previous != str(r.protein_id):
                    raise RuntimeError(f'alias collision {a}: {previous} vs {r.protein_id}')
                alias[a]=str(r.protein_id)
    olde,oldx=load(OLD); fille,fillx=load(FILL)
    if oldx.shape[1]!=fillx.shape[1]: raise RuntimeError('embedding dimensions differ')
    dim=int(oldx.shape[1])
    by_pid={}
    duplicate_audit=[]
    # Historical accessions may have sequence-identical aliases. Preserve exactly one canonical coordinate only after numerical agreement.
    for r in olde.itertuples(index=False):
        pid=alias.get(str(r.Entry))
        if pid is None: continue
        vec=np.asarray(oldx[int(r.row)],dtype=np.float32)
        if pid in by_pid:
            prev=by_pid[pid]['vec']
            cos=float(np.dot(prev,vec)/max(float(np.linalg.norm(prev)*np.linalg.norm(vec)),1e-12))
            duplicate_audit.append({'protein_id':pid,'first_entry':by_pid[pid]['Entry'],'duplicate_entry':str(r.Entry),'cosine':cos})
            if cos < 0.99999: raise RuntimeError(f'global alias embeddings disagree for {pid}: {cos}')
            continue
        by_pid[pid]={'vec':vec,'Entry':str(r.Entry),'source':'historical_aligned'}
    for r in fille.itertuples(index=False):
        pid=str(r.Entry)
        if pid not in set(p.protein_id.astype(str)): raise RuntimeError(f'fill id not canonical: {pid}')
        if pid in by_pid: raise RuntimeError(f'fill unexpectedly overlaps historical coordinate: {pid}')
        by_pid[pid]={'vec':np.asarray(fillx[int(r.row)],dtype=np.float32),'Entry':pid,'source':'sequence_fill'}
    missing=set(p.protein_id.astype(str))-set(by_pid)
    extra=set(by_pid)-set(p.protein_id.astype(str))
    if missing or extra: raise RuntimeError(f'alignment mismatch missing={sorted(missing)} extra={sorted(extra)}')
    aligned=np.zeros((len(p),dim),dtype=np.float32)
    rows=[]
    for i,r in enumerate(p.itertuples(index=False)):
        item=by_pid[str(r.protein_id)]
        aligned[i]=item['vec']
        rows.append({'row':i,'protein_id':str(r.protein_id),'Entry':item['Entry'],'source':item['source']})
    norms=np.linalg.norm(aligned,axis=1)
    if np.any(norms<=0): raise RuntimeError('zero global embedding found')
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(OUT/'embeddings.npy',aligned)
    np.save(OUT/'available.npy',np.ones(len(p),dtype=bool))
    pd.DataFrame(rows).to_csv(OUT/'entries.csv',index=False)
    pd.DataFrame(duplicate_audit).to_csv(OUT/'duplicate_alias_audit.csv',index=False)
    manifest={
      'version':'terpene-global-esmc-aligned-v1','protein_count':int(len(p)),'available_count':int(len(p)),'dimension':dim,
      'source_counts':{str(k):int(v) for k,v in pd.DataFrame(rows).source.value_counts().to_dict().items()},
      'duplicate_alias_count':int(len(duplicate_audit)),
      'duplicate_alias_min_cosine':float(min((x['cosine'] for x in duplicate_audit),default=1.0)),
      'semantic_role':'complete base sequence coordinate for the fixed 1,421-protein terpene manifold',
      'labels_used':False,
      'sha256':{'protein_entities':sha(PROTEINS),'historical_embeddings':sha(OLD/'embeddings.npy'),'fill_embeddings':sha(FILL/'embeddings.npy')}
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
