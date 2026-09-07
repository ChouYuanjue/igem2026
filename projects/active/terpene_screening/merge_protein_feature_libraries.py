from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load(path: Path):
    e=pd.read_csv(path/'entries.csv',dtype=str).fillna(''); ids=[c for c in ['Entry','protein_id'] if c in e]
    if len(ids)!=1: raise ValueError(f"expected one ID column under {path}; got {ids}")
    e['row']=pd.to_numeric(e['row'],errors='raise').astype(int); e=e.sort_values('row').reset_index(drop=True)
    x=np.load(path/'embeddings.npy',mmap_mode='r')
    if len(e)!=len(x): raise ValueError(f"entries/matrix mismatch under {path}")
    if e[ids[0]].duplicated().any(): raise ValueError(f"duplicate IDs under {path}")
    return e.rename(columns={ids[0]:'Entry'}),x

def main():
    p=argparse.ArgumentParser(description='Merge disjoint protein feature libraries into a requested registered ID order without imputation.')
    p.add_argument('--library',type=Path,action='append',required=True)
    p.add_argument('--target-entries',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--chunk-size',type=int,default=4096)
    a=p.parse_args(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    target=pd.read_csv(a.target_entries.resolve(),dtype=str).fillna(''); tc=[c for c in ['Entry','protein_id'] if c in target]
    if len(tc)!=1: raise ValueError('target entries require exactly one protein ID column')
    if 'row' in target: target['row']=pd.to_numeric(target['row'],errors='raise').astype(int); target=target.sort_values('row').reset_index(drop=True)
    ids=target[tc[0]].astype(str).tolist();
    if len(ids)!=len(set(ids)): raise ValueError('duplicate target IDs')
    loaded=[]; loc={}; dim=None
    for li,path in enumerate(a.library):
        e,x=load(path.resolve());
        if dim is None: dim=int(x.shape[1])
        elif dim!=int(x.shape[1]): raise ValueError('feature dimensions differ')
        for row,pid in enumerate(e.Entry.astype(str)):
            if pid in loc: raise ValueError(f'protein appears in multiple libraries: {pid}')
            loc[pid]=(li,row)
        loaded.append((path.resolve(),e,x))
    missing=[pid for pid in ids if pid not in loc]
    extra=set(loc)-set(ids)
    if missing or extra: raise ValueError(f'coverage mismatch missing={len(missing)} extra={len(extra)} examples={missing[:5]}/{sorted(extra)[:5]}')
    matrix=np.lib.format.open_memmap(out/'embeddings.npy',mode='w+',dtype=np.float32,shape=(len(ids),dim))
    for start in range(0,len(ids),a.chunk_size):
        stop=min(start+a.chunk_size,len(ids))
        for i,pid in enumerate(ids[start:stop],start):
            li,row=loc[pid]; matrix[i]=np.asarray(loaded[li][2][row],dtype=np.float32)
    matrix.flush(); pd.DataFrame({'row':np.arange(len(ids),dtype=np.int64),'Entry':ids}).to_csv(out/'entries.csv',index=False)
    source_provenance=[]
    for path, entries, values in loaded:
        manifest_path=path/'manifest.json'
        source_provenance.append({
            'path':str(path),
            'protein_count':int(len(entries)),
            'feature_dimension':int(values.shape[1]),
            'entries_sha256':sha256_file(path/'entries.csv'),
            'manifest_sha256':sha256_file(manifest_path) if manifest_path.is_file() else None,
            'manifest':json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else None,
        })
    manifest={
        'version':'deterministic-protein-feature-union-v2',
        'protein_count':len(ids),
        'feature_dimension':dim,
        'target_entries':str(a.target_entries.resolve()),
        'target_entries_sha256':sha256_file(a.target_entries.resolve()),
        'libraries':[str(x[0]) for x in loaded],
        'source_provenance':source_provenance,
        'coverage_complete':True,
        'imputation':'none',
        'target_order_preserved':True,
        'merge_semantics':'exact float32 row copy from disjoint source libraries; no renormalization or transformation',
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
