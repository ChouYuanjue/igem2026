from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]

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
    manifest={'version':'deterministic-protein-feature-union-v1','protein_count':len(ids),'feature_dimension':dim,'target_entries':str(a.target_entries.resolve()),'libraries':[str(x[0]) for x in loaded],'coverage_complete':True,'imputation':'none','target_order_preserved':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
