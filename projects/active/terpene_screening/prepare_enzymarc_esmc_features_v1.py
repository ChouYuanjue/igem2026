from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[3]
from projects.active.terpene_screening.extract_esmc_embeddings import batched_mean_embeddings,build_length_batches
from projects.active.terpene_screening.rank_open_world import load_esmc_model_cached,normalize_rows,load_protein_library
SUPPORT=ROOT/'results/enzymarc_open_world_v1/support'
GATE=ROOT/'results/enzymarc_open_world_v1/sequence_form_gate'
OUT=ROOT/'results/enzymarc_open_world_v1/esmc_features'
GENERAL=ROOT/'data/catalyst_candidate_universes/general_merged/proteins'
SEQ=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'

def decoy_id(parent:str,category:str)->str: return f'ENZYMARC::{parent}::{category}'

def parity(device:str)->dict:
    parents=pd.read_csv(SUPPORT/'parents.csv',dtype=str).fillna('').sort_values('parent_accession')
    lib,ids=load_protein_library(GENERAL); idx={x:i for i,x in enumerate(ids)}
    # deterministic spread over the supported parent list, no labels or model scores
    rows=np.linspace(0,len(parents)-1,12,dtype=int); sub=parents.iloc[rows]
    model=load_esmc_model_cached('esmc_600m',device)
    got=normalize_rows(batched_mean_embeddings(model,sub.parent_sequence.astype(str).tolist(),device))
    ref=np.stack([lib[idx[x]] for x in sub.parent_accession.astype(str)],axis=0)
    diff=np.abs(got-ref)
    out={'count':len(sub),'max_abs_diff':float(diff.max()),'mean_abs_diff':float(diff.mean()),'cosine_min':float(np.sum(got*ref,axis=1).min()),'ids':sub.parent_accession.astype(str).tolist()}
    return out

def materialize(device:str,out:Path,max_batch_tokens:int,max_batch_size:int)->dict:
    out.mkdir(parents=True,exist_ok=True)
    audit=parity(device)
    # Float32 registry features were historically materialized by the same batched path;
    # tolerate only tiny numerical drift from CUDA/bfloat16 execution.
    if audit['max_abs_diff']>5e-4 or audit['cosine_min']<0.99999:
        raise RuntimeError(f'ESM-C parity failed: {audit}')
    gate=json.loads((GATE/'manifest.json').read_text())
    if gate.get('status')!='sequence_form_gate_frozen': raise RuntimeError('sequence-form gate is not finalized')
    eligible=set(pd.read_csv(GATE/'eligible_parents.csv',dtype=str).parent_accession.astype(str))
    d=pd.read_csv(SUPPORT/'decoys.csv',dtype=str).fillna(''); d=d[d.parent_accession.isin(eligible)].sort_values(['parent_accession','category']).reset_index(drop=True)
    entries=pd.DataFrame({'row':np.arange(len(d),dtype=int),'Entry':[decoy_id(p,c) for p,c in zip(d.parent_accession,d.category)],'parent_accession':d.parent_accession,'category':d.category})
    entries.to_csv(out/'entries.csv',index=False)
    matrix=np.lib.format.open_memmap(out/'embeddings.npy',mode='w+',dtype=np.float16,shape=(len(d),1152))
    items=[(str(i),str(s)) for i,s in enumerate(d.decoy_sequence)]
    batches=build_length_batches(items,max_batch_tokens,max_batch_size); model=load_esmc_model_cached('esmc_600m',device); done=0
    for bi,batch in enumerate(batches):
        vectors=normalize_rows(batched_mean_embeddings(model,[s for _,s in batch],device)).astype(np.float16)
        rows=np.asarray([int(i) for i,_ in batch],dtype=int); matrix[rows]=vectors; done+=len(rows)
        if bi%200==0: matrix.flush(); print(json.dumps({'batch':bi,'of':len(batches),'done':done,'total':len(d)}),flush=True)
    matrix.flush()
    result={'status':'ready','count':len(d),'dimension':1152,'dtype':'float16','model':'esmc_600m','normalization':'L2 row normalization after mean residue embedding','max_batch_tokens':max_batch_tokens,'max_batch_size':max_batch_size,'parity':audit,'labels_used':False,'model_compatibility_scores_read':False,'sequence_form_gate_sha256':__import__('hashlib').sha256((GATE/'manifest.json').read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['parity','materialize']); ap.add_argument('--device',default='cuda'); ap.add_argument('--output-dir',type=Path,default=OUT); ap.add_argument('--max-batch-tokens',type=int,default=8192); ap.add_argument('--max-batch-size',type=int,default=32); a=ap.parse_args()
    print(json.dumps(parity(a.device) if a.action=='parity' else materialize(a.device,a.output_dir.resolve(),a.max_batch_tokens,a.max_batch_size),indent=2))
if __name__=='__main__': main()
