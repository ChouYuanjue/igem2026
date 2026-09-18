from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from projects.active.fibre.evidence.mechanism import (
    TYPE_I_ASPARTATE, NSE, DTE, DXDD, QW,
)
from reproducibility.bime_rank.scripts.extract_esmc_motif_context_embeddings import build_length_batches
from projects.active.fibre.runtime.cli import load_esmc_model_cached, _ESMC_MODEL_SOURCE, _ESMC_LOCAL_SPECS

DEFAULT_INPUT=ROOT/'data/terpene_marts_adaptation/protein_architecture_annotations.csv'
DEFAULT_OUTPUT=ROOT/'data/terpene_family_aware_motif_coordinates_v1'
VALID_AA=re.compile(r'^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$')
VIEW_NAMES=('typeI_aspartate','nse_dte','dxdd','qw')


def clean_sequence(x: object) -> str:
    return ''.join(str(x).upper().split()).rstrip('*')


def applicable(family: str, view: str) -> bool:
    f=str(family or '')
    has_class_i='classI' in f
    has_class_ii='plant_like_classI_II' in f or 'triterpene_cyclase' in f
    if view in {'typeI_aspartate','nse_dte'}: return has_class_i
    if view=='dxdd': return has_class_ii
    if view=='qw': return 'triterpene_cyclase' in f
    return False


def spans(sequence: str, view: str) -> list[tuple[int,int]]:
    pats=[]
    if view=='typeI_aspartate': pats=[TYPE_I_ASPARTATE]
    elif view=='nse_dte': pats=[NSE,DTE]
    elif view=='dxdd': pats=[DXDD]
    elif view=='qw': pats=[QW]
    out=[]
    for pat in pats:
        out.extend((m.start(),m.end()) for m in pat.finditer(sequence))
    return sorted(set(out))


def normalized_mean_occurrences(residue_embeddings: torch.Tensor, locs: list[tuple[int,int]]) -> torch.Tensor:
    occ=[]
    for a,b in locs:
        v=residue_embeddings[a:b].float().mean(dim=0)
        n=torch.linalg.vector_norm(v)
        if float(n)>0: v=v/n
        occ.append(v)
    if not occ: return torch.zeros(residue_embeddings.shape[-1],device=residue_embeddings.device,dtype=torch.float32)
    v=torch.stack(occ).mean(dim=0)
    n=torch.linalg.vector_norm(v)
    return v/n if float(n)>0 else v


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--model',default='esmc_600m')
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-batch-tokens',type=int,default=4096)
    ap.add_argument('--max-batch-size',type=int,default=16)
    a=ap.parse_args()
    d=pd.read_csv(a.input,dtype=str).fillna('')
    if d.protein_id.duplicated().any(): raise RuntimeError('protein_id must be unique')
    d['sequence']=d.sequence.map(clean_sequence)
    bad=d[~d.sequence.map(lambda s:bool(s) and bool(VALID_AA.fullmatch(s)))]
    if len(bad): raise RuntimeError(f'invalid sequences: {bad.protein_id.tolist()[:10]}')
    items=[(str(r.protein_id),str(r.sequence)) for r in d.itertuples(index=False)]
    model=load_esmc_model_cached(a.model,a.device)
    spec=_ESMC_LOCAL_SPECS.get(a.model)
    if spec is None: raise RuntimeError(f'unknown ESM-C local spec: {a.model}')
    dim=int(spec['d_model'])
    n=len(d)
    matrices={name:np.zeros((n,dim),dtype=np.float32) for name in VIEW_NAMES}
    availability={name:np.zeros(n,dtype=bool) for name in VIEW_NAMES}
    audits=[]
    index={str(pid):i for i,pid in enumerate(d.protein_id.astype(str))}
    family=dict(zip(d.protein_id.astype(str),d.domain_family.astype(str)))
    batches=build_length_batches(items,a.max_batch_tokens,a.max_batch_size)
    for batch in tqdm(batches,desc='family-aware motif coordinates'):
        seqs=[s for _,s in batch]
        tokens=model._tokenize(seqs)
        pad=model.tokenizer.pad_token_id; bos=model.tokenizer.bos_token_id; eos=model.tokenizer.eos_token_id
        device=next(model.parameters()).device
        ctx=torch.autocast(device_type=device.type,dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
        with torch.no_grad(),ctx:
            emb=model.embed(tokens)
            emb,_,_=model.transformer(emb,sequence_id=tokens.eq(pad))
        mask=tokens.ne(pad)
        if bos is not None: mask &= tokens.ne(bos)
        if eos is not None: mask &= tokens.ne(eos)
        for bi,(pid,seq) in enumerate(batch):
            residues=emb[bi,mask[bi]]
            if len(residues)!=len(seq): raise RuntimeError(f'token/residue mismatch {pid}')
            rec={'protein_id':pid,'domain_family':family[pid]}
            for name in VIEW_NAMES:
                locs=spans(seq,name) if applicable(family[pid],name) else []
                rec[f'{name}_applicable']=bool(applicable(family[pid],name))
                rec[f'{name}_occurrences']=len(locs)
                if locs:
                    i=index[pid]; matrices[name][i]=normalized_mean_occurrences(residues,locs).cpu().numpy().astype(np.float32); availability[name][i]=True
            audits.append(rec)
        del tokens,emb,mask
        if device.type=='cuda': torch.cuda.empty_cache()
    out=a.output; out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row':np.arange(n),'protein_id':d.protein_id.astype(str)}).to_csv(out/'protein_ids.csv',index=False)
    for name in VIEW_NAMES:
        np.save(out/f'{name}_embeddings.npy',matrices[name]); np.save(out/f'{name}_available.npy',availability[name])
    audit=pd.DataFrame(audits).sort_values('protein_id'); audit.to_csv(out/'audit.csv',index=False)
    manifest={
      'version':'terpene-family-aware-motif-coordinates-v1','protein_count':n,'dimension':dim,
      'coordinate_definition':'contextual ESM-C residue embeddings averaged over each regex match span; each occurrence is L2-normalized, occurrences are equally averaged and renormalized; no flanking window parameter',
      'views':{
        'typeI_aspartate':{'applicability':'domain_family contains classI','pattern':'DD.{2,3}[DE]'},
        'nse_dte':{'applicability':'domain_family contains classI','patterns':['[ND]D..[ST]...E','DTE']},
        'dxdd':{'applicability':'plant_like_classI_II or triterpene_cyclase','pattern':'D.DD'},
        'qw':{'applicability':'triterpene_cyclase','pattern':'QW'},
      },
      'available_counts':{name:int(availability[name].sum()) for name in VIEW_NAMES},
      'family_annotated_count':int(d.domain_family.ne('').sum()),
      'missing_policy':'not applicable or unobserved motif means missing coordinate, never a zero-valued biological observation or negative evidence',
      'labels_used':False,'model':a.model,'model_source':_ESMC_MODEL_SOURCE.get((a.model,a.device),'unknown'),
      'input_sha256':sha(a.input),
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
