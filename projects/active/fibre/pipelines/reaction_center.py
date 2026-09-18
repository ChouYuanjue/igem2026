from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from archive.terpene_screening.lineage.prepare_tps_active_site_tokens_v1 import stable_mapped_reactions

TOK=ROOT/'data/terpene_tps_active_site_xattn_v1'
RXN=ROOT/'data/terpene_marts_adaptation/reaction_entities.csv'
OUT=ROOT/'data/terpene_reaction_center_observations_v1'
CHEM_DIMS=np.asarray(list(range(0,7))+list(range(10,23)),dtype=int)


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def atom_state(atom: Chem.Atom) -> tuple[int,int,int,int,int]:
    return (int(atom.GetAtomicNum()),int(atom.GetFormalCharge()),int(atom.GetTotalNumHs()),int(atom.GetIsAromatic()),int(atom.GetChiralTag()))


def bond_map(mol: Chem.Mol) -> dict[tuple[int,int],float]:
    out={}
    for b in mol.GetBonds():
        a=int(b.GetBeginAtom().GetAtomMapNum()); c=int(b.GetEndAtom().GetAtomMapNum())
        if a>0 and c>0: out[tuple(sorted((a,c)))]=float(b.GetBondTypeAsDouble())
    return out


def explicit_transition_tokens(mapped: str) -> list[str]:
    left,right=mapped.split('>>')
    a=Chem.MolFromSmiles(left); b=Chem.MolFromSmiles(right)
    if a is None or b is None: raise ValueError('unparseable mapped reaction')
    amap={int(x.GetAtomMapNum()):x for x in a.GetAtoms() if x.GetAtomMapNum()>0}
    bmap={int(x.GetAtomMapNum()):x for x in b.GetAtoms() if x.GetAtomMapNum()>0}
    ab=bond_map(a); bb=bond_map(b); tokens=[]
    for pair in sorted(set(ab)|set(bb)):
        old=float(ab.get(pair,0.0)); new=float(bb.get(pair,0.0))
        if old==new: continue
        z=[]
        for mid in pair:
            atom=bmap.get(mid) or amap.get(mid); z.append(int(atom.GetAtomicNum()) if atom is not None else 0)
        tokens.append(f'bond:{min(z)}-{max(z)}:{old:g}>{new:g}')
    for mid in sorted(set(amap)&set(bmap)):
        old=atom_state(amap[mid]); new=atom_state(bmap[mid])
        if old!=new: tokens.append(f'atom:{old}>{new}')
    return sorted(set(tokens))


def main() -> None:
    rx=pd.read_csv(RXN,dtype=str).fillna('')
    ent=pd.read_csv(TOK/'reaction_entries.csv',dtype=str).fillna(''); ent['row']=pd.to_numeric(ent.row).astype(int); ent=ent.sort_values('row')
    if ent.reaction_id.tolist()!=rx.reaction_id.tolist(): raise RuntimeError('reaction token order mismatch')
    f=np.load(TOK/'reaction_atom_features.npy').astype(np.float32)
    mask=np.load(TOK/'reaction_atom_mask.npy').astype(bool)
    amap=np.load(TOK/'reaction_atom_map.npy').astype(np.int32)
    side=np.load(TOK/'reaction_atom_side.npy').astype(np.int8)
    changed=np.load(TOK/'reaction_atom_changed.npy').astype(bool)
    mapped,ma=stable_mapped_reactions()
    if len(mapped)!=len(rx): raise RuntimeError(f'mapped coverage {len(mapped)} != {len(rx)}')

    # One empirical measure per reaction over changed atom-state transitions.
    measures=[]; token_sets=[]; audits=[]
    for i,rid in enumerate(rx.reaction_id.astype(str)):
        by={}
        valid=np.flatnonzero(mask[i]&changed[i]&(amap[i]>0))
        for k in valid:
            mid=int(amap[i,k]); s=int(side[i,k]); by.setdefault(mid,{})[s]=f[i,k,CHEM_DIMS].astype(np.float64)
        rows=[]
        for mid in sorted(by):
            before=by[mid].get(0); after=by[mid].get(1)
            z0=np.zeros(len(CHEM_DIMS),dtype=np.float64) if before is None else before
            z1=np.zeros(len(CHEM_DIMS),dtype=np.float64) if after is None else after
            rows.append(np.concatenate([z0,z1,[float(before is not None),float(after is not None)]]))
        x=np.stack(rows).astype(np.float32)
        measures.append(x)
        ts=explicit_transition_tokens(mapped[rid]); token_sets.append(ts)
        audits.append({'row':i,'reaction_id':rid,'transition_atoms':len(x),'paired_atoms':sum((0 in by[m] and 1 in by[m]) for m in by),'product_only_atoms':sum((0 not in by[m] and 1 in by[m]) for m in by),'token_count':len(ts)})
    if any(len(x)==0 for x in measures): raise RuntimeError('empty changed-atom measure')
    max_atoms=max(map(len,measures)); dim=measures[0].shape[1]
    arr=np.zeros((len(measures),max_atoms,dim),dtype=np.float32); msk=np.zeros((len(measures),max_atoms),dtype=bool)
    for i,x in enumerate(measures): arr[i,:len(x)]=x; msk[i,:len(x)]=True
    OUT.mkdir(parents=True,exist_ok=True)
    np.save(OUT/'transition_atom_states.npy',arr); np.save(OUT/'transition_atom_mask.npy',msk)
    pd.DataFrame({'row':np.arange(len(rx)),'reaction_id':rx.reaction_id.astype(str)}).to_csv(OUT/'reaction_ids.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'audit.csv',index=False)
    with (OUT/'transition_tokens.jsonl').open('w') as h:
        for rid,t in zip(rx.reaction_id.astype(str),token_sets): h.write(json.dumps({'reaction_id':rid,'tokens':t},sort_keys=True)+'\n')
    manifest={
      'version':'terpene-reaction-center-observations-v1','reaction_count':len(rx),'transition_atom_dimension':dim,'max_transition_atoms':max_atoms,
      'atom_state_definition':'concatenate reactant and product chemical atom-vector blocks [dims 0:7,10:23] from current TPS token asset plus before/after presence bits; product/changed/map-ID bookkeeping dimensions are excluded',
      'measure_definition':'uniform empirical measure over changed atom-map IDs; product-only new atoms are represented with absent-before state and presence bits',
      'transition_token_definition':'unhashed explicit changed-bond element/order transitions plus mapped atom-state transitions',
      'all_reactions_observed':True,'labels_used':False,
      'input_sha256':{'reaction_entities':sha(RXN),'reaction_atom_features':sha(TOK/'reaction_atom_features.npy'),'reaction_atom_changed':sha(TOK/'reaction_atom_changed.npy'),'reaction_mapping_audit':sha(TOK/'reaction_mapping_audit.csv')},
      'mapped_selection':'same stable_mapped_reactions() provenance used by current 453-row TPS active-site token asset; highest RXNMapper confidence within exact reaction_signature matches, tie by source reaction id',
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2)); print(pd.DataFrame(audits).describe().to_string())

if __name__=='__main__': main()
