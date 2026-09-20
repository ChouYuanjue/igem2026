from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import ot
from rdkit import Chem, rdBase
from scipy.spatial.distance import cdist


CHEM_DIMS=np.asarray(list(range(0,7))+list(range(10,23)),dtype=int)
ATOM_DIM=23


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def atom_state(atom: Chem.Atom) -> tuple[int,int,int,int,int]:
    return (
        int(atom.GetAtomicNum()),
        int(atom.GetFormalCharge()),
        int(atom.GetTotalNumHs()),
        int(atom.GetIsAromatic()),
        int(atom.GetChiralTag()),
    )


def atom_vector(atom: Chem.Atom, *, product: bool, changed: bool) -> np.ndarray:
    """The same 23-dimensional atom state used by the current FIBRE TPS asset."""
    v=[]
    v += [
        atom.GetAtomicNum()/100.0,
        min(atom.GetDegree(),6)/6.0,
        max(-4,min(4,atom.GetFormalCharge()))/4.0,
        min(atom.GetTotalNumHs(),4)/4.0,
    ]
    v += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        int(atom.GetChiralTag())/4.0,
        float(product),
        float(changed),
        float(atom.GetAtomMapNum()>0),
    ]
    hyb=str(atom.GetHybridization())
    hs=["SP","SP2","SP3","SP3D","SP3D2"]
    v += [float(hyb==x) for x in hs]+[float(hyb not in hs)]
    z=atom.GetAtomicNum()
    classes=[
        z==6,z==7,z==8,z==15,z==16,z in {9,17,35,53},
        z not in {6,7,8,9,15,16,17,35,53},
    ]
    v += [float(x) for x in classes]
    out=np.asarray(v,dtype=np.float32)
    if out.shape!=(ATOM_DIM,):
        raise RuntimeError(f"unexpected atom feature dimension {out.shape}")
    return out


def _mapped_atoms(mol: Chem.Mol) -> dict[int,Chem.Atom]:
    out={}
    for atom in mol.GetAtoms():
        mid=int(atom.GetAtomMapNum())
        if mid<=0:
            continue
        if mid in out:
            raise ValueError(f"duplicate atom-map id {mid}")
        out[mid]=atom
    return out


def _bonds(mol: Chem.Mol) -> dict[tuple[int,int],float]:
    out={}
    for bond in mol.GetBonds():
        a=int(bond.GetBeginAtom().GetAtomMapNum())
        b=int(bond.GetEndAtom().GetAtomMapNum())
        if a>0 and b>0:
            out[tuple(sorted((a,b)))]=float(bond.GetBondTypeAsDouble())
    return out


def parse_mapped_reaction(mapped: str) -> tuple[Chem.Mol,Chem.Mol]:
    if mapped.count(">>")!=1:
        raise ValueError("mapped reaction must contain exactly one >>")
    left,right=mapped.split(">>")
    a=Chem.MolFromSmiles(left)
    b=Chem.MolFromSmiles(right)
    if a is None or b is None:
        raise ValueError("mapped reaction is not parseable by RDKit")
    if not _mapped_atoms(a) or not _mapped_atoms(b):
        raise ValueError("mapped reaction contains no atom-map IDs")
    return a,b


def changed_map_ids(mapped: str) -> set[int]:
    a,b=parse_mapped_reaction(mapped)
    amap=_mapped_atoms(a); bmap=_mapped_atoms(b)
    ab=_bonds(a); bb=_bonds(b)
    changed=set()
    for pair in set(ab)|set(bb):
        if ab.get(pair,0.0)!=bb.get(pair,0.0):
            changed.update(pair)
    for mid in set(amap)&set(bmap):
        if atom_state(amap[mid])!=atom_state(bmap[mid]):
            changed.add(mid)
    return changed


def explicit_transition_tokens(mapped: str) -> list[str]:
    a,b=parse_mapped_reaction(mapped)
    amap=_mapped_atoms(a); bmap=_mapped_atoms(b)
    ab=_bonds(a); bb=_bonds(b)
    tokens=[]
    for pair in sorted(set(ab)|set(bb)):
        old=float(ab.get(pair,0.0)); new=float(bb.get(pair,0.0))
        if old==new:
            continue
        z=[]
        for mid in pair:
            atom=bmap.get(mid) or amap.get(mid)
            z.append(int(atom.GetAtomicNum()) if atom is not None else 0)
        tokens.append(f"bond:{min(z)}-{max(z)}:{old:g}>{new:g}")
    for mid in sorted(set(amap)&set(bmap)):
        old=atom_state(amap[mid]); new=atom_state(bmap[mid])
        if old!=new:
            tokens.append(f"atom:{old}>{new}")
    return sorted(set(tokens))


def transition_measure(mapped: str) -> tuple[np.ndarray,list[str]]:
    a,b=parse_mapped_reaction(mapped)
    amap=_mapped_atoms(a); bmap=_mapped_atoms(b)
    changed=changed_map_ids(mapped)
    rows=[]
    for mid in sorted(changed):
        before=amap.get(mid)
        after=bmap.get(mid)
        if before is None and after is None:
            continue
        z0=(
            np.zeros(len(CHEM_DIMS),dtype=np.float32)
            if before is None
            else atom_vector(before,product=False,changed=True)[CHEM_DIMS]
        )
        z1=(
            np.zeros(len(CHEM_DIMS),dtype=np.float32)
            if after is None
            else atom_vector(after,product=True,changed=True)[CHEM_DIMS]
        )
        rows.append(np.concatenate([
            z0,z1,
            np.asarray([float(before is not None),float(after is not None)],dtype=np.float32),
        ]))
    if not rows:
        raise ValueError("mapped reaction has no changed mapped atom state")
    return np.stack(rows).astype(np.float32),explicit_transition_tokens(mapped)


def wasserstein_distance(x: np.ndarray, y: np.ndarray) -> float:
    a=np.full(len(x),1.0/len(x),dtype=np.float64)
    b=np.full(len(y),1.0/len(y),dtype=np.float64)
    cost=cdist(np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64),metric="sqeuclidean")
    return float(np.sqrt(max(float(ot.emd2(a,b,cost)),0.0)))


def token_jaccard(a: set[str], b: set[str]) -> float:
    union=a|b
    return float(1.0-(len(a&b)/len(union) if union else 1.0))


def load_mapped_registry(reactions: pd.DataFrame, mapped_path: str | Path) -> pd.DataFrame:
    mapped=pd.read_csv(mapped_path,dtype=str).fillna("")
    if "reaction_id" not in mapped.columns:
        raise ValueError("mapped reaction table requires reaction_id")
    mapped_column=next(
        (x for x in ("mapped_rxn","mapped_reaction","mapped_reaction_smiles") if x in mapped.columns),
        None,
    )
    if mapped_column is None:
        raise ValueError("mapped reaction table requires mapped_rxn or mapped_reaction")
    if mapped.reaction_id.duplicated().any():
        raise ValueError("mapped reaction table has duplicate reaction_id")
    if "success" in mapped.columns:
        success=mapped.success.astype(str).str.lower().isin({"true","1","yes"})
        mapped=mapped[success].copy()
    source=reactions[["reaction_id"]].merge(
        mapped[["reaction_id",mapped_column]],
        on="reaction_id",how="left",validate="one_to_one",
    )
    return source.rename(columns={mapped_column:"mapped_reaction"}).fillna("")


def build_reaction_center_views(
    reactions_path: str | Path,
    mapped_reactions_path: str | Path,
    output_dir: str | Path,
) -> dict:
    reactions=pd.read_csv(reactions_path,dtype=str).fillna("")
    if "reaction_id" not in reactions.columns or reactions.reaction_id.duplicated().any():
        raise ValueError("reactions requires unique reaction_id")
    registry=load_mapped_registry(reactions,mapped_reactions_path)
    n=len(reactions)
    measures=[None]*n
    tokens=[None]*n
    audit=[]
    for i,row in enumerate(registry.itertuples(index=False)):
        rid=str(row.reaction_id); mapped=str(row.mapped_reaction).strip()
        if not mapped:
            audit.append({"row":i,"reaction_id":rid,"status":"mapping_unavailable","error":""})
            continue
        try:
            measure,token_list=transition_measure(mapped)
            measures[i]=measure
            tokens[i]=set(token_list)
            audit.append({
                "row":i,"reaction_id":rid,"status":"ok","error":"",
                "transition_atoms":int(len(measure)),"token_count":int(len(token_list)),
            })
        except Exception as exc:
            audit.append({
                "row":i,"reaction_id":rid,"status":"unavailable",
                "error":f"{type(exc).__name__}: {exc}",
            })

    available=np.asarray([x is not None for x in measures],dtype=bool)
    if int(available.sum())<2:
        raise ValueError("reaction-center geometry requires at least two mapped reactions with changed atoms")
    wd=np.full((n,n),np.inf,dtype=np.float32)
    jd=np.full((n,n),np.inf,dtype=np.float32)
    np.fill_diagonal(wd,0.0); np.fill_diagonal(jd,0.0)
    ids=np.flatnonzero(available)
    for pos,i in enumerate(ids):
        for j in ids[:pos]:
            w=wasserstein_distance(measures[int(i)],measures[int(j)])
            t=token_jaccard(tokens[int(i)],tokens[int(j)])
            wd[i,j]=wd[j,i]=w
            jd[i,j]=jd[j,i]=t

    max_atoms=max(len(measures[int(i)]) for i in ids)
    dim=int(measures[int(ids[0])].shape[1])
    states=np.zeros((n,max_atoms,dim),dtype=np.float32)
    state_mask=np.zeros((n,max_atoms),dtype=bool)
    for i in ids:
        x=measures[int(i)]
        states[i,:len(x)]=x
        state_mask[i,:len(x)]=True

    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({
        "row":np.arange(n,dtype=int),
        "reaction_id":reactions.reaction_id.astype(str),
    }).to_csv(out/"reaction_ids.csv",index=False)
    pd.DataFrame(audit).to_csv(out/"audit.csv",index=False)
    np.save(out/"transition_atom_states.npy",states)
    np.save(out/"transition_atom_mask.npy",state_mask)
    np.save(out/"center_wasserstein_distance.npy",wd)
    np.save(out/"center_token_jaccard_distance.npy",jd)
    np.save(out/"center_available.npy",available)
    with (out/"transition_tokens.jsonl").open("w",encoding="utf-8") as fh:
        for rid,value in zip(reactions.reaction_id.astype(str),tokens):
            fh.write(json.dumps({
                "reaction_id":rid,
                "tokens":sorted(value) if value is not None else [],
                "available":value is not None,
            },sort_keys=True)+"\n")

    manifest={
        "schema":"fibre-portable-reaction-center-v1",
        "scientific_role":(
            "label-free catalytic-local reaction coordinate rebuilt from atom-mapped reactions; "
            "exact balanced W2 on equal-mass changed-atom transition states plus Jaccard distance "
            "on explicit changed-bond/changed-atom transition tokens"
        ),
        "reaction_count":int(n),
        "available_count":int(available.sum()),
        "unavailable_count":int((~available).sum()),
        "transition_atom_dimension":dim,
        "max_transition_atoms":int(max_atoms),
        "labels_used":False,
        "missing_policy":"failed/unavailable mapping is a missing coordinate, never negative evidence",
        "mapping_contract":(
            "mapped reaction SMILES are an upstream pinned preprocessing artifact; "
            "RXNMapper confidence is provenance and is not used as a geometric weight"
        ),
        "rdkit_version":str(rdBase.rdkitVersion),
        "pot_version":str(getattr(ot,"__version__","unknown")),
        "input_sha256":{
            "reactions":sha256_file(reactions_path),
            "mapped_reactions":sha256_file(mapped_reactions_path),
        },
        "outputs":{
            "wasserstein_distance":"center_wasserstein_distance.npy",
            "token_jaccard_distance":"center_token_jaccard_distance.npy",
            "availability":"center_available.npy",
            "audit":"audit.csv",
        },
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return manifest


def main() -> None:
    ap=argparse.ArgumentParser(
        description="Rebuild FIBRE reaction-center distance coordinates from a mapped reaction registry."
    )
    ap.add_argument("--reactions",type=Path,required=True)
    ap.add_argument("--mapped-reactions",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(build_reaction_center_views(
        args.reactions,args.mapped_reactions,args.output_dir
    ),indent=2))


if __name__=="__main__":
    main()
