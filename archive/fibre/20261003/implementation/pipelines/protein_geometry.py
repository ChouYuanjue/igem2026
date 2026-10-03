from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.geometry.multiscale import (
    diffusion_conformal_affinity,
    one_step_diffusion_distance,
    partial_observation_pullback_affinity,
)

DEFAULT_PROTEINS = ROOT / 'data/terpene_marts_adaptation/protein_entities.csv'
DEFAULT_GLOBAL = ROOT / 'data/terpene_global_esmc_aligned_v1'
DEFAULT_STRUCTURAL = ROOT / 'data/terpene_structural_observations_v1'
DEFAULT_POCKET_LOCAL = ROOT / 'data/terpene_pocket_local_aligned_v2'
DEFAULT_MOTIFS = ROOT / 'data/terpene_family_aware_motif_coordinates_v1'
DEFAULT_OUTPUT = ROOT / 'data/terpene_multiresolution_protein_geometry_v4'


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def _alias_map(proteins: pd.DataFrame) -> dict[str,str]:
    out: dict[str,str]={}
    for row in proteins.itertuples(index=False):
        for token in re.split(r'[;,|\s]+',str(row.aliases)):
            token=token.strip()
            if token:
                previous=out.get(token)
                if previous is not None and previous != str(row.protein_id):
                    raise RuntimeError(f'alias collision: {token}: {previous} vs {row.protein_id}')
                out[token]=str(row.protein_id)
    return out


def _aligned_global_embeddings(proteins: pd.DataFrame, directory: Path) -> tuple[np.ndarray,np.ndarray,dict[str,object]]:
    entries=pd.read_csv(directory/'entries.csv',dtype=str).fillna('')
    matrix=np.load(directory/'embeddings.npy',mmap_mode='r')
    available=np.load(directory/'available.npy').astype(bool)
    if 'protein_id' not in entries.columns:
        raise RuntimeError('canonical global asset must expose protein_id')
    entries['row']=pd.to_numeric(entries['row']).astype(int)
    order=entries.sort_values('row').protein_id.astype(str).tolist()
    expected=proteins.protein_id.astype(str).tolist()
    if order != expected or matrix.shape[0] != len(expected) or len(available) != len(expected):
        raise RuntimeError('canonical global asset is not aligned to protein universe')
    if not np.all(available):
        raise RuntimeError('base global sequence coordinate must be complete')
    return np.asarray(matrix,dtype=np.float32),available,{
        'source_rows':int(len(entries)),
        'available_proteins':int(available.sum()),
        'canonical_aligned':True,
    }


def _chordal_distance(embeddings: np.ndarray, available: np.ndarray) -> np.ndarray:
    x=np.asarray(embeddings,dtype=np.float64)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    good=np.asarray(available,dtype=bool)&(norm[:,0]>1e-12)
    xn=np.zeros_like(x)
    xn[good]=x[good]/norm[good]
    sim=np.clip(xn@xn.T,-1.0,1.0)
    d=np.sqrt(np.maximum(2.0-2.0*sim,0.0))
    invalid=~(good[:,None]&good[None,:])
    d[invalid]=np.inf
    np.fill_diagonal(d,0.0)
    return d


def _structural_diffusion_view(directory: Path) -> tuple[np.ndarray,np.ndarray,dict[str,object]]:
    sim=np.load(directory/'similarity.npy').astype(np.float64)
    available=np.load(directory/'available.npy').astype(bool)
    if sim.shape != (len(available),len(available)):
        raise RuntimeError(f'structural view shape mismatch: {directory}')
    sim=np.where(np.outer(available,available),np.maximum(sim,0.0),0.0)
    np.fill_diagonal(sim,0.0)
    affinity=csr_matrix(sim)
    degree=np.asarray((affinity>0).sum(axis=1)).reshape(-1)
    relational_available=available&(degree>0)
    # Diffusion distance is a genuine Euclidean/pseudometric distance between
    # one-step Markov neighbourhoods. It avoids asserting triangle inequality
    # for raw Foldseek or OT-derived similarity scores.
    d=one_step_diffusion_distance(affinity)
    invalid=~(relational_available[:,None]&relational_available[None,:])
    d[invalid]=np.inf
    np.fill_diagonal(d,0.0)
    return d,relational_available,{
        'raw_available':int(available.sum()),
        'relational_available':int(relational_available.sum()),
        'isolated_observed_nodes':int(np.sum(available&~relational_available)),
        'raw_positive_undirected_edges':int(affinity.nnz//2),
        'raw_degree_median':float(np.median(degree[relational_available])) if np.any(relational_available) else 0.0,
    }


def main() -> None:
    ap=argparse.ArgumentParser(description='Build one partially observed multiresolution protein geometry for the terpene submanifold.')
    ap.add_argument('--proteins',type=Path,default=DEFAULT_PROTEINS)
    ap.add_argument('--global-embeddings',type=Path,default=DEFAULT_GLOBAL)
    ap.add_argument('--structural-root',type=Path,default=DEFAULT_STRUCTURAL)
    ap.add_argument('--pocket-local',type=Path,default=DEFAULT_POCKET_LOCAL)
    ap.add_argument('--motifs',type=Path,default=DEFAULT_MOTIFS)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=ap.parse_args()

    proteins=pd.read_csv(args.proteins,dtype=str).fillna('')
    if proteins.protein_id.duplicated().any():
        raise RuntimeError('protein universe must contain unique protein_id values')
    n=len(proteins)
    global_emb,global_available,global_info=_aligned_global_embeddings(proteins,args.global_embeddings)
    global_distance=_chordal_distance(global_emb,global_available)

    pocket_local=np.load(args.pocket_local/'embeddings.npy',mmap_mode='r').astype(np.float32)
    pocket_local_available=np.load(args.pocket_local/'available.npy').astype(bool)
    if pocket_local.shape[0] != n or len(pocket_local_available) != n:
        raise RuntimeError('pocket-local asset is not aligned to the protein universe')
    pocket_local_distance=_chordal_distance(pocket_local,pocket_local_available)

    distances=[global_distance,pocket_local_distance]
    masks=[global_available,pocket_local_available]
    names=['global_esmc_chordal','pocket_local_esmc_chordal']
    motif_info={}
    for motif_name in ['typeI_aspartate','nse_dte','dxdd','qw']:
        emb=np.load(args.motifs/f'{motif_name}_embeddings.npy',mmap_mode='r').astype(np.float32)
        avail=np.load(args.motifs/f'{motif_name}_available.npy').astype(bool)
        if emb.shape[0]!=n or len(avail)!=n:
            raise RuntimeError(f'motif view not aligned: {motif_name}')
        dist=_chordal_distance(emb,avail)
        distances.append(dist); masks.append(avail); names.append(f'{motif_name}_esmc_chordal')
        motif_info[motif_name]={'available':int(avail.sum())}
    structural_info={}
    structural_distances={}
    structural_masks={}
    for name in ['whole_3di','pocket_3di','pocket_ot']:
        d,a,info=_structural_diffusion_view(args.structural_root/name)
        distances.append(d); masks.append(a); names.append(f'{name}_diffusion')
        structural_info[name]=info; structural_distances[name]=d; structural_masks[name]=a

    affinity,pullback_info=partial_observation_pullback_affinity(distances,masks)
    conformal,conformal_info=diffusion_conformal_affinity(affinity)

    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'row':np.arange(n,dtype=int),'protein_id':proteins.protein_id.astype(str)}).to_csv(out/'protein_ids.csv',index=False)
    np.save(out/'global_esmc_embeddings.npy',global_emb)
    np.save(out/'global_esmc_available.npy',global_available)
    np.save(out/'global_esmc_chordal_distance.npy',global_distance.astype(np.float32))
    np.save(out/'pocket_local_esmc_chordal_distance.npy',pocket_local_distance.astype(np.float32))
    np.save(out/'pocket_local_esmc_available.npy',pocket_local_available)
    for motif_name in ['typeI_aspartate','nse_dte','dxdd','qw']:
        np.save(out/f'{motif_name}_available.npy',np.load(args.motifs/f'{motif_name}_available.npy').astype(bool))
    for name in structural_distances:
        np.save(out/f'{name}_diffusion_distance.npy',structural_distances[name].astype(np.float32))
        np.save(out/f'{name}_relational_available.npy',structural_masks[name])
    save_npz(out/'partial_pullback_affinity.npz',affinity)
    save_npz(out/'diffusion_conformal_affinity.npz',conformal)

    view_count=np.zeros(n,dtype=int)
    for a in masks: view_count += a.astype(int)
    coverage=pd.DataFrame({'protein_id':proteins.protein_id.astype(str),'observed_view_count':view_count})
    for name,a in zip(names,masks): coverage[name]=a
    coverage.to_csv(out/'coverage.csv',index=False)

    manifest={
        'version':'terpene-multiresolution-protein-geometry-v4',
        'semantic_role':'label-free partially observed molecular-state geometry; sparse pair labels are not read',
        'protein_count':n,
        'views':names,
        'view_policy':'Each jointly observed view contributes its self-tuned dimensionless pullback energy. Energies are averaged over jointly observed views; missing views contribute neither attraction nor repulsion; no inter-view weights are learned or tuned.',
        'structural_policy':'Raw 3Di/pocket/OT similarities are used only as graph affinities and converted to exact t=1 diffusion distances before entering the pullback metric; raw similarity is not asserted to satisfy metric axioms.',
        'global_info':global_info,
        'motif_info':motif_info,
        'structural_info':structural_info,
        'pullback_info':pullback_info,
        'diffusion_conformal_info':conformal_info,
        'view_count_distribution':{str(int(k)):int(v) for k,v in coverage.observed_view_count.value_counts().sort_index().items()},
        'isolated_after_conformal':int(np.sum(np.asarray(conformal.sum(axis=1)).reshape(-1)<=0)),
        'labels_used':False,
        'input_sha256':{
            'protein_entities':_sha256(args.proteins),
            'global_manifest':_sha256(args.global_embeddings/'manifest.json'),
            'global_entries':_sha256(args.global_embeddings/'entries.csv'),
            'global_embeddings':_sha256(args.global_embeddings/'embeddings.npy'),
            'structural_manifest':_sha256(args.structural_root/'manifest.json'),
            'pocket_local_manifest':_sha256(args.pocket_local/'manifest.json'),
            'pocket_local_embeddings':_sha256(args.pocket_local/'embeddings.npy'),
            'motif_manifest':_sha256(args.motifs/'manifest.json'),
        },
    }
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
