from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.geometry.multiscale import _self_tuning_scale

PG=ROOT/'data/terpene_multiresolution_protein_geometry_v4'
RG=ROOT/'data/terpene_multiresolution_reaction_geometry_v1'
OUT=ROOT/'data/terpene_correspondence_deployment_atlas_v1'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def affinity_to_unit_length_graph(path: Path) -> tuple[csr_matrix,float]:
    w=load_npz(path).astype(np.float64).maximum(load_npz(path).astype(np.float64).T).tocsr()
    w.setdiag(0); w.eliminate_zeros()
    upper=w.tocoo(); mask=upper.row<upper.col
    raw=np.sqrt(np.maximum(-np.log(np.clip(upper.data[mask],1e-300,1.0)),1e-12))
    ell=float(np.median(raw))
    length=np.sqrt(np.maximum(-np.log(np.clip(w.data,1e-300,1.0)),1e-12))/ell
    return csr_matrix((length,w.indices,w.indptr),shape=w.shape),ell


def save_scales(prefix: str, root: Path, specs: list[tuple[str,str,str]]) -> dict:
    info={}
    for name,dist_file,avail_file in specs:
        d=np.load(root/dist_file,mmap_mode='r')
        a=np.load(root/avail_file).astype(bool)
        sigma=_self_tuning_scale(np.asarray(d,dtype=np.float64),a,epsilon=1e-8)
        np.save(OUT/f'{prefix}_{name}_available.npy',a)
        np.save(OUT/f'{prefix}_{name}_scale.npy',sigma.astype(np.float32))
        info[name]={'available':int(a.sum()),'scale_finite':int(np.isfinite(sigma).sum())}
    return info


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p_specs=[
        ('global_esmc','global_esmc_chordal_distance.npy','global_esmc_available.npy'),
        ('pocket_local_esmc','pocket_local_esmc_chordal_distance.npy','pocket_local_esmc_available.npy'),
        ('whole_3di','whole_3di_diffusion_distance.npy','whole_3di_relational_available.npy'),
        ('pocket_3di','pocket_3di_diffusion_distance.npy','pocket_3di_relational_available.npy'),
        ('pocket_ot','pocket_ot_diffusion_distance.npy','pocket_ot_relational_available.npy'),
    ]
    # Catalytic motif coordinate embeddings remain reproducible raw measurement maps.
    # They are not needed by the first online cross-distance implementation and
    # therefore are not duplicated into this low-latency deployment bundle yet.
    r_specs=[
        ('drfp','drfp_chordal_distance.npy','drfp_available.npy'),
        ('reactant','reactant_diffusion_distance.npy','reactant_available.npy'),
        ('product','product_diffusion_distance.npy','product_available.npy'),
    ]
    p_info=save_scales('protein',PG,p_specs)
    r_info=save_scales('reaction',RG,r_specs)

    p_graph,p_ell=affinity_to_unit_length_graph(PG/'partial_pullback_affinity.npz')
    r_graph,r_ell=affinity_to_unit_length_graph(RG/'partial_pullback_affinity.npz')
    save_npz(OUT/'protein_unit_length_graph.npz',p_graph)
    save_npz(OUT/'reaction_unit_length_graph.npz',r_graph)

    # Preserve the exact canonical entity order used by every deployment array.
    (OUT/'protein_ids.csv').write_bytes((PG/'protein_ids.csv').read_bytes())
    (OUT/'reaction_ids.csv').write_bytes((RG/'reaction_ids.csv').read_bytes())

    manifest={
        'version':'terpene-correspondence-deployment-atlas-v1',
        'purpose':'immutable reference atlas bundle for low-latency out-of-sample query attachment',
        'reference_policy':'reference-reference geometry is frozen; online requests only acquire query-side observations and attach the query to this graph',
        'factor_characteristic_lengths':{'protein':p_ell,'reaction':r_ell},
        'protein_reference_measurements':p_info,
        'reaction_reference_measurements':r_info,
        'protein_online_note':'global ESM-C is the minimum extension coordinate; cached structure/pocket coordinates may refine a query when cross-distance extractors are available',
        'reaction_online_note':'DRFP/reactant/product global chemistry define canonical ranking geometry; atom-mapped reaction-center observations remain mechanism evidence',
        'labels_used':False,
        'input_sha256':{
            'protein_manifest':sha(PG/'manifest.json'),
            'reaction_manifest':sha(RG/'manifest.json'),
            'protein_affinity':sha(PG/'partial_pullback_affinity.npz'),
            'reaction_affinity':sha(RG/'partial_pullback_affinity.npz'),
        },
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
