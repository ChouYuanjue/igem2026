from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT=Path(__file__).resolve().parents[4]


def test_canonical_global_esmc_covers_fixed_terpene_universe():
    proteins=pd.read_csv(ROOT/'data/terpene_marts_adaptation/protein_entities.csv',dtype=str).fillna('')
    d=ROOT/'data/terpene_global_esmc_aligned_v1'
    entries=pd.read_csv(d/'entries.csv',dtype=str).fillna('')
    entries['row']=pd.to_numeric(entries['row']).astype(int)
    entries=entries.sort_values('row')
    matrix=np.load(d/'embeddings.npy',mmap_mode='r')
    available=np.load(d/'available.npy')
    manifest=json.loads((d/'manifest.json').read_text())
    assert len(proteins)==1421
    assert entries.protein_id.tolist()==proteins.protein_id.tolist()
    assert matrix.shape==(1421,1152)
    assert available.shape==(1421,) and bool(np.all(available))
    assert manifest['available_count']==1421
    assert manifest['source_counts']=={'historical_aligned':1414,'sequence_fill':7}
    assert manifest['duplicate_alias_min_cosine']>0.99999
    assert manifest['labels_used'] is False


def test_pocket_local_asset_uses_only_current_semantics():
    d=ROOT/'data/terpene_pocket_local_aligned_v2'
    manifest=json.loads((d/'manifest.json').read_text())
    available=np.load(d/'available.npy')
    entries=pd.read_csv(d/'entries.csv',dtype=str).fillna('')
    stale=pd.read_csv(d/'excluded_stale_alias_observations.csv',dtype=str).fillna('')
    assert int(available.sum())==1287
    assert manifest['source_counts']['historical_exact_semantic_reuse']==711
    assert manifest['source_counts']['external_fill_current_semantics']==567
    assert manifest['source_counts']['current_gap_recompute']==9
    assert manifest['stale_alias_observation_count']==13==len(stale)
    historical_used=set(entries.loc[entries.source.eq('historical_exact_semantic_reuse'),'Entry'])
    # A stale historical observation may share the current accession with a freshly recomputed vector;
    # what is forbidden is reusing that stale vector under historical provenance.
    assert not (historical_used & set(stale.stale_uid))
    for uid in stale.loc[stale.current_uid.eq(stale.stale_uid),'current_uid']:
        row=entries[entries.current_uid.eq(uid)]
        assert len(row)==1 and row.iloc[0].source=='current_gap_recompute'
    assert manifest['labels_used'] is False


def test_family_aware_motif_coordinates_are_partial_observations():
    d=ROOT/'data/terpene_family_aware_motif_coordinates_v1'
    manifest=json.loads((d/'manifest.json').read_text())
    assert manifest['available_counts']=={
        'typeI_aspartate':827,'nse_dte':647,'dxdd':142,'qw':27,
    }
    for name,count in manifest['available_counts'].items():
        a=np.load(d/f'{name}_available.npy')
        x=np.load(d/f'{name}_embeddings.npy',mmap_mode='r')
        assert x.shape==(1421,1152)
        assert int(a.sum())==count
        assert np.all(np.linalg.norm(x[a],axis=1)>0)
        assert np.all(x[~a]==0)  # storage sentinel only; unavailable rows are masked from geometry.
    assert manifest['labels_used'] is False


def test_v4_has_complete_base_coordinate_no_isolates_and_no_labels():
    d=ROOT/'data/terpene_multiresolution_protein_geometry_v4'
    manifest=json.loads((d/'manifest.json').read_text())
    coverage=pd.read_csv(d/'coverage.csv')
    global_available=np.load(d/'global_esmc_available.npy')
    graph=load_npz(d/'diffusion_conformal_affinity.npz').tocsr()
    assert manifest['version']=='terpene-multiresolution-protein-geometry-v4'
    assert manifest['view_count']==9 if 'view_count' in manifest else len(manifest['views'])==9
    assert manifest['pullback_info']['view_available_counts'][0]==1421
    assert bool(np.all(global_available))
    assert int(coverage.observed_view_count.min())>=1
    assert manifest['isolated_after_conformal']==0
    assert np.all(np.asarray(graph.sum(axis=1)).reshape(-1)>0)
    assert manifest['labels_used'] is False
