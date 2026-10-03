from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
ROOT=Path(__file__).resolve().parents[4]

def test_reaction_center_observations_cover_current_453_without_labels():
 d=ROOT/'data/terpene_reaction_center_observations_v1'; m=json.loads((d/'manifest.json').read_text()); ids=pd.read_csv(d/'reaction_ids.csv',dtype=str); ids['row']=pd.to_numeric(ids['row']).astype(int); ids=ids.sort_values('row'); cur=pd.read_csv(ROOT/'data/terpene_marts_adaptation/reaction_entities.csv',dtype=str).fillna('')
 assert m['reaction_count']==453 and m['all_reactions_observed'] is True and m['labels_used'] is False
 assert ids.reaction_id.tolist()==cur.reaction_id.tolist()
 x=np.load(d/'transition_atom_states.npy',mmap_mode='r'); mask=np.load(d/'transition_atom_mask.npy')
 assert x.shape[0]==453 and x.shape[2]==42 and mask.shape==x.shape[:2]
 assert np.all(mask.sum(1)>0)

def test_reaction_geometry_v2_is_complete_and_label_free():
 d=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'; m=json.loads((d/'manifest.json').read_text()); g=load_npz(d/'diffusion_conformal_affinity.npz').tocsr(); cov=pd.read_csv(d/'coverage.csv')
 assert m['reaction_count']==453 and m['labels_used'] is False and m['isolated_after_conformal']==0
 assert m['views']==['drfp_chordal','center_transition_wasserstein','center_token_jaccard']
 assert int(cov.observed_view_count.min())>=2
 assert np.all(np.asarray(g.sum(axis=1)).reshape(-1)>0)

def test_reaction_center_distances_are_symmetric_finite_and_zero_diagonal():
 d=ROOT/'data/terpene_multiresolution_reaction_geometry_v2'
 for name in ['center_transition_wasserstein','center_token_jaccard']:
  x=np.load(d/f'{name}_distance.npy')
  assert x.shape==(453,453)
  assert np.all(np.isfinite(x)) and np.all(x>=0)
  assert np.max(np.abs(x-x.T))<1e-6
  assert np.max(np.abs(np.diag(x)))<1e-8

def test_hierarchical_reaction_geometry_refines_but_never_rewires_global_atlas():
    from scipy.sparse import load_npz
    base=load_npz(ROOT/'data/terpene_multiresolution_reaction_geometry_v1/partial_pullback_affinity.npz').tocsr()
    d=ROOT/'data/terpene_reaction_atlas_refined_geometry_v2'
    refined=load_npz(d/'atlas_refined_affinity.npz').tocsr()
    manifest=json.loads((d/'manifest.json').read_text())
    assert manifest['labels_used'] is False
    assert manifest['isolated_after_conformal']==0
    assert manifest['topology_edges']==manifest['refined_edges']
    # Exact support equality: catalytic-local measurements may change edge length, never edge membership.
    assert np.array_equal(base.indptr,refined.indptr)
    assert np.array_equal(base.indices,refined.indices)
    # The local measurements must actually refine numerical edge lengths rather than being decorative.
    assert float(np.max(np.abs(base.data.astype(float)-refined.data.astype(float))))>1e-6
