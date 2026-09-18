from __future__ import annotations

import numpy as np
import pandas as pd

from projects.active.fibre.geometry.structure import query_structural_view


def test_known_whole_structure_query_edges_are_legal_canonical_edges():
    proteins=pd.read_csv('data/terpene_marts_adaptation/protein_entities.csv',dtype=str).fillna('')
    pmap={str(pid):i for i,pid in enumerate(proteins.protein_id.astype(str))}
    candidates=pd.read_csv('data/terpene_p2rank_current_v1/candidates.csv',dtype=str).fillna('')
    pockets=pd.read_csv('results/terpene_p2rank_current_v1/p2rank_pocket_manifest.csv',dtype=str).fillna('')
    row=pockets.iloc[0]
    pid=str(candidates[candidates.UniprotID.eq(str(row.UniprotID))].iloc[0].protein_id)
    q=pmap[pid]
    result=query_structural_view(
        str(row.structure_path),view='whole_3di',work_root='results/starase_navigator_runtime/tmp'
    )
    canonical=np.load('data/terpene_structural_observations_v1/whole_3di/similarity.npy',mmap_mode='r')
    raw=result.raw_similarity
    positive=np.flatnonzero(raw>0)
    assert len(positive)==64
    assert np.all(np.asarray(canonical[q])[positive]>0)
    assert np.all(raw[positive] <= np.asarray(canonical[q])[positive] + 1e-5)
    assert result.observed is True
    assert np.all(np.isfinite(result.diffusion_distance[np.load('data/terpene_correspondence_structural_basis_v1/whole_3di/available.npy').astype(bool)]))
