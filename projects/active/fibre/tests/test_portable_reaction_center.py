from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from projects.active.fibre.portable.core_bundle import build_bundle
from projects.active.fibre.portable.reaction_center import build_reaction_center_views
from projects.active.fibre.portable.reference_query import PortableReferenceBundle


def _dataset(root):
    proteins=pd.DataFrame({
        "protein_id":["P0","P1","P2","P3"],
        "sequence":["ACDE","ACDF","LMNP","LMNQ"],
    })
    reactions=pd.DataFrame({
        "reaction_id":["R0","R1","R2","R3"],
        "reaction_smiles":["CCO>>CC=O","CCCO>>CCC=O","CCN>>CC=N","CO>>C=O"],
    })
    pairs=pd.DataFrame({
        "protein_id":["P0","P2"],
        "reaction_id":["R0","R2"],
    })
    pf=pd.DataFrame({
        "protein_id":["P0","P1","P2","P3"],
        "f0":[1.0,0.9,0.0,0.0],"f1":[0.0,0.1,1.0,0.9],
    })
    rf=pd.DataFrame({
        "reaction_id":["R0","R1","R2","R3"],
        "f0":[1.0,0.8,0.0,0.1],"f1":[0.0,0.2,1.0,0.9],
    })
    mapped=pd.DataFrame({
        "reaction_id":["R0","R1","R2","R3"],
        "mapped_rxn":[
            "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
            "[CH3:1][CH2:2][CH2:3][OH:4]>>[CH3:1][CH2:2][CH:3]=[O:4]",
            "[CH3:1][CH2:2][NH2:3]>>[CH3:1][CH:2]=[NH:3]",
            "",
        ],
        "success":["true","true","true","false"],
        "confidence":["0.99","0.98","0.97",""],
    })
    for name,frame in [
        ("proteins.csv",proteins),("reactions.csv",reactions),("pairs.csv",pairs),
        ("protein_features.csv",pf),("reaction_features.csv",rf),("mapped.csv",mapped),
    ]:
        frame.to_csv(root/name,index=False)


def test_fresh_reaction_center_builder_preserves_missingness(tmp_path):
    _dataset(tmp_path)
    out=tmp_path/"center"
    manifest=build_reaction_center_views(
        tmp_path/"reactions.csv",tmp_path/"mapped.csv",out
    )
    assert manifest["available_count"]==3
    assert manifest["unavailable_count"]==1
    assert manifest["labels_used"] is False
    available=np.load(out/"center_available.npy")
    np.testing.assert_array_equal(available,[True,True,True,False])
    wd=np.load(out/"center_wasserstein_distance.npy")
    jd=np.load(out/"center_token_jaccard_distance.npy")
    assert wd.shape==(4,4) and jd.shape==(4,4)
    assert np.all(np.isfinite(wd[:3,:3]))
    assert np.all(np.isfinite(jd[:3,:3]))
    assert np.all(wd[:3,:3]>=0) and np.all(jd[:3,:3]>=0)
    assert np.allclose(wd[:3,:3],wd[:3,:3].T)
    assert np.allclose(jd[:3,:3],jd[:3,:3].T)
    assert np.isinf(wd[3,0]) and np.isinf(jd[3,0])


def test_distance_views_dense_blockwise_and_oos_are_consistent(tmp_path):
    _dataset(tmp_path)
    center=tmp_path/"center"
    build_reaction_center_views(
        tmp_path/"reactions.csv",tmp_path/"mapped.csv",center
    )
    common=dict(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        reaction_distance_view_paths={
            "center_wasserstein":(
                center/"center_wasserstein_distance.npy",
                center/"center_available.npy",
            ),
            "center_tokens":(
                center/"center_token_jaccard_distance.npy",
                center/"center_available.npy",
            ),
        },
        graph_k=2,
    )
    dense=tmp_path/"dense"
    block=tmp_path/"block"
    md=build_bundle(**common,output_dir=dense,graph_backend="dense_exact")
    mb=build_bundle(
        **common,output_dir=block,graph_backend="blockwise_exact",graph_block_size=2
    )
    assert md["factor_geometry"]["reaction"]["backend"]=="dense_exact_mixed_geometry"
    assert mb["factor_geometry"]["reaction"]["backend"]=="blockwise_exact_mixed_geometry"
    kinds={x["name"]:x["kind"] for x in md["factor_geometry"]["reaction"]["views"]}
    assert kinds=={
        "global":"feature",
        "center_wasserstein":"distance",
        "center_tokens":"distance",
    }
    np.testing.assert_allclose(
        load_npz(dense/"reaction_affinity.npz").toarray(),
        load_npz(block/"reaction_affinity.npz").toarray(),
        rtol=0,atol=1e-6,
    )

    bundle=PortableReferenceBundle(dense)
    reference=np.load(dense/"reaction_views/center_wasserstein_distance.npy")
    # Use R1 as a synthetic external query cross-distance vector. The purpose
    # is to validate the distance-view attachment contract, not self-ranking.
    rows=bundle.rank_enzymes_from_reaction_views(
        {},
        top_k=3,
        distances={"center_wasserstein":np.asarray(reference[1],dtype=float)},
    )
    assert len(rows)==3
    assert all(x["observed_views"]==["center_wasserstein"] for x in rows)
    assert all(np.isfinite(x["correspondence_defect"]) for x in rows)
