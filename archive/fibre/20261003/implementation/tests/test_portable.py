from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse.csgraph import dijkstra

from projects.active.fibre.geometry.correspondence import correspondence_state
from projects.active.fibre.portable.contracts import validate_tables
from projects.active.fibre.portable.core_bundle import build_bundle
from projects.active.fibre.portable.reference_query import PortableReferenceBundle
from projects.active.fibre.portable.validate import validate_dataset


def _write_toy(root: Path):
    proteins=pd.DataFrame({
        "protein_id":["P0","P1","P2","P3"],
        "sequence":["ACDE","ACDF","LMNP","LMNQ"],
    })
    reactions=pd.DataFrame({
        "reaction_id":["R0","R1","R2","R3"],
        "reaction_smiles":["CC>>CO","CCC>>CCO","N>>CN","NN>>CNN"],
    })
    pairs=pd.DataFrame({
        "protein_id":["P0","P2"],
        "reaction_id":["R0","R2"],
    })
    pf=pd.DataFrame({
        "protein_id":["P0","P1","P2","P3"],
        "f0":[1.0,0.9,0.0,0.0],
        "f1":[0.0,0.1,1.0,0.9],
    })
    rf=pd.DataFrame({
        "reaction_id":["R0","R1","R2","R3"],
        "f0":[1.0,0.8,0.0,0.1],
        "f1":[0.0,0.2,1.0,0.9],
    })
    pv=pd.DataFrame({
        "protein_id":["P0","P2","P3"],
        "f0":[1.0,0.0,0.2],
        "f1":[0.0,1.0,0.8],
    })
    rv=pd.DataFrame({
        "reaction_id":["R0","R2","R3"],
        "f0":[1.0,0.0,0.15],
        "f1":[0.0,1.0,0.85],
    })
    for name,frame in [
        ("proteins.csv",proteins),("reactions.csv",reactions),
        ("pairs.csv",pairs),("protein_features.csv",pf),
        ("reaction_features.csv",rf),
        ("protein_optional.csv",pv),("reaction_optional.csv",rv),
    ]:
        frame.to_csv(root/name,index=False)
    return proteins,reactions,pairs


def test_portable_contract_rejects_unknown_pair_entity():
    proteins=pd.DataFrame({"protein_id":["P0"],"sequence":["ACDE"]})
    reactions=pd.DataFrame({"reaction_id":["R0"],"reaction_smiles":["CC>>CO"]})
    pairs=pd.DataFrame({"protein_id":["MISSING"],"reaction_id":["R0"]})
    try:
        validate_tables(proteins,reactions,pairs)
    except ValueError as exc:
        assert "unknown entities" in str(exc)
    else:
        raise AssertionError("unknown pair entity must be rejected")


def test_portable_bundle_uses_support_distances_and_matches_full_state(tmp_path):
    proteins,reactions,pairs=_write_toy(tmp_path)
    report=validate_dataset(
        tmp_path/"proteins.csv",tmp_path/"reactions.csv",tmp_path/"pairs.csv"
    )
    assert report["status"]=="valid"
    out=tmp_path/"bundle"
    manifest=build_bundle(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        output_dir=out,
        graph_k=2,
    )
    assert manifest["distance_storage"].startswith("support-to-all")
    assert not (out/"protein_geodesic.npy").exists()
    assert not (out/"reaction_geodesic.npy").exists()

    pg=load_npz(out/"protein_length_graph.npz")
    rg=load_npz(out/"reaction_length_graph.npz")
    dp=np.asarray(dijkstra(pg,directed=False),dtype=np.float64)
    dr=np.asarray(dijkstra(rg,directed=False),dtype=np.float64)
    pi={x:i for i,x in enumerate(proteins.protein_id)}
    ri={x:i for i,x in enumerate(reactions.reaction_id)}
    pp=np.asarray([
        (ri[x.reaction_id],pi[x.protein_id])
        for x in pairs.itertuples(index=False)
    ],dtype=np.int64)
    full=correspondence_state(dr*dr,dp*dp,pp)

    bundle=PortableReferenceBundle(out)
    for rid in reactions.reaction_id:
        got=bundle.rank_enzymes(rid,top_k=4)
        expected=np.lexsort((
            np.asarray(proteins.protein_id,dtype=str),
            full.defect[ri[rid]],
        ))
        assert [x["protein_id"] for x in got] == [
            proteins.protein_id.iloc[int(i)] for i in expected
        ]
        np.testing.assert_allclose(
            [x["correspondence_defect"] for x in got],
            full.defect[ri[rid],expected],
            rtol=0,atol=1e-6,
        )

    for pid in proteins.protein_id:
        got=bundle.rank_reactions(pid,top_k=4)
        expected=np.lexsort((
            np.asarray(reactions.reaction_id,dtype=str),
            full.defect[:,pi[pid]],
        ))
        assert [x["reaction_id"] for x in got] == [
            reactions.reaction_id.iloc[int(i)] for i in expected
        ]
        np.testing.assert_allclose(
            [x["correspondence_defect"] for x in got],
            full.defect[expected,pi[pid]],
            rtol=0,atol=1e-6,
        )


def test_blockwise_exact_backend_matches_dense_affinity(tmp_path):
    _write_toy(tmp_path)
    dense=tmp_path/"dense"
    block=tmp_path/"block"
    md=build_bundle(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        output_dir=dense,
        graph_k=2,
        graph_backend="dense_exact",
    )
    mb=build_bundle(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        output_dir=block,
        graph_k=2,
        graph_backend="blockwise_exact",
        graph_block_size=2,
    )
    assert md["factor_geometry"]["protein"]["exact"] is True
    assert mb["factor_geometry"]["protein"]["exact"] is True
    np.testing.assert_allclose(
        load_npz(dense/"protein_affinity.npz").toarray(),
        load_npz(block/"protein_affinity.npz").toarray(),
        rtol=0,atol=1e-6,
    )
    np.testing.assert_allclose(
        load_npz(dense/"reaction_affinity.npz").toarray(),
        load_npz(block/"reaction_affinity.npz").toarray(),
        rtol=0,atol=1e-6,
    )
    np.testing.assert_allclose(
        np.load(dense/"protein_reference_scales.npy"),
        np.load(block/"protein_reference_scales.npy"),
        rtol=0,atol=1e-6,
    )


def test_frozen_reference_supports_out_of_sample_feature_query(tmp_path):
    _write_toy(tmp_path)
    out=tmp_path/"bundle"
    build_bundle(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        output_dir=out,
        graph_k=2,
        graph_backend="dense_exact",
    )
    bundle=PortableReferenceBundle(out)
    before=(out/"manifest.json").read_bytes()
    reaction_feature=np.asarray([0.95,0.05],dtype=np.float32)
    protein_feature=np.asarray([0.05,0.95],dtype=np.float32)
    r2e=bundle.rank_enzymes_from_reaction_feature(reaction_feature,top_k=3)
    e2r=bundle.rank_reactions_from_protein_feature(protein_feature,top_k=3)
    assert len(r2e)==3 and len(e2r)==3
    assert all(np.isfinite(x["correspondence_defect"]) for x in r2e+e2r)
    assert all(x["query_mode"]=="out_of_sample_frozen_reference" for x in r2e+e2r)
    assert (out/"manifest.json").read_bytes()==before


def test_partial_multiview_dense_and_blockwise_are_equivalent(tmp_path):
    _write_toy(tmp_path)
    dense=tmp_path/"dense_multi"
    block=tmp_path/"block_multi"
    common=dict(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        protein_view_paths={"pocket":tmp_path/"protein_optional.csv"},
        reaction_view_paths={"center":tmp_path/"reaction_optional.csv"},
        graph_k=2,
    )
    md=build_bundle(**common,output_dir=dense,graph_backend="dense_exact")
    mb=build_bundle(
        **common,output_dir=block,graph_backend="blockwise_exact",graph_block_size=2
    )
    assert md["factor_geometry"]["protein"]["views"][1]["name"]=="pocket"
    assert md["factor_geometry"]["protein"]["views"][1]["available_count"]==3
    assert mb["factor_geometry"]["reaction"]["views"][1]["name"]=="center"
    np.testing.assert_allclose(
        load_npz(dense/"protein_affinity.npz").toarray(),
        load_npz(block/"protein_affinity.npz").toarray(),
        rtol=0,atol=1e-6,
    )
    np.testing.assert_allclose(
        load_npz(dense/"reaction_affinity.npz").toarray(),
        load_npz(block/"reaction_affinity.npz").toarray(),
        rtol=0,atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.load(dense/"protein_views/pocket_available.npy"),
        np.asarray([True,False,True,True]),
    )
    np.testing.assert_array_equal(
        np.load(dense/"reaction_views/center_available.npy"),
        np.asarray([True,False,True,True]),
    )


def test_partial_multiview_oos_query_uses_only_observed_views(tmp_path):
    _write_toy(tmp_path)
    out=tmp_path/"bundle_multi"
    build_bundle(
        proteins_path=tmp_path/"proteins.csv",
        reactions_path=tmp_path/"reactions.csv",
        pairs_path=tmp_path/"pairs.csv",
        protein_features_path=tmp_path/"protein_features.csv",
        reaction_features_path=tmp_path/"reaction_features.csv",
        protein_view_paths={"pocket":tmp_path/"protein_optional.csv"},
        reaction_view_paths={"center":tmp_path/"reaction_optional.csv"},
        output_dir=out,
        graph_k=2,
        graph_backend="dense_exact",
    )
    bundle=PortableReferenceBundle(out)
    r2e=bundle.rank_enzymes_from_reaction_views(
        {
            "global":np.asarray([0.95,0.05],dtype=np.float32),
            "center":np.asarray([0.9,0.1],dtype=np.float32),
        },
        top_k=3,
    )
    e2r=bundle.rank_reactions_from_protein_views(
        {"pocket":np.asarray([0.1,0.9],dtype=np.float32)},
        top_k=3,
    )
    assert all(x["observed_views"]==["center","global"] for x in r2e)
    assert all(x["observed_views"]==["pocket"] for x in e2r)
    assert all(np.isfinite(x["correspondence_defect"]) for x in r2e+e2r)
