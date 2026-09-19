import numpy as np

from projects.active.fibre.geometry.broad_section import (
    collapse_reaction_anchor_costs,
    group_reactions_by_protein_support,
    section_from_group_distances,
)


def _dense_section(qr, cp, pairs):
    anchor=np.full(cp.shape[1],np.inf,float)
    np.minimum.at(anchor,pairs[:,1],qr[pairs[:,0]])
    active=np.isfinite(anchor)
    joint=np.min(cp[:,active]+anchor[active][None,:],axis=1)
    qm=float(np.min(qr[np.unique(pairs[:,0])]))
    cm=np.min(cp[:,np.unique(pairs[:,1])],axis=1)
    return np.maximum(joint-qm-cm,0.0),joint,qm


def test_reaction_support_grouping_is_exact_for_min_plus_section():
    pairs=np.asarray([
        [0,0],[0,1],
        [1,2],
        [2,0],[2,1],  # same protein support as reaction 0
        [3,3],[3,4],
    ],dtype=np.int64)
    qr=np.asarray([0.8,0.2,0.1,0.7],float)
    cp=np.asarray([
        [0.1,0.4,1.0,0.7,0.8],
        [0.9,0.6,0.2,0.3,0.4],
        [0.5,0.1,0.8,0.9,0.2],
    ],float)
    groups=group_reactions_by_protein_support(pairs,reaction_support_size=4)
    assert groups.group_count==3
    g=np.column_stack([
        np.min(cp[:,members],axis=1)
        for members in groups.protein_groups
    ])
    actual=section_from_group_distances(qr,groups.reaction_to_group,g)
    expected=_dense_section(qr,cp,pairs)
    for a,e in zip(actual,expected):
        np.testing.assert_allclose(a,e,rtol=0,atol=1e-14)


def test_collapsed_anchor_uses_best_reaction_with_same_protein_support():
    pairs=np.asarray([[0,0],[1,0],[2,1]],dtype=np.int64)
    groups=group_reactions_by_protein_support(pairs,reaction_support_size=3)
    anchors=collapse_reaction_anchor_costs(
        np.asarray([0.9,0.1,0.4]),groups.reaction_to_group,groups.group_count
    )
    assert sorted(anchors.tolist())==[0.1,0.4]
