import numpy as np
from projects.active.terpene_screening.run_internal_functional_prototype_residual_v1 import (
    build_class_centroids,
    eligible_classes,
    parse_ec_prefixes,
)

def test_parse_ec_prefixes_filters_partial_labels():
    assert parse_ec_prefixes('1.2.3.4;2.7.-.-; 3.5.1.-',2)=={'1.2','2.7','3.5'}
    assert parse_ec_prefixes('1.-.-.-;bad',2)==set()

def test_eligible_classes_requires_both_modalities():
    p={'p1':{'1.1'},'p2':{'1.1'},'p3':{'2.7'}}
    r={'r1':{'1.1'},'r2':{'2.7'}}
    classes,support=eligible_classes(p,r,min_proteins=2,min_reactions=1)
    assert classes==['1.1']; assert support['2.7']['train_proteins']==1

def test_centroid_is_normalized():
    x=np.asarray([[1,0],[0,1],[1,1]],dtype=np.float32)
    c=build_class_centroids(x,['a','b','c'],{'a':{'1.1'},'c':{'1.1'}},['1.1'])
    assert c.shape==(1,2); assert np.isclose(np.linalg.norm(c[0]),1.0)
