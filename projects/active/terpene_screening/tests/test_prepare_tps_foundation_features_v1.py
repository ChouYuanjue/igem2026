import numpy as np
from projects.active.terpene_screening.prepare_tps_foundation_features_v1 import norm

def test_equalblock_normalization_is_independent():
 x=np.array([[3.,4.],[0.,2.]],dtype=np.float32); y=norm(x)
 np.testing.assert_allclose(np.linalg.norm(y,axis=1),1.0,atol=1e-7)
