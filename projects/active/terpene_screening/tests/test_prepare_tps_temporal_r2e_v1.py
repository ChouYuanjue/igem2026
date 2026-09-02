import numpy as np
from projects.active.terpene_screening.prepare_tps_temporal_r2e_v1 import max_binary_tanimoto,seq_key

def test_binary_tanimoto_exact():
 q=np.array([1,1,0,0],dtype=bool); x=np.array([[1,1,0,0],[1,0,1,0],[0,0,1,1]],dtype=bool)
 assert max_binary_tanimoto(q,x)==1.0
 assert abs(max_binary_tanimoto(q,x[1:])-1/3)<1e-12

def test_sequence_keys_are_stable_and_content_bound():
 assert seq_key('AAAA')==seq_key('AAAA') and seq_key('AAAA')!=seq_key('AAAT')
