import numpy as np, pandas as pd
from projects.active.terpene_screening.prepare_tps_xattn_fresh_split_v1 import stable_fold,binary_tanimoto_matrix,grouped_transfer_scores

def test_fresh_fold_is_deterministic():
 assert stable_fold('abc')==stable_fold('abc') and 0<=stable_fold('abc')<5

def test_binary_tanimoto():
 x=np.zeros((2,2115),np.float32); x[0,[0,1]]=1; x[1,[1,2]]=1; t=binary_tanimoto_matrix(x); assert np.isclose(t[0,1],1/3) and np.isclose(t[0,0],1)

def test_grouped_transfer_matches_pairwise_max():
 cos=np.array([[1,.5,.2],[.5,1,.4],[.2,.4,1]],np.float32); rs=np.array([[1,.3],[.3,1]],np.float32)
 train=pd.DataFrame({'Entry':['p0','p1','p2'],'rhea_id':['r0','r0','r1']}); got=grouped_transfer_scores(cos,rs,train,[1],{'p0':0,'p1':1,'p2':2},{'r0':0,'r1':1})[0]
 expected=np.max(np.stack([np.maximum(cos[:,0],0)*.3,np.maximum(cos[:,1],0)*.3,np.maximum(cos[:,2],0)*1.0]),axis=0)
 assert np.allclose(got,expected)
