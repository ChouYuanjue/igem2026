import numpy as np,torch
from projects.active.terpene_screening.tps_active_site_xattn_model import TPSActiveSiteXAttn,XAttnConfig
from projects.active.terpene_screening.run_tps_active_site_xattn_v1 import compact,stable_offset,parameter_count

def cfg(): return XAttnConfig(96,4,1,0.1,3e-4,1e-4,0.1,16)
def test_model_pair_shape_and_finite():
 m=TPSActiveSiteXAttn(cfg()); B=3; p=torch.randn(B,7,1152); pm=torch.ones(B,7,dtype=torch.bool); pt=torch.zeros(B,7,dtype=torch.long); pr=torch.zeros(B,7,dtype=torch.long); r=torch.randn(B,9,23); rm=torch.ones(B,9,dtype=torch.bool); rc=torch.zeros(B,9,dtype=torch.bool); rc[:,1]=True
 y=m(p,pm,pt,pr,r,rm,rc); assert y.shape==(B,) and torch.isfinite(y).all()
def test_compact_preserves_valid_order():
 x=np.arange(2*4*1).reshape(2,4,1); m=np.array([[1,0,1,0],[0,1,1,0]],bool); e=np.arange(8).reshape(2,4); y,ym,c,ye=compact(x,m,e); assert y[0,:2,0].tolist()==[0,2] and ye[1,:2].tolist()==[5,6] and c.tolist()==[2,2]
def test_hash_and_parameter_count_deterministic():
 assert stable_offset('p','q')==stable_offset('p','q'); assert parameter_count(cfg())>0
