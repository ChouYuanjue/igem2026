import numpy as np
from projects.active.terpene_screening.geometry.multiscale import hierarchical_pullback_refinement_affinity

def base_line():
 x=np.arange(6,dtype=float)[:,None]; return np.abs(x-x.T)

def test_missing_refinement_is_exactly_neutral_and_topology_is_base_local():
 d=base_line(); a=np.ones(6,bool); r=np.abs(np.array([0,0,0,10,10,10],float)[:,None]-np.array([0,0,0,10,10,10],float)[None,:]); none=np.zeros(6,bool)
 g0,_=hierarchical_pullback_refinement_affinity(d,a,[],[])
 g1,_=hierarchical_pullback_refinement_affinity(d,a,[r],[none])
 assert np.array_equal(g0.toarray(),g1.toarray())

def test_duplicate_refinement_view_is_idempotent():
 d=base_line(); a=np.ones(6,bool); r=np.abs(np.array([0,0,1,5,5,6],float)[:,None]-np.array([0,0,1,5,5,6],float)[None,:])
 g1,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a])
 g2,_=hierarchical_pullback_refinement_affinity(d,a,[r,r],[a,a])
 assert np.allclose(g1.toarray(),g2.toarray(),rtol=1e-6,atol=1e-7)

def test_refinement_only_reduces_existing_base_edge_conductance():
 d=base_line(); a=np.ones(6,bool); r=np.abs(np.array([0,0,0,20,20,20],float)[:,None]-np.array([0,0,0,20,20,20],float)[None,:])
 g0,_=hierarchical_pullback_refinement_affinity(d,a,[],[])
 g1,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a])
 assert set(zip(*g1.nonzero())) <= set(zip(*g0.nonzero()))
 mask=g1.toarray()>0
 assert np.all(g1.toarray()[mask] <= g0.toarray()[mask]+1e-7)

def test_missing_base_node_uses_fallback_without_rewriting_observed_base_topology():
 d=base_line(); a=np.ones(6,bool); a[-1]=False; d[-1,:]=np.inf; d[:,-1]=np.inf; d[-1,-1]=0
 f=base_line(); fa=np.ones(6,bool)
 g,info=hierarchical_pullback_refinement_affinity(d,a,[],[],fallback_distance=f,fallback_available=fa)
 assert info['base_missing']==1 and info['fallback_directed_edges']>0
 assert g.getrow(5).nnz>0

def test_zero_reliability_is_exactly_neutral_and_unit_reliability_is_full_refinement():
 d=base_line(); a=np.ones(6,bool); r=np.abs(np.array([0,0,0,20,20,20],float)[:,None]-np.array([0,0,0,20,20,20],float)[None,:])
 g0,_=hierarchical_pullback_refinement_affinity(d,a,[],[])
 gz,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a],[np.zeros(6)])
 gu,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a],[np.ones(6)])
 assert np.allclose(g0.toarray(),gz.toarray(),rtol=1e-7,atol=1e-8)
 assert np.any(np.abs(gu.toarray()-g0.toarray())>1e-7)

def test_intermediate_reliability_interpolates_conductance_monotonically():
 d=base_line(); a=np.ones(6,bool); r=np.abs(np.array([0,0,0,20,20,20],float)[:,None]-np.array([0,0,0,20,20,20],float)[None,:])
 g0,_=hierarchical_pullback_refinement_affinity(d,a,[],[])
 gh,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a],[np.full(6,.5)])
 g1,_=hierarchical_pullback_refinement_affinity(d,a,[r],[a],[np.ones(6)])
 mask=g1.toarray()>0
 assert np.all(g1.toarray()[mask] <= gh.toarray()[mask]+1e-7)
 assert np.all(gh.toarray()[mask] <= g0.toarray()[mask]+1e-7)
