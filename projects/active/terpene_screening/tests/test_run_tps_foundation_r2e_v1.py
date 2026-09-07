from projects.active.terpene_screening.run_tps_foundation_r2e_v1 import development_cells,frozen_cells,candidate_spec

def test_cartesian_partitions_are_exact():
 assert development_cells()==[(0,4),(1,4),(2,4),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
 assert len(frozen_cells())==16 and set(development_cells()).isdisjoint(frozen_cells())

def test_frozen_candidate_grid_lookup():
 assert candidate_spec('enzgfm_rdkitplus')['protein']=='enzgfm'
 assert candidate_spec('equalblock_rdkitplus_noreplay')['replay']==0
