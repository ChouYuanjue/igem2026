import numpy as np
from rdkit import Chem
from projects.active.terpene_screening.prepare_tps_active_site_tokens_v1 import PROTEIN_BUDGET,REACTION_BUDGET,ATOM_DIM,atom_vector,motif_centers_for_slots,changed_map_ids

def test_token_budgets_are_frozen():
 assert PROTEIN_BUDGET==126 and REACTION_BUDGET==104 and ATOM_DIM==23

def test_motif_slots_are_fixed_and_bounded():
 x=motif_centers_for_slots('M'*20+'DDQQD'+'A'*120+'NDQQSTTTE'+'A'*20+'QWAAQW')
 assert len(x)==5 and x[0] is not None and x[1] is not None and x[3] is not None and x[4] is not None

def test_changed_map_detects_bond_change():
 m='[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]'
 assert changed_map_ids(m)=={1,2}
 a=Chem.MolFromSmiles('[CH3:1]C').GetAtomWithIdx(0)
 assert atom_vector(a,product=False,changed=True).shape==(23,)
