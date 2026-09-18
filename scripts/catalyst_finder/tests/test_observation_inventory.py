import pandas as pd
from scripts.catalyst_finder.observations.inventory import ObservationInventory, PROTEIN_COVERAGE, REACTION_COVERAGE

def test_inventory_matches_exact_protein_coverage():
    inv=ObservationInventory()
    cov=pd.read_csv(PROTEIN_COVERAGE)
    row=cov[cov.pocket_ot_diffusion.astype(bool)].iloc[0]
    m=inv.protein_measurements(str(row.protein_id))
    assert {"protein_sequence","global_esmc","resolved_structure","pocket_detection","pocket_ot"} <= m

def test_global_only_protein_does_not_fake_structure():
    inv=ObservationInventory()
    cov=pd.read_csv(PROTEIN_COVERAGE)
    row=cov[cov.observed_view_count.eq(1)].iloc[0]
    m=inv.protein_measurements(str(row.protein_id))
    assert m=={"protein_sequence","global_esmc"}

def test_protein_alias_resolves_to_canonical_atlas_point():
    inv=ObservationInventory()
    # Q93YV0 is a known alias in the canonical entity table.
    assert inv.canonical_protein_id("Q93YV0")=="MARTS_SEQ_007542cb57235054"
    assert "global_esmc" in inv.protein_measurements("Q93YV0")

def test_reaction_inventory_has_complete_global_chemistry():
    inv=ObservationInventory()
    cov=pd.read_csv(REACTION_COVERAGE)
    row=cov.iloc[0]
    m=inv.reaction_measurements(str(row.reaction_id))
    assert {"reaction_structure","drfp","reactant_product_neighbourhood","atom_mapping","reaction_center_transition"} <= m


def test_candidate_summaries_separate_factor_geometry_from_mechanism_evidence():
    inv=ObservationInventory()
    protein=inv.summarize_protein_candidates(['Q93YV0'])
    assert protein['candidate_count']==1
    assert protein['reference_resolved_count']==1
    assert protein['measurement_counts']['global_esmc']==1
    assert 'pocket_ot' in protein['factor_measurements']
    assert 'resolved_structure' in protein['prerequisite_measurements']

    reactions=pd.read_csv(REACTION_COVERAGE)
    rid=str(reactions.iloc[0].reaction_id)
    reaction=inv.summarize_reaction_candidates([rid])
    assert reaction['candidate_count']==1
    assert reaction['reference_resolved_count']==1
    assert reaction['measurement_counts']['drfp']==1
    assert 'drfp' in reaction['factor_measurements']
    assert 'reaction_center_transition' in reaction['evidence_only_measurements']


def test_local_structure_lookup_reuses_existing_cache_without_network():
    from pathlib import Path
    from scripts.catalyst_finder.evidence_catalog import IntegratedEvidenceCatalog
    evidence=IntegratedEvidenceCatalog(Path('.').resolve())
    structure=evidence.candidate_protein_structure('A5G9B7')
    assert structure is not None and structure.is_file()
    assert 'AF-A5G9B7-F1-model_v6.cif' in structure.name
