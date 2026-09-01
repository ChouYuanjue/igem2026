import json
from pathlib import Path
from projects.active.terpene_screening.model_capability_registry import scenario_map, validate_final_model_manifest
ROOT=Path(__file__).resolve().parents[4]
def test_internal_mainline_scenario_is_registered():
 s=scenario_map()['clean2023_salted_double_cold_r2e_mainline']
 assert s.strict_clean is True and s.confirmatory is True and s.directions==('reaction_to_enzyme',)
def test_final_mainline_manifest_is_direction_safe_and_no_fake_fusion():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_V1.json').read_text())
 assert validate_final_model_manifest(p)==[]
 assert p['production_default']['expert']=='r2e_similarity_router_v1'
 assert p['router']['uses_target_test_labels'] is False and p['router']['score_fusion_performed'] is False
 assert len(p['explicit_nonclaims'])>=3
def test_capability_evolution_ends_in_confirmed_production_package():
 p=json.loads((ROOT/'projects/active/terpene_screening/CATALYST_CLEAN_MAINLINE_CAPABILITY_V1.json').read_text())
 assert p['status']=='canonical_success_mainline' and [x['stage'] for x in p['evolution']]==[1,2,3,4,5,6,7]
 assert 'sim<0.9' in p['evolution'][-1]['headline']
