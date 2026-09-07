import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_pretarget_source_alignment_is_exact_and_claim_scope_is_bounded():
 r=json.loads((ROOT/'projects/active/terpene_screening/RHEA128_CLEAN2023_SOURCE_ALIGNMENT_V2.json').read_text())
 assert r['status']=='pre_target_reveal_pass' and r['target_release141_associations_read'] is False
 assert r['compact2023']['missing_from_release128_raw_rhea_id']==0
 assert r['clean2023_reconstruction']['exact'] is True
 assert r['clean2023_reconstruction']['missing']==0 and r['clean2023_reconstruction']['extra']==0
 assert 'not a claim that the entire feature/candidate environment is historically frozen to 2023' in r['claim_scope']
