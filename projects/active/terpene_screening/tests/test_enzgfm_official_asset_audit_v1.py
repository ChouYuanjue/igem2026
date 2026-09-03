import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]

def load(): return json.loads((ROOT/"projects/active/terpene_screening/ENZGFM_OFFICIAL_ASSET_AUDIT_V1.json").read_text())

def test_strongest_official_encoder_and_missing_retrieval_head_are_separate():
 d=load(); a=d["official_weight_record"]["archives"]
 assert d["official_weight_record"]["zenodo_record"]=="22042585"
 assert a["EnzGFM_1.5B.zip"]["contains_encoder"] is True
 assert a["EnzGFM_1.5B.zip"]["contains_retrieval_head"] is False
 assert a["EnzGFM_650M.zip"]["contains_retrieval_head"] is False
 assert d["conclusion"]["strongest_author_variant_with_official_encoder_assets"]=="EnzGFM-1.5B"
 assert d["conclusion"]["exact_author_checkpoint_reproduction_available"] is False

def test_no_650m_substitution_for_unique_baseline():
 c=json.loads((ROOT/"projects/active/terpene_screening/CATALYST_CAPABILITY_BASELINE_CONTRACT_V1.json").read_text())
 for cid in ("r2e_sequence_reaction","e2r_sequence_reaction"):
  assert c["contracts"][cid]["authoritative_external_baseline"]=="EnzGFM-1.5B"
  assert "head_checkpoint_missing" in c["contracts"][cid]["execution_status"]
