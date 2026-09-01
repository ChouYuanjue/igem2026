import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RESULT = ROOT / "projects/active/terpene_screening/REACTZYME_NATIVE_SUPPORT_ADAPTATION_V1_RESULT.json"


def test_reactzyme_retention_result_is_a_frozen_rejection():
    d = json.loads(RESULT.read_text())
    assert d["status"] == "rejected_both_safe_policies"
    assert d["decision"] == "retain_unchanged_current_model"
    assert d["selected_policy"] is None
    assert d["external_metrics_used_for_selection"] is False
    assert d["retuning_allowed_from_this_result"] is False
    assert d["policies"]["union_safe_max"]["pass"] is False
    assert d["policies"]["enzyme_safe"]["pass"] is False
    assert d["policies"]["union_safe_max"]["pooled"]["r2e_all"]["n_queries"] == 3226
    assert d["policies"]["enzyme_safe"]["pooled"]["r2e_lt0p3"]["n_queries"] == 204
    assert d["policies"]["enzyme_safe"]["pooled"]["e2r_no_hit"]["n_queries"] == 5627


def test_enzyme_safe_failure_is_not_hidden_by_aggregate_metrics():
    d = json.loads(RESULT.read_text())
    e = d["policies"]["enzyme_safe"]
    assert e["pooled"]["r2e_all"]["delta"]["mrr"] > -0.002
    assert e["pooled"]["e2r_all"]["delta"]["mrr"] > 0
    assert e["pooled"]["r2e_lt0p3"]["delta"]["hit_at_10"] < -0.01
    assert e["pooled"]["r2e_lt0p3"]["delta"]["median_best_positive_rank"] > 50
    assert any(x.startswith("r2e_lt0p3:") for x in e["failures"])
