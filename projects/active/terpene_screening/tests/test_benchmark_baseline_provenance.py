from projects.active.terpene_screening.benchmark_baseline_provenance import payload, validate_record
from projects.active.terpene_screening.model_capability_registry import DEFAULT_SCENARIOS


def test_enzyme405_paper_baseline_is_explicitly_not_a_rerun() -> None:
    record = next(item for item in payload()["records"] if item["scenario_id"] == "enzyme405")
    assert record["scenario_id"] == "enzyme405"
    assert record["model"] == "EnzymeCAGE"
    assert record["status"] == "complete"
    assert record["source_type"] == "paper_reported"
    assert record["metrics"]["top10_sr"] == 0.5797
    assert record["common_ir_metrics"] is None


def test_paper_record_cannot_invent_common_ir_metrics() -> None:
    record = {
        "scenario_id": "enzyme405",
        "model": "EnzymeCAGE",
        "status": "complete",
        "source_type": "paper_reported",
        "protocol": "author protocol",
        "metrics": {"top10_sr": 0.5},
        "common_ir_metrics": {"mrr": 0.9},
    }
    assert any("cannot invent" in error for error in validate_record(record))


def test_every_registered_scenario_has_exactly_one_enzymecage_record() -> None:
    records = payload()["records"]
    expected = {item.scenario_id for item in DEFAULT_SCENARIOS}
    observed = [item["scenario_id"] for item in records]
    assert set(observed) == expected
    assert len(observed) == len(set(observed))
    assert all(item["model"] == "EnzymeCAGE" for item in records)


def test_unavailable_baselines_are_explicit_na_with_reason_and_no_metrics() -> None:
    records = payload()["records"]
    unavailable = [item for item in records if item["status"] == "na"]
    assert unavailable
    for record in unavailable:
        assert record["source_type"] == "not_applicable"
        assert record["metrics"] is None
        assert record["incompatibility_reason"].strip()
        assert validate_record(record) == []
