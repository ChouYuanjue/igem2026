from projects.active.terpene_screening.benchmark_baseline_provenance import payload, validate_record


def test_enzyme405_paper_baseline_is_explicitly_not_a_rerun() -> None:
    record = payload()["records"][0]
    assert record["scenario_id"] == "enzyme405"
    assert record["model"] == "EnzymeCAGE"
    assert record["source_type"] == "paper_reported"
    assert record["metrics"]["top10_sr"] == 0.5797
    assert record["common_ir_metrics"] is None


def test_paper_record_cannot_invent_common_ir_metrics() -> None:
    record = {
        "scenario_id": "enzyme405",
        "model": "EnzymeCAGE",
        "source_type": "paper_reported",
        "protocol": "author protocol",
        "metrics": {"top10_sr": 0.5},
        "common_ir_metrics": {"mrr": 0.9},
    }
    assert any("cannot invent" in error for error in validate_record(record))
