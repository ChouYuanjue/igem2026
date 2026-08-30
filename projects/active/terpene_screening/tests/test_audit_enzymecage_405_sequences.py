import pandas as pd

from projects.active.terpene_screening.audit_enzymecage_405_sequences import audit_sequences


def test_sequence_audit_reports_mismatch_without_correcting() -> None:
    pairs = pd.DataFrame([
        {"UniprotID": "P1", "sequence": "AAAA", "Label": 1},
        {"UniprotID": "P2", "sequence": "BBBB", "Label": 0},
        {"UniprotID": "P3", "sequence": "BBBB", "Label": 0},
    ])
    reference = pd.DataFrame([
        {"protein_id": "P1", "sequence": "AAAA"},
        {"protein_id": "P2", "sequence": "CCCC"},
    ])
    frame, groups, summary = audit_sequences(pairs, reference)
    assert summary["exact_match_uids"] == 1
    assert summary["mismatch_uids"] == 1
    assert summary["reference_missing_uids"] == 1
    assert summary["automatic_sequence_correction_performed"] is False
    assert frame.set_index("UniprotID").loc["P2", "sequence"] == "BBBB"
    duplicate = groups[groups.uid_count.eq(2)].iloc[0]
    assert duplicate.uids == "P2;P3"


def test_sequence_audit_rejects_one_uid_with_multiple_benchmark_sequences() -> None:
    pairs = pd.DataFrame([
        {"UniprotID": "P1", "sequence": "AAAA", "Label": 1},
        {"UniprotID": "P1", "sequence": "AAAB", "Label": 0},
    ])
    reference = pd.DataFrame([{"protein_id": "P1", "sequence": "AAAA"}])
    try:
        audit_sequences(pairs, reference)
    except ValueError as exc:
        assert "multiple sequences" in str(exc)
    else:
        raise AssertionError("multiple benchmark sequences for one UID must fail")
