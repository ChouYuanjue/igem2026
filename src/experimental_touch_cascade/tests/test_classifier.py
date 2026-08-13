from experimental_touch_cascade.classifier import SummaryEvidence, classify


def test_simple_tiers_only():
    assert classify(SummaryEvidence(False), 1)[0] == "T0"
    assert classify(SummaryEvidence(True, best_pe_level=3), 1)[0] == "T1"
    assert classify(SummaryEvidence(True, best_pe_level=1), 1)[0] == "T2"
    assert classify(SummaryEvidence(True, has_pdb=True), 2)[0] == "T3"
    assert classify(SummaryEvidence(True, functional_experimental=True), 2)[0] == "T4"
    assert classify(SummaryEvidence(True, functional_experimental=True, kinetics_present=True), 2)[0] == "T5"
