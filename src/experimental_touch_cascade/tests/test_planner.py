from experimental_touch_cascade.planner import Stage1Signals, stage1_decision, stage2_decision


def s(**kw):
    base=dict(public_exact=True,best_pe_level=4,reviewed=False,has_pdb=False,structured_experiment=False,force_stage2=False,force_stage3=False)
    base.update(kw)
    return Stage1Signals(**base)


def test_stage1_filters_low_signal():
    assert stage1_decision(s())[0] == "FINALIZE"
    assert stage1_decision(s(public_exact=False))[0] == "FINALIZE"
    assert stage1_decision(s(best_pe_level=1))[0] == "PROMOTE"
    assert stage1_decision(s(best_pe_level=2))[0] == "PROMOTE"
    assert stage1_decision(s(reviewed=True))[0] == "PROMOTE"
    assert stage1_decision(s(has_pdb=True))[0] == "PROMOTE"
    assert stage1_decision(s(force_stage2=True))[0] == "PROMOTE"


def test_stage2_only_promotes_deep_targets_or_publication_hints():
    assert stage2_decision("T1", False, False)[0] == "FINALIZE"
    assert stage2_decision("T1", True, False)[0] == "PROMOTE"
    assert stage2_decision("T4", True, False)[0] == "FINALIZE"
    assert stage2_decision("T4", False, True)[0] == "PROMOTE"
