from __future__ import annotations

from projects.active.terpene_screening.evaluate_fair_cage_fewshot import stable_trial_seed


def test_stable_trial_seed_is_deterministic_and_trial_specific():
    first = stable_trial_seed("RHEA:12345", 2, 7)
    second = stable_trial_seed("RHEA:12345", 2, 7)
    different_rep = stable_trial_seed("RHEA:12345", 2, 8)
    different_reaction = stable_trial_seed("RHEA:54321", 2, 7)

    assert first == second
    assert first != different_rep
    assert first != different_reaction
