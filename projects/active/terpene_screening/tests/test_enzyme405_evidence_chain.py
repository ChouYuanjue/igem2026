from copy import deepcopy

from projects.active.terpene_screening.audit_enzyme405_evidence_chain import audit_evidence_chain


def _fixtures():
    recipe = {
        "hidden_dim": 768, "embedding_dim": 320, "dropout": 0.1,
        "epochs": 8, "steps_per_epoch": 60, "reaction_batch_size": 64,
        "protein_batch_size": 48, "neighbor_k": 32, "hard_negatives": 80,
        "random_negatives": 8, "temperature": 0.035, "topk": 10,
        "topk_weight": 1.0, "margin": 0.12, "r2e_weight": 0.98,
        "learning_rate": 0.0003, "weight_decay": 0.0001,
        "reaction_novelty_repeat": 0,
    }
    freeze = {"selected_before_enzyme405_reveal": True, "selected_candidate": "rankstrong_r2e98", "selected_recipe": recipe}
    native = {"top1_sr": 0.1, "top3_sr": 0.2, "top5_sr": 0.3, "top10_sr": 0.5, "top10_dcg": 0.4, "top1_percent_ef": 30.0}
    result = {
        "reservoir_mode": "full_official", "target_labels_used_for_routing": False,
        "model_dir": "/tmp/rankstrong_r2e98", "queries": 295, "candidate_uids": 8615,
        "model_summary": {
            "target_benchmark_labels_read": False, "target_benchmark_metadata_used_for_training": False,
            "dev_fold": -1, "n_source_pairs": 218537, "n_train_pairs": 218537,
            "model_config": {k: recipe[k] for k in ("hidden_dim", "embedding_dim", "dropout")},
            "training": {**{k: recipe[k] for k in (
                "epochs", "steps_per_epoch", "reaction_batch_size", "protein_batch_size", "neighbor_k",
                "hard_negatives", "random_negatives", "temperature", "topk", "topk_weight", "margin",
                "r2e_weight", "learning_rate", "weight_decay")}, "reaction_novelty_replay": {"repeat": 0}},
        },
        "neural_metrics": {"enzymecage_native_r2e": native},
        "protocol_aware_metrics": {"enzymecage_native_r2e": {"top10_sr": 1.0}},
    }
    bootstrap = {"score": "neural_score_only", "queries": 295, "metrics": {
        "native_sr1": {"estimate": 0.1}, "native_sr3": {"estimate": 0.2},
        "native_sr5": {"estimate": 0.3}, "native_sr10": {"estimate": 0.5},
        "native_dcg10": {"estimate": 0.4}, "native_ef1": {"estimate": 30.0},
    }}
    sequence = {"automatic_sequence_correction_performed": False, "benchmark_uids": 8615,
                "reference_covered_uids": 8000, "exact_match_uids": 7990, "mismatch_uids": 10,
                "mismatched_positive_uids": 1, "reference_missing_uids": 615}
    baseline = {"records": [{"scenario_id": "enzyme405", "model": "EnzymeCAGE", "source_type": "paper_reported",
                             "comparison_role": "context_only_author_report_not_primary_reproducible_baseline",
                             "metrics": {"top10_sr": 0.58, "top1_percent_ef": 36.0, "top10_dcg": 0.45}}]}
    local = {
        "model_selection_allowed": False,
        "paper_reported_metrics_role": "context_only_not_used_for_reproducible_delta",
        "enzyme405_100_local_reconstruction": {
            "support": {"valid_reactions": 99},
            "enzymecage": {"top5_sr": 0.29, "top10_sr": 0.58},
            "catalyst_frozen_same_support": {
                "model": "rankstrong_r2e98", "evaluation_role": "post_reveal_descriptive_only",
                "top5_sr": 0.30, "top10_sr": 0.50,
            },
        },
    }
    return freeze, result, bootstrap, sequence, baseline, local


def test_clean_evidence_chain_passes_and_uses_neural_metrics() -> None:
    report = audit_evidence_chain(*_fixtures())
    assert report["status"] == "pass"
    assert report["primary_score_family"] == "neural_score_only"
    assert report["reproducible_same_support_comparison"]["top5_sr"]["leader"] == "Catalyst"
    assert report["reproducible_same_support_comparison"]["top10_sr"]["leader"] == "EnzymeCAGE"
    assert report["paper_metric_context"]["role"] == "context_only_not_used_for_reproducible_delta"


def test_post_reveal_lineage_mismatch_fails() -> None:
    freeze, result, bootstrap, sequence, baseline, local = _fixtures()
    bad = deepcopy(result)
    bad["model_dir"] = "/tmp/different_candidate"
    report = audit_evidence_chain(freeze, bad, bootstrap, sequence, baseline, local)
    assert report["status"] == "fail"
    assert any("selected candidate/model directory" in value for value in report["errors"])
