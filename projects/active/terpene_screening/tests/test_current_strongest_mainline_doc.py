from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[4]


def test_current_retrieval_status_is_a_compatibility_redirect():
    text = (ROOT / "projects/active/terpene_screening/CURRENT_RETRIEVAL_STATUS.md").read_text()
    assert "Compatibility redirect" in text
    assert "canonical.json" in text and "terpene_v1.yaml" in text
    assert "BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json" in text
    assert "not" in text and "independent current-result document" in text


def test_historical_human_docs_are_redirects_not_current_authority():
    for name in (
        "CURRENT_STRONGEST_VS_ENZYMECAGE.md",
        "RETRIEVAL_CAPABILITY_SCORECARD.md",
        "RETRIEVAL_EVIDENCE_LEDGER.md",
    ):
        text = (ROOT / "projects/active/terpene_screening" / name).read_text()
        assert "canonical.json" in text
        assert "superseded" in text
        assert "历史原文" in text


def test_public_project_readme_describes_one_bime_rank_system():
    text = (ROOT / "projects/active/terpene_screening/README.md").read_text()
    assert "BiME-Rank" in text
    assert "Bidirectional Multi-Expert Learning-to-Rank" in text
    assert "bime-rank-production-routes-v2" in text
    assert "185,918 proteins" in text and "11,081 reactions" in text
    assert "availability-aware structural evidence" in text.lower()
    assert "Known-positive context" in text
    assert "Cost-aware execution" in text
    assert "not eight competing" in text


def test_current_scorecard_keeps_claim_scopes_separate():
    score = json.loads(
        (ROOT / "projects/active/terpene_screening/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json").read_text()
    )
    assert score["method"] == "BiME-Rank"
    assert score["candidate_universes"] == {"r2e_proteins": 185918, "e2r_reactions": 11081}
    rules = " ".join(score["presentation_rules"])
    assert "different candidate universes" in rules
    assert "known-positive" in rules
    zero = score["zero_shot_external"]
    assert "clipzyme_strict_temporal_same_support" in zero
    assert "enzyme405" in zero
    assert "tps_enzymecage_same_support" in zero
    assert "selenzyme_author_pool" in zero
    assert "conditional_known_positive" in score
    assert score["conditional_known_positive"]["r2e"]["claim_boundary"].startswith("one-known-positive")


def test_release_manifest_and_ci_use_master():
    manifest = json.loads((ROOT / "reproducibility/research_release_manifest.json").read_text())
    assert manifest["release_branch"] == "master"
    workflow = (ROOT / ".github/workflows/terpene-ci.yml").read_text()
    assert "branches: [master]" in workflow
    assert "branches: [main]" not in workflow
    assert "build_research_release_manifest.py" in workflow
    assert "git diff --exit-code -- reproducibility/research_release_manifest.json" in workflow
