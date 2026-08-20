from pathlib import Path
from types import SimpleNamespace
import pytest

from experimental_touch_cascade.safety import SeparationError, assert_runtime_separation


def profile(tmp_path, evidence, run):
    repo=tmp_path/"repo"; repo.mkdir(exist_ok=True)
    er=repo/"local_candidate_libraries"/"experimental_touch_evidence"
    rr=repo/"local_candidate_libraries"/"experimental_touch_runs"
    return SimpleNamespace(
        repo_root=repo,
        source=SimpleNamespace(candidates_path=repo/"source.csv", metadata_path=None),
        runtime=SimpleNamespace(evidence_db=evidence,run_root=run,allowed_evidence_root=er,allowed_run_root=rr),
    )


def test_allowed_roots(tmp_path):
    repo=tmp_path/"repo"; repo.mkdir()
    p=profile(tmp_path,repo/"local_candidate_libraries"/"experimental_touch_evidence"/"x"/"e.sqlite",repo/"local_candidate_libraries"/"experimental_touch_runs"/"x")
    assert_runtime_separation(p)


def test_rejects_production_collision(tmp_path):
    repo=tmp_path/"repo"; repo.mkdir()
    p=profile(tmp_path,repo/"external_repos"/"igem_database"/"e.sqlite",repo/"local_candidate_libraries"/"experimental_touch_runs"/"x")
    with pytest.raises(SeparationError):
        assert_runtime_separation(p)
