import pytest

from projects.active.terpene_screening.run_enzymecage_orphan335_author_retrieval import AUTHOR, load_author_module


def test_author_retrieval_wrapper_only_supplies_missing_counter():
    if not (AUTHOR / "retrieve.py").is_file():
        pytest.skip("EnzymeCAGE upstream source is a pinned external repository and is not vendored in the portable release clone")
    m = load_author_module()
    import collections
    assert m.Counter is collections.Counter
    assert callable(m.run_retrieval)
