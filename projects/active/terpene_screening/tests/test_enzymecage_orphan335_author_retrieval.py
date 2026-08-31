from projects.active.terpene_screening.run_enzymecage_orphan335_author_retrieval import load_author_module

def test_author_retrieval_wrapper_only_supplies_missing_counter():
    m=load_author_module()
    import collections
    assert m.Counter is collections.Counter
    assert callable(m.run_retrieval)
