from experimental_touch_cascade.pipeline import _article_exact_identity


def article(text):
    return {"title": text, "abstract": ""}


def test_linked_uniprot_reference_is_identity_confirmed_without_alias_in_abstract():
    exact, why = _article_exact_identity(article("Biochemical characterization of a diterpene cyclase"), ["P0DPK6"], True)
    assert exact
    assert why == ["linked_exact_sequence_uniprot_reference"]


def test_alias_search_requires_alias_in_article_text():
    assert _article_exact_identity(article("Biochemical characterization of Q93NX6"), ["Q93NX6"], False)[0]
    assert not _article_exact_identity(article("Biochemical characterization of a related enzyme"), ["Q93NX6"], False)[0]
