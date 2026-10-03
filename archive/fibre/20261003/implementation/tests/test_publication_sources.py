from projects.active.fibre.evidence.publication_sources import (
    extract_citation_metadata,
    extract_publication_identifiers,
    normalize_doi,
    normalized_title,
    publication_seed_values,
)


def test_publication_identifier_parsing_supports_common_seed_forms():
    assert extract_publication_identifiers(
        'https://pubmed.ncbi.nlm.nih.gov/21818683/'
    )['pmid']=='21818683'
    assert extract_publication_identifiers(
        'https://pmc.ncbi.nlm.nih.gov/articles/PMC6945850/'
    )['pmcid']=='PMC6945850'
    assert normalize_doi('DOI 10.1515/HF.2009.019')=='10.1515/hf.2009.019'
    assert extract_publication_identifiers(
        'https://doi.org/10.1021/acscatal.5c04814'
    )['doi']=='10.1021/acscatal.5c04814'


def test_citation_meta_parser_extracts_standard_html_metadata():
    html='''<html><head>
    <meta name="citation_title" content="An enzyme paper">
    <meta name="citation_doi" content="10.1000/XYZ.1">
    <meta name="citation_pmid" content="123456">
    </head></html>'''
    x=extract_citation_metadata(html)
    assert x['title']=='An enzyme paper'
    assert x['doi']=='10.1000/xyz.1'
    assert x['pmid']=='123456'


def test_publication_seed_values_support_mixed_identifiers_urls_and_titles():
    seeds=publication_seed_values('https://doi.org/10.1000/ABC.1')
    assert ('doi','10.1000/abc.1') in seeds
    assert ('url','https://doi.org/10.1000/ABC.1') in seeds
    assert publication_seed_values('A paper title with no identifier') == [
        ('title','A paper title with no identifier')
    ]


def test_title_normalization_is_stable_for_bibliographic_matching():
    assert normalized_title('A β-Pinene Synthase: Study!') == (
        'a pinene synthase study'
    )
