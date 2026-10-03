from projects.active.fibre.evidence.publication_context import (
    extract_candidate_blocks, explicit_quantities,
)


XML='''<article><body>
<sec><title>Enzyme assays</title>
<p>The purified enzyme was assayed in 50 mM HEPES at pH 7.2 with 10 mM MgCl2 and incubated at 30 °C for 1 h.</p>
<table-wrap><label>Table 1</label><table><tr><td>Substrate A</td><td>no detectable activity</td></tr></table></table-wrap>
</sec>
<sec><title>Protein purification</title>
<p>The protein was purified by Ni-NTA affinity chromatography and stored at -80 °C.</p>
</sec>
</body></article>'''


def test_candidate_extraction_covers_paragraphs_and_tables_with_section_paths():
    blocks=extract_candidate_blocks(XML)
    assert len(blocks)==3
    assay=blocks[0]
    assert assay.block_type=='paragraph'
    assert assay.section_path==('Enzyme assays',)
    assert {'pH','temperature','metal_or_cofactor','concentration','assay_context'} <= set(assay.cue_families)
    table=blocks[1]
    assert table.block_type=='table'
    assert 'inactive_or_detection_limit' in table.cue_families
    assert table.section_path==('Enzyme assays',)


def test_purification_is_detected_as_a_separate_cue_not_silently_dropped():
    blocks=extract_candidate_blocks(XML)
    purification=blocks[2]
    assert purification.section_path==('Protein purification',)
    assert 'expression_or_purification' in purification.cue_families
    assert 'temperature' in purification.cue_families


def test_explicit_quantities_are_source_bound_and_unit_preserving():
    q=explicit_quantities('pH 7.2, 10 mM MgCl2, 30 °C, 1 h')
    units={x['unit'] for x in q}
    assert 'pH' in units
    assert 'mM' in units
    assert '°C' in units
    assert 'h' in units
