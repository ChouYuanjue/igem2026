from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
def test_current_strongest_is_success_mainline_not_failure_roadmap():
 text=(ROOT/'projects/active/terpene_screening/CURRENT_STRONGEST_VS_ENZYMECAGE.md').read_text()
 assert '## Canonical successful evolution' in text
 assert 'Geometry-bounded reaction-center residual V3' in text and 'cap **0.10**' in text
 assert 'results/catalyst_clean_mainline_v1/r2e_center_bounded_cap0p1' in text
 assert '## Current routed-system contract' in text and 'score-fused' in text
 assert '## Historical evidence boundary' in text
 assert '## Current improvement target' not in text
 assert text.index('## Canonical successful evolution') < text.index('## Historical evidence boundary')
