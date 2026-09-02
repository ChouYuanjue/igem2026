from pathlib import Path
from projects.active.terpene_screening.prepare_enzymarc_open_world_v1 import parse_fasta

def test_parse_enzymarc_header(tmp_path:Path):
 p=tmp_path/'x.fa'; p.write_text('>P12345|catalytic|EC:1.2.3.4\nACDE\n>Q9XYZ1|5A|EC:2.3.4.5\nFGHI\n')
 x=list(parse_fasta(p,'catalytic')); assert x[0]['parent_accession']=='P12345' and x[0]['original_ec']=='1.2.3.4' and x[0]['decoy_sequence']=='ACDE'; assert len(x)==2
