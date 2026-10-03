from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
import xml.etree.ElementTree as ET


CUE_PATTERNS={
    'pH':re.compile(r'\bpH\b',re.I),
    'temperature':re.compile(r'(?:\btemperature\b|°\s*C\b|\bdegrees?\s*C\b)',re.I),
    'metal_or_cofactor':re.compile(
        r'\b(?:cofactor|metal ion|MgCl2|MnCl2|ZnCl2|FeCl2|CaCl2|CoCl2|NiCl2|'
        r'Mg\(2\+\)|Mn\(2\+\)|Zn\(2\+\)|Fe\(2\+\)|Ca\(2\+\)|Co\(2\+\)|Ni\(2\+\)|'
        r'Mg2\+|Mn2\+|Zn2\+|Fe2\+|Ca2\+|Co2\+|Ni2\+)\b',re.I
    ),
    'kinetic':re.compile(
        r'\b(?:kcat|k_cat|K\s*m|K_m|Michaelis|catalytic efficiency|specific activity|'
        r'initial rate|turnover number|Vmax|V_max|kinetic)\b',re.I
    ),
    'concentration':re.compile(
        r'\b\d+(?:\.\d+)?\s*(?:mM|µM|μM|uM|nM|M|mg\s*/\s*mL|g\s*/\s*L)\b',re.I
    ),
    'assay_context':re.compile(
        r'\b(?:enzyme assay|activity assay|reaction mixture|assay mixture|assay(?:ed|ing)?|'
        r'incubat(?:e|ed|ing|ion)?|buffer|substrate concentration|enzyme concentration|reaction volume)\b',re.I
    ),
    'conversion_or_yield':re.compile(
        r'\b(?:conversion|yield|product formation|turnover|percent conversion|% conversion)\b',re.I
    ),
    'selectivity':re.compile(
        r'\b(?:enantioselect|stereoselect|regioselect|diastereoselect|product ratio|'
        r'enantiomeric excess|\bee\b)\b',re.I
    ),
    'stability':re.compile(
        r'\b(?:half-life|half life|thermal stability|thermostab|residual activity|'
        r'melting temperature|Tm\b|temperature optimum|pH optimum)\b',re.I
    ),
    'inactive_or_detection_limit':re.compile(
        r'\b(?:no detectable activity|not detected|inactive|below detection|'
        r'limit of detection|detection limit|no activity|failed to convert)\b',re.I
    ),
    'substrate_scope':re.compile(
        r'\b(?:substrate scope|substrate specificity|promiscu|accepted substrate|'
        r'product profile|substrate preference)\b',re.I
    ),
    'expression_or_purification':re.compile(
        r'\b(?:heterologous expression|expressed in|purified by|affinity chromatography|'
        r'Ni-NTA|cell lysate|protein purification)\b',re.I
    ),
}


@dataclass(frozen=True)
class SourceBlock:
    block_id: str
    block_type: str
    source_index: int
    section_path: tuple[str,...]
    text: str
    cue_families: tuple[str,...]

    def to_dict(self):
        row=asdict(self)
        row['section_path']=list(self.section_path)
        row['cue_families']=list(self.cue_families)
        return row


def _tag(node) -> str:
    return str(node.tag).split('}')[-1]


def _clean_text(node) -> str:
    return ' '.join(' '.join(node.itertext()).split())


def _parent_map(root):
    return {child:parent for parent in root.iter() for child in parent}


def _section_path(node,parent) -> tuple[str,...]:
    titles=[]
    cur=node
    while cur in parent:
        cur=parent[cur]
        if _tag(cur)!='sec':
            continue
        title=next((x for x in list(cur) if _tag(x)=='title'),None)
        if title is not None:
            text=_clean_text(title)
            if text:
                titles.append(text)
    return tuple(reversed(titles))


def cue_families(text: str) -> tuple[str,...]:
    return tuple(name for name,rx in CUE_PATTERNS.items() if rx.search(text))


def iter_jats_blocks(xml_text: str):
    root=ET.fromstring(xml_text)
    parent=_parent_map(root)
    index=0
    for node in root.iter():
        kind=_tag(node)
        if kind not in {'p','table-wrap'}:
            continue
        text=_clean_text(node)
        if len(text)<25:
            continue
        families=cue_families(text)
        block_type='table' if kind=='table-wrap' else 'paragraph'
        digest=hashlib.sha256(
            f'{block_type}\x1f{index}\x1f{text}'.encode('utf-8')
        ).hexdigest()[:20]
        yield SourceBlock(
            block_id=digest,block_type=block_type,source_index=index,
            section_path=_section_path(node,parent),text=text,cue_families=families,
        )
        index+=1


def extract_candidate_blocks(xml_text: str, families: set[str] | None=None) -> list[SourceBlock]:
    out=[]
    for block in iter_jats_blocks(xml_text):
        if not block.cue_families:
            continue
        if families is not None and not (set(block.cue_families)&families):
            continue
        out.append(block)
    return out


def explicit_quantities(text: str) -> list[dict[str,str]]:
    patterns=(
        (r'(?P<value>[−–-]?\d+(?:\.\d+)?)\s*(?P<unit>°C|° C|degC|degrees? C)',re.I),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mM|µM|μM|uM|nM|M)\b',0),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mg\s*(?:/|\s+)m[lL](?:[−-]?1)?|g\s*(?:/|\s+)L(?:[−-]?1)?)',re.I),
        (r'(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?|h|minutes?|mins?|min|seconds?|secs?|sec|s)\b',re.I),
    )
    out=[]; seen=set()
    for pattern,flags in patterns:
        for m in re.finditer(pattern,text,flags):
            key=(m.group(0),m.start())
            if key in seen: continue
            seen.add(key)
            out.append({'raw':m.group(0),'value':m.group('value'),'unit':m.group('unit')})
    for m in re.finditer(r'\bpH\s*(?P<value>\d+(?:\.\d+)?)',text,re.I):
        key=(m.group(0),m.start())
        if key in seen: continue
        seen.add(key)
        out.append({'raw':m.group(0),'value':m.group('value'),'unit':'pH'})
    return out
