from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import hashlib
import re
from typing import Iterable
from urllib.parse import urlparse


_DOI_RE=re.compile(r"(?i)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)")
_PMCID_RE=re.compile(r"(?i)\b(PMC\d+)\b")
_PMID_PATTERNS=(
    re.compile(r"(?i)pubmed(?:\.ncbi\.nlm\.nih\.gov)?/(?:pubmed/)?(\d+)(?:[/?#]|$)"),
    re.compile(r"(?i)[?&](?:term|pmid)=(\d+)(?:[&#]|$)"),
)


def normalize_doi(value: str) -> str:
    value=str(value or '').strip()
    value=re.sub(r'(?i)^doi\s*:?\s*','',value)
    value=re.sub(r'(?i)^https?://(?:dx\.)?doi\.org/','',value)
    match=_DOI_RE.search(value)
    if not match:
        return ''
    return match.group(1).rstrip('.,);]').lower()


def extract_publication_identifiers(value: str) -> dict[str,str]:
    text=str(value or '').strip()
    out={'pmid':'','pmcid':'','doi':''}
    pmc=_PMCID_RE.search(text)
    if pmc:
        out['pmcid']=pmc.group(1).upper()
    for rx in _PMID_PATTERNS:
        m=rx.search(text)
        if m:
            out['pmid']=m.group(1)
            break
    out['doi']=normalize_doi(text)
    return out


def stable_source_key(source: str) -> str:
    return hashlib.sha256(str(source).strip().encode('utf-8')).hexdigest()[:20]


class _CitationMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str,list[str]]={}
        self._in_title=False
        self._title=[]

    def handle_starttag(self,tag,attrs):
        a={str(k).lower():str(v) for k,v in attrs if v is not None}
        if tag.lower()=='meta':
            key=(a.get('name') or a.get('property') or '').lower().strip()
            value=(a.get('content') or '').strip()
            if key and value:
                self.meta.setdefault(key,[]).append(value)
        elif tag.lower()=='title':
            self._in_title=True

    def handle_endtag(self,tag):
        if tag.lower()=='title':
            self._in_title=False

    def handle_data(self,data):
        if self._in_title:
            self._title.append(data)

    @property
    def title(self) -> str:
        return ' '.join(' '.join(self._title).split())


def extract_citation_metadata(html: str) -> dict[str,str]:
    parser=_CitationMetaParser()
    parser.feed(str(html or ''))
    meta=parser.meta
    def first(*keys: str) -> str:
        for key in keys:
            vals=meta.get(key.lower()) or []
            if vals:
                return vals[0].strip()
        return ''
    doi=normalize_doi(first('citation_doi','dc.identifier','prism.doi'))
    pmid=first('citation_pmid','pmid')
    pmcid=first('citation_pmcid','pmcid')
    if pmcid:
        m=_PMCID_RE.search(pmcid)
        pmcid=m.group(1).upper() if m else ''
    if pmid and not pmid.isdigit():
        pmid=''
    title=first('citation_title','dc.title','og:title') or parser.title
    return {
        'doi':doi,
        'pmid':pmid,
        'pmcid':pmcid,
        'title':' '.join(title.split()),
    }


def normalized_title(value: str) -> str:
    value=re.sub(r'[^A-Za-z0-9]+',' ',str(value or '').lower())
    return ' '.join(value.split())


@dataclass(frozen=True)
class PublicationSeed:
    source: str
    source_key: str
    source_row_count: int=0
    unique_enzyme_count: int=0
    unique_reaction_count: int=0

    def to_dict(self) -> dict:
        return asdict(self)


def deduplicate_seeds(seeds: Iterable[PublicationSeed]) -> list[PublicationSeed]:
    merged: dict[str,PublicationSeed]={}
    for seed in seeds:
        key=seed.source_key
        prev=merged.get(key)
        if prev is None:
            merged[key]=seed
            continue
        merged[key]=PublicationSeed(
            source=prev.source,
            source_key=key,
            source_row_count=max(prev.source_row_count,seed.source_row_count),
            unique_enzyme_count=max(prev.unique_enzyme_count,seed.unique_enzyme_count),
            unique_reaction_count=max(prev.unique_reaction_count,seed.unique_reaction_count),
        )
    return sorted(merged.values(),key=lambda x:(-x.source_row_count,x.source))


def looks_like_url(value: str) -> bool:
    try:
        parsed=urlparse(str(value).strip())
    except ValueError:
        return False
    return parsed.scheme in {'http','https'} and bool(parsed.netloc)


def publication_seed_values(value: str) -> list[tuple[str,str]]:
    """Return all useful discovery seeds encoded in one publication string."""
    text=str(value or '').strip()
    if not text:
        return []
    ids=extract_publication_identifiers(text)
    out=[]
    for kind in ('pmid','pmcid','doi'):
        if ids.get(kind):
            out.append((kind,str(ids[kind])))
    if looks_like_url(text):
        out.append(('url',text))
    elif not out:
        out.append(('title',text))
    seen=set(); unique=[]
    for item in out:
        if item in seen: continue
        seen.add(item); unique.append(item)
    return unique
