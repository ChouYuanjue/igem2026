#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

PROTOCOL_ID = 'experimental-touch-v2-20260813'
UA = 'NJU-lab-experimental-touch-profiler/2.0 (research audit)'
UNIPARC_URL = 'https://rest.uniprot.org/uniparc/search'
UNIPROTKB_URL = 'https://rest.uniprot.org/uniprotkb/search'
ECO_EXP = 'ECO:0000269'

UNIPROT_FIELDS = [
    'accession', 'reviewed', 'protein_existence', 'xref_pdb',
    'cc_catalytic_activity', 'cc_function', 'cc_activity_regulation',
    'kinetics', 'cc_mass_spectrometry'
]


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def md5_seq(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield i // n, xs[i:i+n]


def get_with_retry(url, params, timeout=75, retries=8):
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(
                url, params=params, timeout=timeout,
                headers={'User-Agent': UA, 'Accept': 'text/tab-separated-values'}
            )
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last = RuntimeError(f'HTTP {r.status_code}: {r.text[:500]}')
                retry_after = r.headers.get('Retry-After')
                if retry_after:
                    try: delay = max(float(retry_after), 0.5)
                    except Exception: delay = 1.0 * (2 ** min(attempt, 5))
                else:
                    delay = 0.75 * (2 ** min(attempt, 5))
                time.sleep(delay)
                continue
            raise RuntimeError(f'HTTP {r.status_code}: {r.text[:1000]}')
        except Exception as e:
            last = e
            time.sleep(0.75 * (2 ** min(attempt, 5)))
    raise last


def normalize_uniprot_acc(x: str) -> str:
    return re.sub(r'\.\d+$', '', str(x).strip())

def split_accessions(s):
    if not s or pd.isna(s):
        return []
    return [normalize_uniprot_acc(x) for x in re.split(r'[;,\s]+', str(s).strip()) if x]


def read_json_gz(path: Path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def write_json_gz(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with gzip.open(tmp, 'wt', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
    tmp.replace(path)


def uniparc_batch(batch_idx, items, cache_dir: Path):
    cache = cache_dir / f'batch_{batch_idx:05d}.json.gz'
    batch_md5 = [x['sequence_md5'] for x in items]
    key = sha256_text('\n'.join(batch_md5))
    if cache.exists():
        obj = read_json_gz(cache)
        if obj.get('input_key') == key:
            return obj

    q = ' OR '.join(f'checksum:{x}' for x in batch_md5)
    r = get_with_retry(UNIPARC_URL, {
        'query': f'({q})', 'format': 'tsv',
        'fields': 'upi,sequence,accession,first_seen,last_seen',
        'size': 500,
    })
    rows = []
    reader = csv.DictReader(io.StringIO(r.text), delimiter='\t')
    for z in reader:
        seq = ''.join((z.get('Sequence') or '').split()).upper().rstrip('*')
        if not seq:
            continue
        m = md5_seq(seq)
        if m not in set(batch_md5):
            raise RuntimeError(f'UniParc returned sequence outside checksum query: {m}')
        rows.append({
            'sequence_md5': m,
            'uniparc_id': z.get('Entry', ''),
            'uniprot_accessions': split_accessions(z.get('UniProtKB', '')),
            'first_seen': z.get('First seen', ''),
            'last_seen': z.get('Last seen', ''),
        })
    obj = {
        'input_key': key,
        'input_count': len(items),
        'hit_count': len(rows),
        'uniprot_release': r.headers.get('X-UniProt-Release', ''),
        'uniprot_release_date': r.headers.get('X-UniProt-Release-Date', ''),
        'queried_utc': pd.Timestamp.utcnow().isoformat(),
        'records': rows,
    }
    write_json_gz(cache, obj)
    return obj


def uniprot_batch(batch_idx, accessions, cache_dir: Path):
    cache = cache_dir / f'batch_{batch_idx:05d}.json.gz'
    key = sha256_text('\n'.join(accessions))
    if cache.exists():
        obj = read_json_gz(cache)
        if obj.get('input_key') == key:
            return obj

    q = ' OR '.join(f'accession:{x}' for x in accessions)
    r = get_with_retry(UNIPROTKB_URL, {
        'query': f'({q})', 'format': 'tsv',
        'fields': ','.join(UNIPROT_FIELDS), 'size': 500,
    })
    reader = csv.DictReader(io.StringIO(r.text), delimiter='\t')
    rows = []
    for z in reader:
        acc = str(z.get('Entry', '')).strip()
        if not acc:
            continue
        rows.append({
            'accession': acc,
            'reviewed': str(z.get('Reviewed', '')).strip(),
            'protein_existence': str(z.get('Protein existence', '')).strip(),
            'pdb': str(z.get('PDB', '')).strip(),
            'catalytic_activity': str(z.get('Catalytic activity', '')).strip(),
            'function': str(z.get('Function [CC]', '')).strip(),
            'activity_regulation': str(z.get('Activity regulation', '')).strip(),
            'kinetics': str(z.get('Kinetics', '')).strip(),
            'mass_spectrometry': str(z.get('Mass spectrometry', '')).strip(),
        })
    obj = {
        'input_key': key,
        'input_count': len(accessions),
        'returned_count': len(rows),
        'uniprot_release': r.headers.get('X-UniProt-Release', ''),
        'uniprot_release_date': r.headers.get('X-UniProt-Release-Date', ''),
        'queried_utc': pd.Timestamp.utcnow().isoformat(),
        'records': rows,
    }
    write_json_gz(cache, obj)
    return obj


def pe_level(text: str):
    t = (text or '').lower()
    if 'protein level' in t: return 1
    if 'transcript level' in t: return 2
    if 'homology' in t: return 3
    if 'predicted' in t: return 4
    if 'uncertain' in t: return 5
    return None


def split_pdb(s: str):
    if not s: return []
    return sorted(set(x.strip() for x in re.split(r'[;,\s]+', s) if x.strip()))


def classify(md5, upi, uniprot_map):
    if upi is None:
        return {
            'experimental_touch_level': 'T0',
            'experimental_touch_numeric': 0,
            'experimental_touch_label': 'exact sequence not found in UniParc public sequence archive snapshot',
            'touch_t2plus_eligible': False,
            'best_pe_level': None,
            'best_pe_text': '',
            'reviewed_exact': False,
            'pdb_exact_ids': '',
            'experimental_functional_eco': False,
            'experimental_catalytic_eco': False,
            'kinetics_present': False,
            'mass_spectrometry_annotation_present': False,
            'public_evidence_confidence': 'high_for_sequence-archive_absence;low_for_claim_of_no_unlinked_or_unpublished_experiment',
        }

    norm_accs = sorted(set(normalize_uniprot_acc(a) for a in upi['uniprot_accessions']))
    audits = [uniprot_map[a] for a in norm_accs if a in uniprot_map]
    levels = [pe_level(a['protein_existence']) for a in audits]
    levels = [x for x in levels if x is not None]
    best = min(levels) if levels else None
    best_text = ';'.join(sorted(set(a['protein_existence'] for a in audits if a['protein_existence'])))
    reviewed = any(a['reviewed'].lower() == 'reviewed' for a in audits)
    pdb = sorted({p for a in audits for p in split_pdb(a['pdb'])})
    catalytic_exp = any(ECO_EXP in a['catalytic_activity'] for a in audits)
    function_exp = any(
        ECO_EXP in (a['catalytic_activity'] + ' ' + a['function'] + ' ' + a['activity_regulation'])
        for a in audits
    )
    kinetics = any(bool(a['kinetics'].strip()) for a in audits)
    ms = any(bool(a['mass_spectrometry'].strip()) for a in audits)

    # v2 is deliberately conservative: every T2-T5 requires positive protein-level or deeper evidence.
    if function_exp and (pdb or kinetics):
        level, num, label = 'T5', 5, 'experimental function/catalysis plus experimental structure and/or kinetics'
    elif function_exp:
        level, num, label = 'T4', 4, 'exact-sequence UniProt annotation has published experimental function/catalysis evidence (ECO:0000269)'
    elif pdb:
        level, num, label = 'T3', 3, 'exact-sequence UniProt accession has experimental PDB structure cross-reference'
    elif best == 1 or ms:
        level, num, label = 'T2', 2, 'positive protein-level experimental existence evidence for exact-sequence UniProt entry'
    else:
        level, num, label = 'T1', 1, 'exact sequence is public, but no positive protein-level experimental evidence was found in linked exact-sequence UniProt entries'

    conf = ('high_for_positive_linked_public_evidence' if num >= 2 else
            'high_for_public_sequence_presence;moderate_for_absence_of_unlinked_experimental_literature')
    return {
        'experimental_touch_level': level,
        'experimental_touch_numeric': num,
        'experimental_touch_label': label,
        'touch_t2plus_eligible': num >= 2,
        'best_pe_level': best,
        'best_pe_text': best_text,
        'reviewed_exact': reviewed,
        'pdb_exact_ids': ';'.join(pdb),
        'experimental_functional_eco': function_exp,
        'experimental_catalytic_eco': catalytic_exp,
        'kinetics_present': kinetics,
        'mass_spectrometry_annotation_present': ms,
        'public_evidence_confidence': conf,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--library-dir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--uniparc-batch-size', type=int, default=100)
    ap.add_argument('--uniprot-batch-size', type=int, default=80)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--output-name', default='experimental_touch_v2')
    args = ap.parse_args()

    lib = args.library_dir.resolve()
    cdir = lib / 'candidates'
    cache_root = cdir / f'{args.output_name}_cache'
    uc = cache_root / 'uniparc'; kc = cache_root / 'uniprotkb'
    uc.mkdir(parents=True, exist_ok=True); kc.mkdir(parents=True, exist_ok=True)

    model = pd.read_csv(cdir / 'candidates_model.csv', dtype=str).fillna('')
    if args.limit:
        model = model.iloc[:args.limit].copy()
    if model.enzyme_id.duplicated().any():
        raise RuntimeError('candidate enzyme IDs are not unique')
    model['sequence'] = model.sequence.astype(str).str.replace(r'\s+', '', regex=True).str.upper().str.rstrip('*')
    model['sequence_md5'] = model.sequence.map(md5_seq)
    model['sequence_sha256_check'] = model.sequence.map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    items = model[['enzyme_id','sequence_md5']].to_dict('records')
    print(f'INPUT candidates={len(model):,}', flush=True)

    # Phase 1: exact public sequence presence in UniParc.
    ubatches = list(chunks(items, args.uniparc_batch_size))
    uobjs = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(uniparc_batch, i, b, uc): i for i, b in ubatches}
        for n, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            uobjs[i] = f.result()
            if n % 25 == 0 or n == len(futs):
                hits = sum(x['hit_count'] for x in uobjs.values())
                print(f'UNIPARC batches={n}/{len(futs)} hits_so_far={hits:,}', flush=True)

    up_by_md5 = {}
    releases = set(); release_dates = set()
    for i in sorted(uobjs):
        obj = uobjs[i]
        if obj.get('uniprot_release'): releases.add(obj['uniprot_release'])
        if obj.get('uniprot_release_date'): release_dates.add(obj['uniprot_release_date'])
        for r in obj['records']:
            if r['sequence_md5'] in up_by_md5:
                raise RuntimeError(f'duplicate UniParc exact sequence result: {r["sequence_md5"]}')
            up_by_md5[r['sequence_md5']] = r
    print(f'UNIPARC_COMPLETE exact_public={len(up_by_md5):,} T0_preliminary={len(model)-len(up_by_md5):,}', flush=True)

    accessions = sorted({normalize_uniprot_acc(a) for r in up_by_md5.values() for a in r['uniprot_accessions'] if normalize_uniprot_acc(a)})
    print(f'UNIPROTKB_ACCESSIONS unique={len(accessions):,}', flush=True)

    # Phase 2: positive experimental evidence from exact-sequence UniProtKB accessions.
    kbatches = list(chunks(accessions, args.uniprot_batch_size))
    kobjs = {}
    if kbatches:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(uniprot_batch, i, b, kc): i for i, b in kbatches}
            for n, f in enumerate(as_completed(futs), 1):
                i = futs[f]
                kobjs[i] = f.result()
                if n % 25 == 0 or n == len(futs):
                    returned = sum(x['returned_count'] for x in kobjs.values())
                    print(f'UNIPROTKB batches={n}/{len(futs)} active_records_so_far={returned:,}', flush=True)

    uniprot_map = {}
    for i in sorted(kobjs):
        obj = kobjs[i]
        if obj.get('uniprot_release'): releases.add(obj['uniprot_release'])
        if obj.get('uniprot_release_date'): release_dates.add(obj['uniprot_release_date'])
        for r in obj['records']:
            uniprot_map[r['accession']] = r
    print(f'UNIPROTKB_COMPLETE active_exact_accessions={len(uniprot_map):,}', flush=True)

    # Metadata join, then classification.
    meta_cols = [
        'enzyme_id','sequence_sha256','length','screening_sources','first_source_file',
        'first_locus_tag','source_file_count','source_files','organisms','locus_tags',
        'protein_ids','prokaryote_scope_evidence'
    ]
    meta = pd.read_csv(cdir / 'candidates_metadata.tsv.gz', sep='\t', dtype=str).fillna('')
    meta = meta[[c for c in meta_cols if c in meta.columns]]
    base = model[['enzyme_id','sequence_md5','sequence_sha256_check']].merge(meta, on='enzyme_id', how='left', validate='one_to_one')
    if 'sequence_sha256' in base.columns:
        bad = base[(base.sequence_sha256 != '') & (base.sequence_sha256 != base.sequence_sha256_check)]
        if len(bad): raise RuntimeError(f'sequence SHA mismatch for {len(bad)} candidates')

    rows = []
    for r in base.to_dict('records'):
        m = r['sequence_md5']; up = up_by_md5.get(m)
        z = dict(r)
        z['uniparc_exact_found'] = up is not None
        z['uniparc_id'] = '' if up is None else up['uniparc_id']
        z['uniparc_first_seen'] = '' if up is None else up['first_seen']
        z['uniparc_last_seen'] = '' if up is None else up['last_seen']
        norm_accs = [] if up is None else sorted(set(normalize_uniprot_acc(a) for a in up['uniprot_accessions'] if normalize_uniprot_acc(a)))
        z['uniprot_accessions'] = ';'.join(norm_accs)
        z['active_uniprot_accession_count'] = sum(a in uniprot_map for a in norm_accs)
        z.update(classify(m, up, uniprot_map))
        z['touch_protocol_id'] = PROTOCOL_ID
        z['touch_evidence_scope'] = 'exact-sequence UniParc + linked exact-sequence UniProtKB evidence; conservative positive-evidence tiering'
        z['touch_snapshot_utc'] = pd.Timestamp.utcnow().isoformat()
        rows.append(z)

    out = pd.DataFrame(rows)
    if len(out) != len(model) or out.enzyme_id.nunique() != len(model):
        raise RuntimeError('output coverage assertion failed')
    if not out.experimental_touch_level.isin(['T0','T1','T2','T3','T4','T5']).all():
        raise RuntimeError('non-T0..T5 tag found')

    counts = out.experimental_touch_level.value_counts().reindex(['T0','T1','T2','T3','T4','T5'], fill_value=0)
    out_tsv = cdir / f'{args.output_name}.tsv.gz'
    out.to_csv(out_tsv, sep='\t', index=False, compression='gzip')

    # Lightweight index for downstream strict T2+ workflow.
    ids = out.loc[out.touch_t2plus_eligible, 'enzyme_id'].astype(str).tolist()
    (cdir / f'{args.output_name}_T2plus_ids.txt').write_text('\n'.join(ids) + ('\n' if ids else ''), encoding='utf-8')
    counts.rename_axis('experimental_touch_level').rename('candidate_count').reset_index().to_csv(
        cdir / f'{args.output_name}_counts.csv', index=False
    )

    # Separate SQLite: no mutation of production DB or normalized proteome DB.
    db = cdir / f'{args.output_name}.sqlite'
    if db.exists(): db.unlink()
    con = sqlite3.connect(db)
    sqlcols = [
        'enzyme_id','sequence_sha256','sequence_md5','experimental_touch_level','experimental_touch_numeric',
        'touch_t2plus_eligible','experimental_touch_label','public_evidence_confidence',
        'uniparc_exact_found','uniparc_id','uniparc_first_seen','uniparc_last_seen','uniprot_accessions',
        'active_uniprot_accession_count','best_pe_level','best_pe_text','reviewed_exact','pdb_exact_ids',
        'experimental_functional_eco','experimental_catalytic_eco','kinetics_present',
        'mass_spectrometry_annotation_present','length','screening_sources','first_source_file',
        'first_locus_tag','source_files','organisms','locus_tags','prokaryote_scope_evidence',
        'touch_protocol_id','touch_evidence_scope','touch_snapshot_utc'
    ]
    sqlcols = [c for c in sqlcols if c in out.columns]
    out[sqlcols].to_sql('candidate_touch', con, index=False)
    con.execute('CREATE UNIQUE INDEX idx_touch_candidate ON candidate_touch(enzyme_id)')
    con.execute('CREATE INDEX idx_touch_level ON candidate_touch(experimental_touch_level)')
    con.execute('CREATE INDEX idx_touch_t2plus ON candidate_touch(touch_t2plus_eligible)')
    con.commit(); con.close()

    manifest = {
        'protocol_id': PROTOCOL_ID,
        'candidate_count': int(len(out)),
        'tag_coverage': 1.0,
        'touch_counts': {k:int(v) for k,v in counts.items()},
        't2plus_count': int(out.touch_t2plus_eligible.sum()),
        'uniparc_exact_found_count': int(out.uniparc_exact_found.sum()),
        'uniparc_exact_absent_count': int((~out.uniparc_exact_found).sum()),
        'unique_exact_uniprot_accessions_queried': int(len(accessions)),
        'active_exact_uniprot_accessions_returned': int(len(uniprot_map)),
        'uniprot_release': sorted(releases),
        'uniprot_release_date': sorted(release_dates),
        'definitions': {
            'T0': 'exact sequence absent from UniParc public archive snapshot',
            'T1': 'exact sequence public; no positive protein-level experimental evidence found in linked exact-sequence UniProtKB entries',
            'T2': 'positive protein-level experimental existence evidence',
            'T3': 'experimental PDB structure cross-reference for exact-sequence UniProt accession',
            'T4': 'published experimental function/catalysis annotation (ECO:0000269)',
            'T5': 'T4-level experimental function/catalysis plus experimental structure and/or kinetics',
        },
        'selection_contract': 'T2-T5 are safe as a strict positive-evidence gate under this evidence scope. T0/T1 are conservative negative calls and do not prove absence of unlinked or unpublished experiments.',
        'reaction_novelty_contract': 'Reaction-known-positive remains a separate reaction-specific relation and is never encoded by T0-T5.',
        'privacy': 'UniParc lookup sends MD5 checksum queries. Returned public sequences are used only to map query results back to MD5. Private full candidate sequences are not submitted as search payloads.',
        'production_registry_mutated': False,
        'outputs': {
            'tags_tsv_gz': str(out_tsv.relative_to(lib.parent.parent) if False else out_tsv),
            'sqlite': str(db),
            't2plus_ids': str(cdir / f'{args.output_name}_T2plus_ids.txt'),
        },
        'generated_utc': pd.Timestamp.utcnow().isoformat(),
    }
    (cdir / f'{args.output_name}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

if __name__ == '__main__':
    main()
