"""Resolve reviewed claim IDs; fail closed on stale or superseded primaries."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def resolve(claim: str, root: Path = ROOT) -> Path:
    index = json.loads((root/'reproducibility/bime_rank/canonical.json').read_text())
    entry = index['claims'][claim]
    rel = entry['primary']
    if any(rel == old or rel.startswith(old + '/') for old in index['superseded']):
        raise ValueError(f'Superseded asset cannot be canonical: {rel}')
    path = root / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != entry['sha256']:
        raise ValueError(f'Canonical asset changed: {rel}; review selection and regenerate index')
    return path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('claim',nargs='?')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:
        index=json.loads((ROOT/'reproducibility/bime_rank/canonical.json').read_text())
        for claim in index['claims']:resolve(claim)
        print(f"Verified {len(index['claims'])} canonical primaries")
    elif args.claim:print(resolve(args.claim))
    else:parser.error('provide a claim ID or --verify')
if __name__=='__main__':main()
