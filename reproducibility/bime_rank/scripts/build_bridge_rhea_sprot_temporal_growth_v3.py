from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_ROOT = Path('/tmp/rhea_hist')
DEFAULT_OUT = ROOT / 'results/bridge_rhea_sprot_temporal_growth_v3'
RELEASES = (128, 134, 138, 142)
EXPECTED = {
    128: '06a88c1fb29a3170bc533e9557e8da7b278c6c0e2583159492c08aaea958abe0',
    134: 'b5e3927284ae0f6fed10a68c7709d75f8445043bf12d0dfa11cc2b71739c78ef',
    138: '25a3ed70b0c555835f10680b47a6538a6a4fa5123eb7a320b4f9107794ea7693',
    142: 'ff2e1038df2352d4ce4f3e9ae840c350211d8193e79b6a0a2d29fc1c2f392def',
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_alias_map(meta: pd.DataFrame) -> tuple[dict[str, str], int]:
    alias_to_ids: dict[str, set[str]] = {}
    for row in meta[['protein_id', 'canonical_accession', 'aliases']].fillna('').itertuples(index=False):
        values = {str(row.protein_id), str(row.canonical_accession)}
        values.update(x for x in re.split(r';+', str(row.aliases)) if x)
        for alias in values:
            if alias:
                alias_to_ids.setdefault(alias, set()).add(str(row.protein_id))
    ambiguous = sum(len(ids) > 1 for ids in alias_to_ids.values())
    return {a: next(iter(ids)) for a, ids in alias_to_ids.items() if len(ids) == 1}, ambiguous


def load_release(n: int, root: Path, alias: dict[str, str], reaction_ids: set[str]):
    path = root / str(n) / 'tsv/rhea2uniprot_sprot.tsv'
    digest = sha256(path)
    if digest != EXPECTED[n]:
        raise ValueError(f'Rhea release {n} mapping hash drift: {digest}')
    source = pd.read_csv(path, sep='\t', dtype=str).fillna('')
    frame = source[['RHEA_ID', 'ID']].copy()
    frame['protein_id'] = frame.ID.map(alias).fillna('')
    frame['reaction_id'] = 'RHEA:' + frame.RHEA_ID.astype(str)
    frame = frame[frame.protein_id.ne('') & frame.reaction_id.isin(reaction_ids)][['protein_id', 'reaction_id']].drop_duplicates()
    return frame.reset_index(drop=True), int(len(source)), digest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--release-root', type=Path, default=DEFAULT_RELEASE_ROOT)
    ap.add_argument('--output-root', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(ROOT / 'data/catalyst_candidate_universes/general_merged/protein_metadata.csv', dtype=str).fillna('')
    alias, ambiguous = build_alias_map(meta)
    reactions = pd.read_csv(ROOT / 'data/catalyst_candidate_universes/general_merged/reactions.csv', dtype=str).fillna('')
    reaction_ids = set(reactions.reaction_id.astype(str))
    clean = pd.read_csv(ROOT / 'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv', dtype=str).fillna('').drop_duplicates(['protein_id', 'reaction_id'])
    clean_set = set(map(tuple, clean[['protein_id', 'reaction_id']].to_numpy()))

    frames = {}; graphs = {}; audits = {}
    for n in RELEASES:
        frame, source_rows, digest = load_release(n, args.release_root.resolve(), alias, reaction_ids)
        frames[n] = frame
        graphs[n] = set(map(tuple, frame[['protein_id', 'reaction_id']].to_numpy()))
        audits[n] = {'source_rows': source_rows, 'mapped_pairs': int(len(frame)), 'proteins': int(frame.protein_id.nunique()), 'reactions': int(frame.reaction_id.nunique()), 'sha256': digest}

    p128 = set(frames[128].protein_id); r128 = set(frames[128].reaction_id)
    rows = []
    for protein_id, reaction_id in sorted(graphs[142] - graphs[128]):
        if (protein_id, reaction_id) in clean_set:
            continue
        p_seen = protein_id in p128; r_seen = reaction_id in r128
        n_old = int(p_seen) + int(r_seen)
        growth = {2: 'edge_completion', 1: 'single_endpoint_expansion', 0: 'double_endpoint_expansion'}[n_old]
        orientation = 'both_old' if n_old == 2 else ('new_protein' if (not p_seen and r_seen) else ('new_reaction' if (p_seen and not r_seen) else 'both_new'))
        first = 134 if (protein_id, reaction_id) in graphs[134] else (138 if (protein_id, reaction_id) in graphs[138] else 142)
        rows.append((protein_id, reaction_id, growth, orientation, first, p_seen, r_seen))
    target = pd.DataFrame(rows, columns=['protein_id','reaction_id','growth_class','orientation','first_observed_release','protein_present_r128','reaction_present_r128'])
    target.to_csv(out / 'targets.csv', index=False)

    summary = {
        'schema': 'bridge-rhea-sprot-temporal-growth-v3',
        'source': 'official Rhea historical rhea2uniprot_sprot.tsv mappings',
        'official_archive_pattern': 'https://ftp.expasy.org/databases/rhea/old_releases/{release}.tar.bz2',
        'baseline_release': {'number': 128, 'release_date': '2023-06-28'},
        'anchors': {'134': '2024-07-24', '138': '2025-04-09'},
        'final_release': {'number': 142, 'release_date': '2026-09-02'},
        'mapping': {'unambiguous_protein_alias_keys': int(len(alias)), 'ambiguous_alias_keys': int(ambiguous), 'reaction_id_semantics': 'direction-specific RHEA_ID', 'clean2023_pair_overlap_excluded': True},
        'releases': {str(n): audits[n] for n in RELEASES},
        'targets': {
            'edges': int(len(target)), 'r2e_queries': int(target.reaction_id.nunique()), 'e2r_queries': int(target.protein_id.nunique()),
            'by_growth_class': {k: int(v) for k, v in target.growth_class.value_counts().items()},
            'by_orientation': {k: int(v) for k, v in target.orientation.value_counts().items()},
            'by_first_release': {str(k): int(v) for k, v in target.first_observed_release.value_counts().sort_index().items()},
            'headline': {'graph_completion': int(target.growth_class.eq('edge_completion').sum()), 'graph_expansion': int(target.growth_class.ne('edge_completion').sum())},
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
