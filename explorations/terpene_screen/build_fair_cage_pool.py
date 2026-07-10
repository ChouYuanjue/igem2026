from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'terpene_cage_fair'
RESULT_DIR = PROJECT_ROOT / 'results' / 'terpene_cage_fair'
GATE_POOL_PATH = PROJECT_ROOT / 'data' / 'terpene_gate_matrix' / 'gate_candidate_pools.csv'
POSITIVE_PATH = PROJECT_ROOT / 'data' / 'terpene' / 'enzyme_terpene_synthase.tsv'
CANDIDATE_PATH = PROJECT_ROOT / 'data' / 'terpene' / 'all_seq_terpene_synthase.tsv'
OLD_CAGE_PATH = PROJECT_ROOT / 'results' / 'terpene_cage_screen' / 'all_rhea_gate' / 'all_pair_scores.csv'

MAIN_GATES = ['rxn_balanced_top20', 'rxn_balanced_top50', 'weighted_top100', 'recall_union_core']
FEWSHOT_SEEDS = [1, 2, 3, 5]
TOPN_PER_SEED = 100


def kmers(seq: str, k: int = 3) -> set[str]:
    seq = re.sub(r'[^A-Z]', '', str(seq).upper())
    if len(seq) < k:
        return set()
    return {seq[i:i+k] for i in range(len(seq) - k + 1)}


def build_seed_topn_pairs(positive: pd.DataFrame, candidate: pd.DataFrame) -> set[tuple[str, str]]:
    seq = dict(zip(candidate['Entry'].astype(str), candidate['Sequence'].astype(str)))
    candidate_ids = sorted(seq)
    candidate_set = set(candidate_ids)
    candidate_kmers = {uid: kmers(sequence) for uid, sequence in seq.items()}
    candidate_sizes = {uid: len(kset) for uid, kset in candidate_kmers.items()}
    inverted: dict[str, set[str]] = defaultdict(set)
    for uid, kset in candidate_kmers.items():
        for kmer in kset:
            inverted[kmer].add(uid)

    def topn_for_seed(seed_uid: str) -> list[str]:
        seed_kmers = candidate_kmers.get(seed_uid, set())
        if not seed_kmers:
            return []
        counts: dict[str, int] = defaultdict(int)
        for kmer in seed_kmers:
            for uid in inverted.get(kmer, set()):
                counts[uid] += 1
        seed_size = len(seed_kmers)
        rows: list[tuple[str, float]] = []
        for uid, intersection in counts.items():
            denom = seed_size + candidate_sizes.get(uid, 0) - intersection
            if denom > 0:
                rows.append((uid, intersection / denom))
        rows.sort(key=lambda item: (-item[1], item[0]))
        return [uid for uid, _ in rows[:TOPN_PER_SEED]]

    cache: dict[str, list[str]] = {}
    pairs: set[tuple[str, str]] = set()
    for m in FEWSHOT_SEEDS:
        for rhea_id, group in positive.groupby('rhea_id', sort=False):
            positives = sorted(set(group['Entry'].astype(str)) & candidate_set)
            if len(positives) < m + 1:
                continue
            for seed_uid in positives:
                if seed_uid not in cache:
                    cache[seed_uid] = topn_for_seed(seed_uid)
                for uid in cache[seed_uid]:
                    if uid != seed_uid:
                        pairs.add((rhea_id, uid))
    return pairs


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    gate = pd.read_csv(GATE_POOL_PATH, dtype=str).fillna('')
    positive = pd.read_csv(POSITIVE_PATH, sep='\t', dtype=str).fillna('')
    candidate = pd.read_csv(CANDIDATE_PATH, sep='\t', dtype=str).fillna('')

    main_pairs = set(map(tuple, gate[gate['gate_id'].isin(MAIN_GATES)][['reaction_id', 'uniprot_id']].drop_duplicates().values))
    fewshot_pairs = build_seed_topn_pairs(positive, candidate)
    all_pairs = main_pairs | fewshot_pairs

    reaction_smiles = positive.drop_duplicates('rhea_id').set_index('rhea_id')['smiles_seq'].astype(str).to_dict()
    sequence = candidate.drop_duplicates('Entry').set_index('Entry')['Sequence'].astype(str).to_dict()
    positive_pairs = set(map(tuple, positive[['rhea_id', 'Entry']].astype(str).values))

    rows = []
    missing_reaction_smiles = set()
    missing_sequence = set()
    for rhea_id, uid in sorted(all_pairs):
        rxn = reaction_smiles.get(rhea_id, '')
        seq = sequence.get(uid, '')
        if not rxn:
            missing_reaction_smiles.add(rhea_id)
            continue
        if not seq:
            missing_sequence.add(uid)
            continue
        label = 1 if (rhea_id, uid) in positive_pairs else 0
        sources = []
        if (rhea_id, uid) in main_pairs:
            sources.append('main_gate')
        if (rhea_id, uid) in fewshot_pairs:
            sources.append('fewshot_seed_neighbor')
        rows.append({
            'reaction_id': rhea_id,
            'rhea_id': rhea_id,
            'reaction_smiles': rxn,
            'CANO_RXN_SMILES': rxn,
            'enzyme_id': uid,
            'uniprot_id': uid,
            'UniprotID': uid,
            'sequence': seq,
            'Sequence': seq,
            'label': label,
            'Label': label,
            'fair_cage_sources': ';'.join(sources),
        })

    all_df = pd.DataFrame(rows)
    all_path = DATA_DIR / 'fair_cage_candidate_pairs.csv'
    all_df.to_csv(all_path, index=False)

    old_pairs: set[tuple[str, str]] = set()
    if OLD_CAGE_PATH.exists():
        old = pd.read_csv(OLD_CAGE_PATH, usecols=['reaction_id', 'uniprot_id'], dtype=str)
        old_pairs = set(map(tuple, old[['reaction_id', 'uniprot_id']].values))
    pair_keys = all_df[['reaction_id', 'uniprot_id']].apply(tuple, axis=1)
    missing_df = all_df[~pair_keys.isin(old_pairs)].copy()
    missing_path = DATA_DIR / 'fair_cage_candidate_pairs_missing_old_scores.csv'
    missing_df.to_csv(missing_path, index=False)

    summary = {
        'main_gates': MAIN_GATES,
        'topn_per_seed': TOPN_PER_SEED,
        'main_gate_pairs': len(main_pairs),
        'fewshot_seed_neighbor_pairs': len(fewshot_pairs),
        'combined_unique_pairs_before_omitting_missing_smiles': len(all_pairs),
        'pairs_written_for_cage': int(len(all_df)),
        'reactions_written_for_cage': int(all_df['reaction_id'].nunique()) if not all_df.empty else 0,
        'positive_pairs_written_for_cage': int(all_df['label'].sum()) if not all_df.empty else 0,
        'missing_reaction_smiles_reactions': sorted(missing_reaction_smiles),
        'missing_sequence_candidates': sorted(missing_sequence),
        'old_cage_pairs_reused': int(len(all_df) - len(missing_df)),
        'pairs_missing_old_cage_scores': int(len(missing_df)),
        'all_pairs_csv': str(all_path),
        'missing_old_scores_csv': str(missing_path),
    }
    (RESULT_DIR / 'fair_cage_pool_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
