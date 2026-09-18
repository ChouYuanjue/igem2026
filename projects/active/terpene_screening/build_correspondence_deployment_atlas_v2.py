from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (
    best_match_similarity,
    reaction_features,
)
from projects.active.terpene_screening.prepare_marts_dataset import reaction_signature
from projects.active.terpene_screening.multiscale_geometry import _self_tuning_scale

CACHE = ROOT / 'data/terpene_marts_adaptation'
PG = ROOT / 'data/terpene_multiresolution_protein_geometry_v4'
RG = ROOT / 'data/terpene_multiresolution_reaction_geometry_v1'
GLOBAL = ROOT / 'data/terpene_global_esmc_aligned_v1'
GENERAL_RXN = ROOT / 'data/catalyst_candidate_universes/general_merged/reactions.csv'
CURRENT_TPS = ROOT / 'data/terpene/enzyme_terpene_synthase.tsv'
OUT = ROOT / 'data/terpene_correspondence_deployment_atlas_v2'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def unit_length_graph(path: Path) -> tuple[csr_matrix, float]:
    w = load_npz(path).astype(np.float64)
    w = w.maximum(w.T).tocsr()
    w.setdiag(0.0)
    w.eliminate_zeros()
    coo = w.tocoo()
    upper = coo.row < coo.col
    raw = np.sqrt(np.maximum(-np.log(np.clip(coo.data[upper], 1e-300, 1.0)), 1e-12))
    ell = float(np.median(raw))
    if not np.isfinite(ell) or ell <= 0:
        raise RuntimeError('invalid factor characteristic length')
    length = np.sqrt(np.maximum(-np.log(np.clip(w.data, 1e-300, 1.0)), 1e-12)) / ell
    graph = csr_matrix((length, w.indices, w.indptr), shape=w.shape)
    return graph, ell


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm[:, 0] <= 1e-12):
        raise RuntimeError('reference embedding contains a zero row')
    return (x / norm).astype(np.float32)


def symmetric_side_similarity(features: list[dict[str, object]], key: str) -> np.ndarray:
    n = len(features)
    sim = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if features[i][key]:
            sim[i, i] = 1.0
        for j in range(i):
            if not (features[i][key] and features[j][key]):
                continue
            a = float(best_match_similarity(features[i][key], features[j][key]))
            b = float(best_match_similarity(features[j][key], features[i][key]))
            sim[i, j] = sim[j, i] = 0.5 * (a + b)
    return sim


def diffusion_basis(similarity: np.ndarray, available: np.ndarray) -> dict[str, np.ndarray]:
    sim = np.asarray(similarity, dtype=np.float64).copy()
    available = np.asarray(available, dtype=bool)
    sim = np.where(np.outer(available, available), np.maximum(sim, 0.0), 0.0)
    np.fill_diagonal(sim, 0.0)
    degree = sim.sum(axis=1)
    relational = available & (degree > 1e-12)
    total = float(degree.sum())
    if total <= 1e-12:
        raise RuntimeError('empty reference diffusion graph')
    pi = np.maximum(degree / total, 1e-12)
    p = sim / np.maximum(degree[:, None], 1e-12)
    weighted = p / np.sqrt(pi[None, :])
    norm = np.sum(weighted * weighted, axis=1)
    gram = weighted @ weighted.T
    d2 = np.maximum(norm[:, None] + norm[None, :] - 2.0 * gram, 0.0)
    distance = np.sqrt(d2)
    distance[~(relational[:, None] & relational[None, :])] = np.inf
    np.fill_diagonal(distance, 0.0)
    return {
        'weighted_transition': weighted.astype(np.float32),
        'weighted_norm': norm.astype(np.float32),
        'stationary': pi.astype(np.float32),
        'available': relational,
        'distance': distance.astype(np.float32),
    }


def product_aliases(reactions: pd.DataFrame) -> pd.DataFrame:
    general = pd.read_csv(GENERAL_RXN, dtype=str).fillna('')
    general['reaction_signature'] = general.reaction_smiles.map(reaction_signature)
    aliases_by_sig = (
        general[general.reaction_signature.ne('')]
        .groupby('reaction_signature').reaction_id
        .apply(lambda values: sorted(set(values.astype(str))))
        .to_dict()
    )
    current = pd.read_csv(CURRENT_TPS, sep='\t', dtype=str).fillna('')
    current_ids = set(current.rhea_id.astype(str))
    rows = []
    for row in reactions.itertuples(index=False):
        aliases = list(aliases_by_sig.get(str(row.reaction_signature), []))
        rhea = sorted(value for value in aliases if value.startswith('RHEA:'))
        current_rhea = sorted(value for value in rhea if value in current_ids)
        if current_rhea:
            primary = current_rhea[0]
        elif rhea:
            primary = rhea[0]
        elif aliases:
            primary = aliases[0]
        else:
            primary = str(row.reaction_id)
        rows.append({
            'reaction_id': str(row.reaction_id),
            'primary_alias': primary,
            'aliases': ';'.join(aliases),
            'rhea_aliases': ';'.join(rhea),
            'reaction_signature': str(row.reaction_signature),
            'reaction_smiles': str(row.reaction_smiles),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_csv(CACHE / 'protein_entities.csv', dtype=str).fillna('')
    reactions = pd.read_csv(CACHE / 'reaction_entities.csv', dtype=str).fillna('')
    protein_order_table = pd.read_csv(PG / 'protein_ids.csv', dtype={'protein_id': str})
    protein_order_table['row'] = pd.to_numeric(protein_order_table['row']).astype(int)
    reaction_order_table = pd.read_csv(RG / 'reaction_ids.csv', dtype={'reaction_id': str})
    reaction_order_table['row'] = pd.to_numeric(reaction_order_table['row']).astype(int)
    if protein_order_table.sort_values('row').protein_id.astype(str).tolist() != proteins.protein_id.astype(str).tolist():
        raise RuntimeError('protein metadata order differs from canonical atlas row order')
    if reaction_order_table.sort_values('row').reaction_id.astype(str).tolist() != reactions.reaction_id.astype(str).tolist():
        raise RuntimeError('reaction metadata order differs from canonical atlas row order')

    pgraph, pell = unit_length_graph(PG / 'partial_pullback_affinity.npz')
    rgraph, rell = unit_length_graph(RG / 'partial_pullback_affinity.npz')
    save_npz(OUT / 'protein_unit_length_graph.npz', pgraph)
    save_npz(OUT / 'reaction_unit_length_graph.npz', rgraph)
    np.save(OUT / 'protein_geodesic.npy', np.asarray(shortest_path(pgraph, directed=False), dtype=np.float32))
    np.save(OUT / 'reaction_geodesic.npy', np.asarray(shortest_path(rgraph, directed=False), dtype=np.float32))

    # Exact ID order and product aliases.
    shutil.copy2(PG / 'protein_ids.csv', OUT / 'protein_ids.csv')
    shutil.copy2(RG / 'reaction_ids.csv', OUT / 'reaction_ids.csv')
    proteins[['protein_id', 'aliases', 'sequence']].to_csv(OUT / 'protein_entities.csv', index=False)
    product_aliases(reactions).to_csv(OUT / 'reaction_entities.csv', index=False)

    # Protein Standard query sufficient statistics.
    global_embeddings = np.load(GLOBAL / 'embeddings.npy', mmap_mode='r')
    pbase = normalized_rows(global_embeddings)
    np.save(OUT / 'protein_global_esmc_normalized.npy', pbase)
    p_dist = np.load(PG / 'global_esmc_chordal_distance.npy', mmap_mode='r')
    p_avail = np.load(PG / 'global_esmc_available.npy').astype(bool)
    np.save(OUT / 'protein_global_esmc_available.npy', p_avail)
    np.save(OUT / 'protein_global_esmc_scale.npy', _self_tuning_scale(np.asarray(p_dist, dtype=np.float64), p_avail, epsilon=1e-8).astype(np.float32))

    # Preserve all protein reference-view availability/scales for future Deep query
    # cross-distance executors without loading research N x N matrices online.
    p_specs = [
        ('pocket_local_esmc', 'pocket_local_esmc_chordal_distance.npy', 'pocket_local_esmc_available.npy'),
        ('whole_3di', 'whole_3di_diffusion_distance.npy', 'whole_3di_relational_available.npy'),
        ('pocket_3di', 'pocket_3di_diffusion_distance.npy', 'pocket_3di_relational_available.npy'),
        ('pocket_ot', 'pocket_ot_diffusion_distance.npy', 'pocket_ot_relational_available.npy'),
    ]
    for name, dist_name, avail_name in p_specs:
        d = np.load(PG / dist_name, mmap_mode='r')
        a = np.load(PG / avail_name).astype(bool)
        np.save(OUT / f'protein_{name}_available.npy', a)
        np.save(OUT / f'protein_{name}_scale.npy', _self_tuning_scale(np.asarray(d, dtype=np.float64), a, epsilon=1e-8).astype(np.float32))

    # Exact MARTS DRFP encoder target; query encoding must canonicalize first.
    drfp = np.asarray(np.load(CACHE / 'reaction_features.npy', mmap_mode='r')[:, :2048], dtype=np.float32)
    rnorm = np.linalg.norm(drfp, axis=1, keepdims=True)
    ravail = rnorm[:, 0] > 1e-12
    drfp_normalized = np.zeros_like(drfp, dtype=np.float32)
    drfp_normalized[ravail] = drfp[ravail] / rnorm[ravail]
    np.save(OUT / 'reaction_drfp_normalized.npy', drfp_normalized)
    r_drfp_dist = np.load(RG / 'drfp_chordal_distance.npy', mmap_mode='r')
    np.save(OUT / 'reaction_drfp_available.npy', ravail)
    np.save(OUT / 'reaction_drfp_scale.npy', _self_tuning_scale(np.asarray(r_drfp_dist, dtype=np.float64), ravail, epsilon=1e-8).astype(np.float32))

    # Frozen-reference one-step diffusion sufficient statistics for query->reference
    # reactant/product neighbourhood distances.
    chemistry = [reaction_features(value) for value in reactions.reaction_smiles.astype(str)]
    diffusion_checks = {}
    for name, key, canonical_file in [
        ('reactant', 'reactant_fps', 'reactant_diffusion_distance.npy'),
        ('product', 'product_fps', 'product_diffusion_distance.npy'),
    ]:
        raw = symmetric_side_similarity(chemistry, key)
        available = np.asarray([bool(value[key]) for value in chemistry], dtype=bool)
        basis = diffusion_basis(raw, available)
        canonical = np.load(RG / canonical_file).astype(np.float32)
        finite = np.isfinite(canonical) & np.isfinite(basis['distance'])
        max_abs = float(np.max(np.abs(canonical[finite] - basis['distance'][finite]))) if np.any(finite) else 0.0
        if max_abs > 2e-5 or not np.array_equal(np.isfinite(canonical), np.isfinite(basis['distance'])):
            raise RuntimeError(f'{name} diffusion basis does not reproduce canonical geometry: max_abs={max_abs}')
        diffusion_checks[name] = max_abs
        np.save(OUT / f'reaction_{name}_weighted_transition.npy', basis['weighted_transition'])
        np.save(OUT / f'reaction_{name}_weighted_norm.npy', basis['weighted_norm'])
        np.save(OUT / f'reaction_{name}_stationary.npy', basis['stationary'])
        np.save(OUT / f'reaction_{name}_available.npy', basis['available'])
        np.save(OUT / f'reaction_{name}_scale.npy', _self_tuning_scale(np.asarray(canonical, dtype=np.float64), basis['available'], epsilon=1e-8).astype(np.float32))

    manifest = {
        'version': 'terpene-correspondence-deployment-atlas-v2',
        'purpose': 'self-contained read-only production reference atlas for exact MARTS correspondence-field query extension',
        'reference_policy': 'reference-reference geometry is immutable; external queries attach out of sample and never rebuild the atlas synchronously',
        'factor_characteristic_lengths': {'protein': pell, 'reaction': rell},
        'standard_online_views': {
            'protein': ['global_esmc'],
            'reaction': ['drfp', 'reactant_neighbourhood', 'product_neighbourhood'],
        },
        'deep_online_policy': 'additional views may refine query attachment only after an executor actually materialises their query-to-reference distances; planned views are not treated as computed',
        'reaction_diffusion_reproduction_max_abs': diffusion_checks,
        'labels_used': False,
        'input_sha256': {
            'protein_entities': sha256(CACHE / 'protein_entities.csv'),
            'reaction_entities': sha256(CACHE / 'reaction_entities.csv'),
            'protein_geometry_manifest': sha256(PG / 'manifest.json'),
            'reaction_geometry_manifest': sha256(RG / 'manifest.json'),
            'global_esmc_manifest': sha256(GLOBAL / 'manifest.json'),
            'general_reaction_registry': sha256(GENERAL_RXN),
            'current_tps_pairs': sha256(CURRENT_TPS),
        },
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
