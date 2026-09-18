from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.geometry.multiscale import _self_tuning_scale

PROTEINS = ROOT / 'data/terpene_marts_adaptation/protein_entities.csv'
P2RANK = ROOT / 'results/terpene_p2rank_current_v1/p2rank_pocket_manifest.csv'
STRUCTURAL = ROOT / 'data/terpene_structural_observations_v1'
PROTEIN_GEOM = ROOT / 'data/terpene_multiresolution_protein_geometry_v4'
FOLDSEEK = ROOT / 'tools/external/foldseek/bin/foldseek'
OUT = ROOT / 'data/terpene_correspondence_structural_basis_v1'
THREADS = 24


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def diffusion_basis(similarity: np.ndarray, available: np.ndarray) -> dict[str, np.ndarray]:
    sim = np.asarray(similarity, dtype=np.float64).copy()
    available = np.asarray(available, dtype=bool)
    sim = np.where(np.outer(available, available), np.maximum(sim, 0.0), 0.0)
    np.fill_diagonal(sim, 0.0)
    degree = sim.sum(axis=1)
    relational = available & (degree > 1e-12)
    total = float(degree.sum())
    if total <= 1e-12:
        raise RuntimeError('empty structural reference graph')
    pi = np.maximum(degree / total, 1e-12)
    transition = sim / np.maximum(degree[:, None], 1e-12)
    weighted = transition / np.sqrt(pi[None, :])
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
        'available': relational.astype(bool),
        'distance': distance.astype(np.float32),
    }


def clean_prefix(parent: Path, prefix: str) -> None:
    for path in parent.glob(prefix + '*'):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def key(value: str) -> str:
    name = Path(str(value)).name
    for suffix in ('.cif', '.pdb'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    return name


def build_view(
    *,
    name: str,
    frame: pd.DataFrame,
    path_column: str,
    raw_similarity: Path,
    raw_available: Path,
    canonical_distance: Path,
    canonical_available: Path,
) -> dict[str, object]:
    out = OUT / name
    inp = out / 'reference_input'
    fs = out / 'foldseek'
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(inp, ignore_errors=True)
    inp.mkdir(parents=True, exist_ok=True)
    fs.mkdir(parents=True, exist_ok=True)

    proteins = pd.read_csv(PROTEINS, dtype=str).fillna('')
    protein_to_row = {str(pid): i for i, pid in enumerate(proteins.protein_id.astype(str))}
    rows: list[dict[str, object]] = []
    accession_to_row: dict[str, int] = {}
    for item in frame.itertuples(index=False):
        accession = str(item.UniprotID)
        protein_id = str(getattr(item, 'protein_id'))
        row = protein_to_row.get(protein_id)
        source = Path(str(getattr(item, path_column)))
        if row is None or not accession or not source.is_file() or source.stat().st_size <= 0:
            continue
        suffix = '.pdb' if source.suffix.lower() == '.pdb' else '.cif'
        target = inp / f'{accession}{suffix}'
        target.symlink_to(source.resolve())
        accession_to_row[accession] = int(row)
        rows.append({
            'row': int(row), 'protein_id': protein_id, 'uniprot': accession,
            'source_path': str(source.resolve()), 'source_sha256': sha256(source),
        })
    index = pd.DataFrame(rows).sort_values('row', kind='stable').reset_index(drop=True)
    index.to_csv(out / 'reference_index.csv', index=False)

    expected_available = np.load(raw_available).astype(bool)
    actual_rows = set(index.row.astype(int))
    expected_rows = set(np.flatnonzero(expected_available).tolist())
    if actual_rows != expected_rows:
        missing = sorted(expected_rows - actual_rows)[:10]
        extra = sorted(actual_rows - expected_rows)[:10]
        raise RuntimeError(f'{name}: reference structure set mismatch missing={missing} extra={extra}')

    db = fs / 'reference_db'
    result = fs / 'self_result'
    tmp = fs / 'tmp'
    self_tsv = fs / 'self.tsv'
    clean_prefix(fs, 'reference_db')
    clean_prefix(fs, 'self_result')
    shutil.rmtree(tmp, ignore_errors=True)
    self_tsv.unlink(missing_ok=True)

    run([str(FOLDSEEK), 'createdb', str(inp), str(db), '--threads', str(THREADS), '-v', '2'])
    run([
        str(FOLDSEEK), 'search', str(db), str(db), str(result), str(tmp),
        '--max-seqs', '1', '-e', '1000', '-s', '7.5',
        '--threads', str(THREADS), '-v', '2',
    ])
    run([
        str(FOLDSEEK), 'convertalis', str(db), str(db), str(result), str(self_tsv),
        '--format-output', 'query,target,bits', '--threads', str(THREADS), '-v', '2',
    ])
    self_frame = pd.read_csv(self_tsv, sep='\t', header=None, names=['query','target','bits'], dtype={'query':str,'target':str})
    self_frame['bits'] = pd.to_numeric(self_frame.bits, errors='coerce').fillna(0.0)
    self_bits = np.zeros(len(proteins), dtype=np.float32)
    for row in self_frame.itertuples(index=False):
        q = key(row.query); t = key(row.target)
        if q != t or q not in accession_to_row:
            continue
        idx = accession_to_row[q]
        self_bits[idx] = max(float(self_bits[idx]), float(row.bits))
    if np.any(self_bits[expected_available] <= 0):
        bad = np.flatnonzero(expected_available & (self_bits <= 0))[:10].tolist()
        raise RuntimeError(f'{name}: missing Foldseek self bits at rows {bad}')
    np.save(out / 'reference_self_bits.npy', self_bits)

    raw = np.load(raw_similarity, mmap_mode='r')
    basis = diffusion_basis(np.asarray(raw, dtype=np.float64), expected_available)
    canonical = np.load(canonical_distance, mmap_mode='r')
    can_avail = np.load(canonical_available).astype(bool)
    if not np.array_equal(basis['available'], can_avail):
        raise RuntimeError(f'{name}: relational availability differs from canonical geometry')
    finite = np.isfinite(canonical) & np.isfinite(basis['distance'])
    max_abs = float(np.max(np.abs(np.asarray(canonical)[finite] - basis['distance'][finite]))) if np.any(finite) else 0.0
    if max_abs > 2e-5 or not np.array_equal(np.isfinite(canonical), np.isfinite(basis['distance'])):
        raise RuntimeError(f'{name}: frozen diffusion basis does not reproduce canonical distance, max_abs={max_abs}')
    np.save(out / 'weighted_transition.npy', basis['weighted_transition'])
    np.save(out / 'weighted_norm.npy', basis['weighted_norm'])
    np.save(out / 'stationary.npy', basis['stationary'])
    np.save(out / 'available.npy', basis['available'])
    scale = _self_tuning_scale(np.asarray(canonical, dtype=np.float64), can_avail, epsilon=1e-8)
    np.save(out / 'scale.npy', scale.astype(np.float32))

    # Search results are not needed after extracting reference self scores.
    clean_prefix(fs, 'self_result')
    shutil.rmtree(tmp, ignore_errors=True)

    return {
        'view': name,
        'reference_count': int(expected_available.sum()),
        'foldseek_db_prefix': str(db.relative_to(OUT)),
        'reference_index': str((out / 'reference_index.csv').relative_to(OUT)),
        'self_bits': str((out / 'reference_self_bits.npy').relative_to(OUT)),
        'diffusion_reproduction_max_abs': max_abs,
        'raw_similarity_sha256': sha256(raw_similarity),
        'canonical_distance_sha256': sha256(canonical_distance),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proteins = pd.read_csv(PROTEINS, dtype=str).fillna('')
    p2 = pd.read_csv(P2RANK, dtype=str).fillna('')
    if p2.UniprotID.duplicated().any():
        raise RuntimeError('P2Rank manifest has duplicate UniProt IDs')
    # Recover canonical protein IDs from the current P2Rank candidate mapping.
    candidates = pd.read_csv(ROOT / 'data/terpene_p2rank_current_v1/candidates.csv', dtype=str).fillna('')
    pid = candidates[['UniprotID','protein_id']].drop_duplicates()
    p2 = p2.merge(pid, on='UniprotID', how='left', validate='one_to_one')
    if p2.protein_id.eq('').any() or p2.protein_id.isna().any():
        raise RuntimeError('failed to map P2Rank structures to canonical protein IDs')

    whole = build_view(
        name='whole_3di', frame=p2, path_column='structure_path',
        raw_similarity=STRUCTURAL/'whole_3di/similarity.npy',
        raw_available=STRUCTURAL/'whole_3di/available.npy',
        canonical_distance=PROTEIN_GEOM/'whole_3di_diffusion_distance.npy',
        canonical_available=PROTEIN_GEOM/'whole_3di_relational_available.npy',
    )
    pocket = build_view(
        name='pocket_3di', frame=p2, path_column='pocket_pdb_path',
        raw_similarity=STRUCTURAL/'pocket_3di/similarity.npy',
        raw_available=STRUCTURAL/'pocket_3di/available.npy',
        canonical_distance=PROTEIN_GEOM/'pocket_3di_diffusion_distance.npy',
        canonical_available=PROTEIN_GEOM/'pocket_3di_relational_available.npy',
    )
    manifest = {
        'version': 'terpene-correspondence-structural-basis-v1',
        'purpose': 'immutable Foldseek reference basis for single-query whole-structure and pocket-3Di extension of the canonical protein manifold',
        'foldseek_version': subprocess.check_output([str(FOLDSEEK),'version'], text=True).strip(),
        'foldseek_search_contract': {'max_seqs':64,'evalue':1000,'sensitivity':7.5},
        'protein_count': int(len(proteins)),
        'views': {'whole_3di': whole, 'pocket_3di': pocket},
        'input_sha256': {
            'protein_entities': sha256(PROTEINS),
            'p2rank_manifest': sha256(P2RANK),
        },
        'missing_is_negative_evidence': False,
    }
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
