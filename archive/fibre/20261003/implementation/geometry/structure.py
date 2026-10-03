from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASIS = ROOT / 'data/terpene_correspondence_structural_basis_v1'
DEFAULT_FOLDSEEK = ROOT / 'tools/external/foldseek/bin/foldseek'


@dataclass(frozen=True)
class StructuralQueryResult:
    raw_similarity: np.ndarray
    diffusion_distance: np.ndarray
    observed: bool
    hit_count: int
    query_self_bits: float
    view: str


def _name(value: str) -> str:
    value = Path(str(value)).name
    for suffix in ('.cif', '.pdb'):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
    return value


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _query_self_bits(query_db: Path, foldseek: Path, work: Path) -> float:
    result = work / 'self_result'
    tmp = work / 'self_tmp'
    tsv = work / 'self.tsv'
    _run([
        str(foldseek), 'search', str(query_db), str(query_db), str(result), str(tmp),
        '--max-seqs', '1', '-e', '1000', '-s', '7.5', '--threads', '2', '-v', '1',
    ])
    _run([
        str(foldseek), 'convertalis', str(query_db), str(query_db), str(result), str(tsv),
        '--format-output', 'query,target,bits', '-v', '1',
    ])
    frame = pd.read_csv(tsv, sep='\t', header=None, names=['query','target','bits'])
    frame['bits'] = pd.to_numeric(frame.bits, errors='coerce').fillna(0.0)
    rows = frame[frame.apply(lambda row: _name(row['query']) == _name(row['target']), axis=1)]
    value = float(rows.bits.max()) if len(rows) else 0.0
    if value <= 0:
        raise RuntimeError('Foldseek query self score is missing')
    return value


def frozen_diffusion_cross(raw_affinity: np.ndarray, view_dir: Path) -> tuple[np.ndarray, bool]:
    affinity = np.asarray(raw_affinity, dtype=np.float64).reshape(-1)
    available = np.load(view_dir / 'available.npy').astype(bool)
    affinity = np.where(available, np.maximum(affinity, 0.0), 0.0)
    degree = float(affinity.sum())
    if degree <= 1e-12:
        return np.full(len(affinity), np.inf, dtype=np.float64), False
    transition = affinity / degree
    stationary = np.load(view_dir / 'stationary.npy').astype(np.float64)
    weighted_query = transition / np.sqrt(np.maximum(stationary, 1e-12))
    qnorm = float(weighted_query @ weighted_query)
    reference_weighted = np.load(view_dir / 'weighted_transition.npy', mmap_mode='r')
    reference_norm = np.load(view_dir / 'weighted_norm.npy').astype(np.float64)
    d2 = np.maximum(qnorm + reference_norm - 2.0 * (np.asarray(reference_weighted, dtype=np.float64) @ weighted_query), 0.0)
    distance = np.sqrt(d2)
    distance[~available] = np.inf
    return distance, True


def query_structural_view(
    structure_path: str | Path,
    *,
    view: str,
    basis_root: str | Path = DEFAULT_BASIS,
    foldseek_path: str | Path = DEFAULT_FOLDSEEK,
    max_seqs: int | None = None,
    work_root: str | Path | None = None,
) -> StructuralQueryResult:
    structure = Path(structure_path).resolve()
    if not structure.is_file() or structure.stat().st_size <= 0:
        raise FileNotFoundError(structure)
    basis = Path(basis_root).resolve()
    manifest = json.loads((basis / 'manifest.json').read_text())
    if view not in manifest['views']:
        raise ValueError(f'unsupported structural view: {view}')
    view_dir = basis / view
    ref_db = basis / manifest['views'][view]['foldseek_db_prefix']
    reference_index = pd.read_csv(view_dir / 'reference_index.csv', dtype=str).fillna('')
    reference_index['row'] = pd.to_numeric(reference_index.row, errors='raise').astype(int)
    n = int(manifest['protein_count'])
    row_by_accession = {str(row.uniprot): int(row.row) for row in reference_index.itertuples(index=False)}
    self_bits = np.load(view_dir / 'reference_self_bits.npy').astype(np.float64)
    contract = manifest.get('foldseek_search_contract') or {}
    max_hits = int(max_seqs or contract.get('max_seqs') or max(1, math.ceil(math.sqrt(n))))
    sensitivity = float(contract.get('sensitivity') or 7.5)
    evalue = float(contract.get('evalue') or 1000.0)
    foldseek = Path(foldseek_path).resolve()
    if not foldseek.is_file():
        raise FileNotFoundError(foldseek)

    base = Path(work_root).resolve() if work_root is not None else None
    if base is not None:
        base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='corr_struct_', dir=str(base) if base else None) as tmpdir:
        work = Path(tmpdir)
        query_input = work / 'query_input'
        query_input.mkdir()
        query_file = query_input / f'QUERY{structure.suffix.lower() if structure.suffix else ".cif"}'
        try:
            query_file.symlink_to(structure)
        except OSError:
            shutil.copy2(structure, query_file)
        query_db = work / 'query_db'
        result_db = work / 'result'
        search_tmp = work / 'search_tmp'
        tsv = work / 'hits.tsv'
        _run([str(foldseek), 'createdb', str(query_input), str(query_db), '--threads', '2', '-v', '1'])
        query_self = _query_self_bits(query_db, foldseek, work)
        _run([
            str(foldseek), 'search', str(query_db), str(ref_db), str(result_db), str(search_tmp),
            '--max-seqs', str(max_hits), '-e', str(evalue), '-s', str(sensitivity),
            '--threads', '4', '-v', '1',
        ])
        _run([
            str(foldseek), 'convertalis', str(query_db), str(ref_db), str(result_db), str(tsv),
            '--format-output', 'query,target,bits', '--threads', '2', '-v', '1',
        ])
        if tsv.is_file() and tsv.stat().st_size:
            hits = pd.read_csv(tsv, sep='\t', header=None, names=['query','target','bits'], dtype={'query':str,'target':str})
            hits['bits'] = pd.to_numeric(hits.bits, errors='coerce').fillna(0.0)
        else:
            hits = pd.DataFrame(columns=['query','target','bits'])

    raw = np.zeros(n, dtype=np.float64)
    for row in hits.itertuples(index=False):
        accession = _name(row.target)
        idx = row_by_accession.get(accession)
        if idx is None:
            continue
        denom = float(np.sqrt(query_self * self_bits[idx]))
        if denom <= 0:
            continue
        score = float(np.clip(float(row.bits) / denom, 0.0, 1.0))
        raw[idx] = max(raw[idx], score)
    distance, observed = frozen_diffusion_cross(raw, view_dir)
    return StructuralQueryResult(
        raw_similarity=raw,
        diffusion_distance=distance,
        observed=bool(observed),
        hit_count=int(np.count_nonzero(raw > 0)),
        query_self_bits=float(query_self),
        view=view,
    )
