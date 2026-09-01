from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.enzgfm_native_same_support_v1_common import (
    DualTowerNative,
    bag_feature,
    build_label_matrix,
    normalize_reaction_bag,
    normalize_sequence,
    query_metrics,
    seed_all,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / 'projects/active/terpene_screening/TIGER_REACTZYME_REACTION_SIMILARITY_BASELINE_CONTRACT_V1.json'
RESULT_ROOT = ROOT / 'results/tiger_reactzyme_reaction_similarity_native_v1'
PREPARED = RESULT_ROOT / 'prepared'
BASE_FEATURE_ROOT = ROOT / 'data/external/enzgfm_current/general_merged_650m_mean_v1'
PROTEIN_SEQUENCE_TSV = ROOT / 'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
TRAIN_POS = ROOT / 'data/external/reactzyme/reaction_smi_split/positive_train_val_mol_smi.pt'
TEST_POS = ROOT / 'data/external/reactzyme/reaction_smi_split/positive_test_mol_smi.pt'
OVERLAY_FEATURE_ROOT = RESULT_ROOT / 'train_only_missing_enzgfm_overlay'
SEED = 20260901
EXPECTED_ARCHIVE_MD5 = '2d9f4e6c78d8daf5752cc2a5ae2bef0d'

TIGER = {
    'e2r': {'hit_at_1': .4155, 'hit_at_5': .6416, 'hit_at_10': .6827, 'hit_at_20': .7540, 'author_avg_positive_rr': .5180},
    'r2e': {'hit_at_1': .4305, 'hit_at_5': .6113, 'hit_at_10': .6994, 'hit_at_20': .7616, 'author_avg_positive_rr': .3185},
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity comparison with the official Zenodo MD5, not security use.
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class CombinedProteinFeatures:
    """Index the fixed 185,918-row EnzGFM library plus a train-only overlay without copying the base matrix."""

    def __init__(self, base: np.ndarray, overlay: np.ndarray | None = None):
        self.base = base
        self.overlay = overlay
        self.base_count = len(base)

    def __getitem__(self, index):
        idx = np.asarray(index, dtype=np.int64)
        scalar = idx.ndim == 0
        flat = idx.reshape(-1)
        out = np.empty((len(flat), self.base.shape[1]), dtype=np.float32)
        base_mask = flat < self.base_count
        if base_mask.any():
            out[base_mask] = np.asarray(self.base[flat[base_mask]], dtype=np.float32)
        if (~base_mask).any():
            if self.overlay is None:
                raise IndexError('overlay protein feature requested but overlay is unavailable')
            oi = flat[~base_mask] - self.base_count
            if (oi < 0).any() or (oi >= len(self.overlay)).any():
                raise IndexError('overlay protein index outside frozen overlay')
            out[~base_mask] = np.asarray(self.overlay[oi], dtype=np.float32)
        return out[0] if scalar else out.reshape((*idx.shape, self.base.shape[1]))


def load_combined_features(require_overlay: bool = True) -> CombinedProteinFeatures:
    base = np.load(BASE_FEATURE_ROOT / 'embeddings.npy', mmap_mode='r')
    overlay_path = OVERLAY_FEATURE_ROOT / 'embeddings.npy'
    overlay = np.load(overlay_path, mmap_mode='r') if overlay_path.exists() else None
    if require_overlay and overlay is None:
        raise FileNotFoundError(f'missing frozen train-only overlay: {overlay_path}')
    return CombinedProteinFeatures(base, overlay)


def paper_common_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = ('hit_at_1', 'hit_at_5', 'hit_at_10', 'hit_at_20', 'author_avg_positive_rr')
    return {k: float(metrics[k]) for k in keys}
