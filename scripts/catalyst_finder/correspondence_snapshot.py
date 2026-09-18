from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ATLAS=ROOT/'data/terpene_correspondence_deployment_atlas_v2'
PAIR_REGISTRY=ROOT/'data/terpene_marts_adaptation/marts_pair_folds.csv'


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


class CorrespondenceSnapshot:
    """Version identifiers for immutable geometry and mutable positive evidence."""

    def __init__(self) -> None:
        manifest=json.loads((ATLAS/'manifest.json').read_text())
        self.atlas_version=str(manifest['version'])
        self.atlas_manifest_sha256=sha256(ATLAS/'manifest.json')
        # Current production snapshot uses every verified pair in the canonical
        # local positive registry. Research evaluators continue to construct
        # fold-specific Omega_train and never read this production identifier.
        self.production_positive_registry=str(PAIR_REGISTRY.relative_to(ROOT))
        self.production_omega_sha256=sha256(PAIR_REGISTRY)

    def as_dict(self) -> dict[str,str]:
        return {
            'atlas_version':self.atlas_version,
            'atlas_manifest_sha256':self.atlas_manifest_sha256,
            'omega_policy':'production_all_verified_positives',
            'omega_registry':self.production_positive_registry,
            'omega_sha256':self.production_omega_sha256,
        }
