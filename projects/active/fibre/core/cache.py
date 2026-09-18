from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_FEATURE_CACHE = ROOT / "data/terpene_feature_cache"

def stable_digest(namespace: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(namespace.encode() + b"\0" + encoded).hexdigest()

@dataclass(frozen=True)
class FeatureCache:
    root: Path = DEFAULT_FEATURE_CACHE
    def path(self, namespace: str, digest: str) -> Path:
        return self.root / namespace / digest[:2] / f"{digest}.npy"
    def get(self, namespace: str, digest: str) -> np.ndarray | None:
        path = self.path(namespace, digest)
        if not path.exists(): return None
        return np.load(path).astype(np.float32)
    def put(self, namespace: str, digest: str, values: np.ndarray) -> Path:
        path = self.path(namespace, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", suffix=".npy", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            np.save(handle, np.asarray(values, dtype=np.float32))
        os.replace(temporary, path)
        return path
