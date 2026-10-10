from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
FEATURE = CAGE / "feature"
for path in (CAGE, FEATURE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_original_torch_load = torch.load


def _mmap_large_extra_load(f, *args, **kwargs):
    path = Path(f) if isinstance(f, (str, Path)) else None
    if (
        path is not None
        and "bridge_layered_v4_r2e_cage/extra_features" in str(path)
        and path.name in {"gvp_protein_feature.pt", "esm_node_feature.pt"}
    ):
        kwargs.setdefault("mmap", True)
        print(f"MMAP_EXTRA_FEATURE {path}", flush=True)
    return _original_torch_load(f, *args, **kwargs)


torch.load = _mmap_large_extra_load

from config import Config
from infer import inference
from utils import check_files, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    model_conf = Config(args.config)
    seed = 42 if not hasattr(model_conf, "seed") else model_conf.seed
    seed_everything(seed)
    check_files(model_conf)
    inference(model_conf)


if __name__ == "__main__":
    main()
