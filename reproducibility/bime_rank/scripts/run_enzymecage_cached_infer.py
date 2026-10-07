from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from enzymecage.dataset.geometric import GeometricDataset

_ORIGINAL_GET_RXN_GRAPH_DATA = GeometricDataset.get_rxn_graph_data
_CACHE_STATS = {"hits": 0, "misses": 0}


def _cached_get_rxn_graph_data(self: GeometricDataset, rxn: str):
    cache = getattr(self, "_bridge_v4_rxn_graph_cache", None)
    if cache is None:
        cache = {}
        setattr(self, "_bridge_v4_rxn_graph_cache", cache)
    if rxn in cache:
        _CACHE_STATS["hits"] += 1
        return cache[rxn]
    value = _ORIGINAL_GET_RXN_GRAPH_DATA(self, rxn)
    cache[rxn] = value
    _CACHE_STATS["misses"] += 1
    return value


GeometricDataset.get_rxn_graph_data = _cached_get_rxn_graph_data

from config import Config
from infer import inference
from utils import check_files, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    model_conf = Config(str(config_path))
    seed = 42 if not hasattr(model_conf, "seed") else model_conf.seed
    seed_everything(seed)
    check_files(model_conf)
    try:
        inference(model_conf)
    finally:
        print(
            "BRIDGE_V4_RXN_GRAPH_CACHE",
            {
                **_CACHE_STATS,
                "unique_reactions_cached": _CACHE_STATS["misses"],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
