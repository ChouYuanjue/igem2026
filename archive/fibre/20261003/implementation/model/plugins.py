from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .fibre import ExpertSpec, FibreRelationalModel


def attach_expert_plugin(
    model: FibreRelationalModel,
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
) -> dict:
    """Attach one independently trained expert adapter to a frozen FIBRE core."""
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    name = str(payload["name"])
    input_dim = int(payload["input_dim"])
    if name in model.expert_adapters:
        raise ValueError(f"expert already attached: {name}")
    model.register_expert(name, ExpertSpec(input_dim))
    model.expert_adapters[name].load_state_dict(payload["state_dict"])
    if device is None:
        device = next(model.parameters()).device
    model.expert_adapters[name].to(torch.device(device))
    return payload


def save_expert_plugin(
    path: str | Path,
    *,
    name: str,
    input_dim: int,
    model: FibreRelationalModel,
    metadata: Mapping[str, object] | None = None,
) -> None:
    if name not in model.expert_adapters:
        raise KeyError(name)
    torch.save(
        {
            "schema": "fibre-expert-plugin-v1",
            "name": str(name),
            "input_dim": int(input_dim),
            "state_dict": model.expert_adapters[name].state_dict(),
            "metadata": dict(metadata or {}),
        },
        Path(path),
    )
