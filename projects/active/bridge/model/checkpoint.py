from __future__ import annotations

from pathlib import Path

import torch

from .fibre import ExpertSpec, FibreRelationalModel


def load_fibre_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> tuple[FibreRelationalModel, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    experts = {
        str(name): ExpertSpec(int(dim))
        for name, dim in dict(payload.get("experts") or {}).items()
    }
    model = FibreRelationalModel(
        protein_input_dim=1152,
        molecule_input_dim=512,
        experts=experts,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(torch.device(device))
    if eval_mode:
        model.eval()
    return model, payload
