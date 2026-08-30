from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from projects.active.terpene_screening.blend_general_evidence_models import ASSET_FILES, checkpoint_names


def compose_state_dicts(
    base: dict[str, torch.Tensor],
    protein_source: dict[str, torch.Tensor],
    reaction_source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if base.keys() != protein_source.keys() or base.keys() != reaction_source.keys():
        raise ValueError("state dict keys differ")
    result: dict[str, torch.Tensor] = {}
    for key, value in base.items():
        if key.startswith("protein_tower."):
            source = protein_source[key]
        elif key.startswith("reaction_tower."):
            source = reaction_source[key]
        else:
            source = value
            if not torch.equal(protein_source[key], value) or not torch.equal(reaction_source[key], value):
                raise ValueError(f"non-tower state drifted for {key}")
        if source.shape != value.shape or source.dtype != value.dtype:
            raise ValueError(f"tensor contract differs for {key}")
        result[key] = source.clone()
    return result


def compose_directories(
    base_dir: Path,
    protein_dir: Path,
    reaction_dir: Path,
    output_dir: Path,
) -> list[str]:
    base_dir = base_dir.resolve()
    protein_dir = protein_dir.resolve()
    reaction_dir = reaction_dir.resolve()
    output_dir = output_dir.resolve()
    signatures = [checkpoint_names(path) for path in (base_dir, protein_dir, reaction_dir)]
    if not signatures[0] or not (signatures[0] == signatures[1] == signatures[2]):
        raise ValueError(
            "checkpoint sets differ: "
            f"base={signatures[0]}, protein={signatures[1]}, reaction={signatures[2]}"
        )

    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for name in signatures[0]:
        base_payload = torch.load(base_dir / "models" / name, map_location="cpu", weights_only=False)
        protein_payload = torch.load(protein_dir / "models" / name, map_location="cpu", weights_only=False)
        reaction_payload = torch.load(reaction_dir / "models" / name, map_location="cpu", weights_only=False)
        payload = {key: value for key, value in base_payload.items() if key != "model_state_dict"}
        payload["model_state_dict"] = compose_state_dicts(
            base_payload["model_state_dict"],
            protein_payload["model_state_dict"],
            reaction_payload["model_state_dict"],
        )
        payload["general_evidence_composition"] = {
            "base": str((base_dir / "models" / name).resolve()),
            "protein_tower": str((protein_dir / "models" / name).resolve()),
            "reaction_tower": str((reaction_dir / "models" / name).resolve()),
        }
        target = output_dir / "models" / name
        torch.save(payload, target)
        outputs.append(str(target))

    for filename in ASSET_FILES:
        source = base_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    summary = {
        "model_type": "general_evidence_composed_directional_towers",
        "base_dir": str(base_dir),
        "protein_tower_dir": str(protein_dir),
        "reaction_tower_dir": str(reaction_dir),
        "checkpoints": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose independently adapted protein and reaction towers by matching seed.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--protein-dir", type=Path, required=True)
    parser.add_argument("--reaction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = compose_directories(args.base_dir, args.protein_dir, args.reaction_dir, args.output_dir)
    print(json.dumps({"checkpoints": outputs}, indent=2))


if __name__ == "__main__":
    main()
