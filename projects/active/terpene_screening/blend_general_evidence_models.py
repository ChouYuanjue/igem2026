from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ASSET_FILES = (
    "feature_schema.json",
    "reaction_feature_matrix.npy",
    "reaction_features.csv",
    "protein_registry.csv",
    "reaction_registry.csv",
    "training_pairs.csv",
)


def blend_state_dicts(
    base: dict[str, torch.Tensor],
    adapted: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if base.keys() != adapted.keys():
        raise ValueError("base and adapted state dict keys differ")
    result: dict[str, torch.Tensor] = {}
    for key, before in base.items():
        after = adapted[key]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"tensor contract differs for {key}")
        if torch.is_floating_point(before):
            result[key] = before.lerp(after, float(alpha))
        else:
            if not torch.equal(before, after):
                raise ValueError(f"non-floating state differs for {key}")
            result[key] = before.clone()
    return result


def checkpoint_names(model_dir: Path) -> tuple[str, ...]:
    return tuple(sorted(path.name for path in (model_dir / "models").glob("production_seed*.pt")))


def blend_directories(base_dir: Path, adapted_dir: Path, output_dir: Path, alpha: float) -> list[str]:
    base_dir = base_dir.resolve()
    adapted_dir = adapted_dir.resolve()
    output_dir = output_dir.resolve()
    base_names = checkpoint_names(base_dir)
    adapted_names = checkpoint_names(adapted_dir)
    if not base_names or base_names != adapted_names:
        raise ValueError(f"checkpoint sets differ: base={base_names}, adapted={adapted_names}")

    model_out = output_dir / "models"
    model_out.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for name in base_names:
        base_payload = torch.load(base_dir / "models" / name, map_location="cpu", weights_only=False)
        adapted_payload = torch.load(adapted_dir / "models" / name, map_location="cpu", weights_only=False)
        state = blend_state_dicts(base_payload["model_state_dict"], adapted_payload["model_state_dict"], alpha)
        payload = {key: value for key, value in adapted_payload.items() if key != "model_state_dict"}
        payload["model_state_dict"] = state
        payload["general_evidence_blend_alpha"] = float(alpha)
        payload["general_evidence_blend_base"] = str((base_dir / "models" / name).resolve())
        payload["general_evidence_blend_adapted"] = str((adapted_dir / "models" / name).resolve())
        target = model_out / name
        torch.save(payload, target)
        outputs.append(str(target))

    for filename in ASSET_FILES:
        source = base_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    summary = {
        "model_type": "general_evidence_weight_blend",
        "alpha": float(alpha),
        "base_dir": str(base_dir),
        "adapted_dir": str(adapted_dir),
        "checkpoints": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate a continuation checkpoint back toward its frozen production base.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True, help="0 = base model, 1 = fully adapted model")
    args = parser.parse_args()
    outputs = blend_directories(args.base_dir, args.adapted_dir, args.output_dir, args.alpha)
    print(json.dumps({"alpha": args.alpha, "checkpoints": outputs}, indent=2))


if __name__ == "__main__":
    main()
