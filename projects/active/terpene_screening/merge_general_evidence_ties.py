from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.blend_general_evidence_models import ASSET_FILES, checkpoint_names
from projects.active.terpene_screening.third_party.ties_merge import ties_merge


def _selected_keys(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> list[str]:
    return [
        key for key, value in state.items()
        if torch.is_floating_point(value) and key.startswith(prefixes)
    ]


def merge_state_dicts_ties(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    *,
    keep_fraction: float,
    scale: float,
    prefixes: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if not experts:
        raise ValueError("at least one expert is required")
    if scale < 0:
        raise ValueError("scale must be non-negative")
    if any(expert.keys() != base.keys() for expert in experts):
        raise ValueError("base and expert state dict keys differ")
    keys = _selected_keys(base, prefixes)
    if not keys:
        raise ValueError(f"no floating parameters matched prefixes {prefixes}")

    # TIES trims task vectors globally, not tensor-by-tensor. Keep exact flatten order.
    base_vector = torch.cat([base[key].reshape(-1).float() for key in keys])
    task_vectors = []
    for expert in experts:
        task_vectors.append(
            torch.cat([(expert[key].float() - base[key].float()).reshape(-1) for key in keys])
        )
        for key, value in base.items():
            if key in keys:
                continue
            if not torch.equal(expert[key], value):
                raise ValueError(f"expert drifted outside selected prefixes at {key}")
    stacked = torch.stack(task_vectors)
    merged_delta = ties_merge(stacked, keep_fraction=keep_fraction)
    merged_vector = base_vector + float(scale) * merged_delta

    result = {key: value.clone() for key, value in base.items()}
    offset = 0
    for key in keys:
        count = base[key].numel()
        result[key] = merged_vector[offset : offset + count].reshape_as(base[key]).to(base[key].dtype)
        offset += count
    if offset != merged_vector.numel():
        raise AssertionError("TIES unflatten offset mismatch")
    return result


def merge_directories(
    base_dir: Path,
    expert_dirs: list[Path],
    output_dir: Path,
    *,
    keep_fraction: float,
    scale: float,
    prefixes: tuple[str, ...],
) -> list[str]:
    base_dir = base_dir.resolve()
    expert_dirs = [path.resolve() for path in expert_dirs]
    output_dir = output_dir.resolve()
    signatures = [checkpoint_names(path) for path in [base_dir, *expert_dirs]]
    if not signatures[0] or any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(f"checkpoint sets differ: {signatures}")
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for name in signatures[0]:
        base_payload = torch.load(base_dir / "models" / name, map_location="cpu", weights_only=False)
        expert_payloads = [
            torch.load(path / "models" / name, map_location="cpu", weights_only=False)
            for path in expert_dirs
        ]
        payload = {key: value for key, value in base_payload.items() if key != "model_state_dict"}
        payload["model_state_dict"] = merge_state_dicts_ties(
            base_payload["model_state_dict"],
            [item["model_state_dict"] for item in expert_payloads],
            keep_fraction=keep_fraction,
            scale=scale,
            prefixes=prefixes,
        )
        payload["general_evidence_ties"] = {
            "base": str((base_dir / "models" / name).resolve()),
            "experts": [str((path / "models" / name).resolve()) for path in expert_dirs],
            "keep_fraction": float(keep_fraction),
            "scale": float(scale),
            "prefixes": list(prefixes),
            "upstream_repository": "https://github.com/prateeky2806/ties-merging",
            "upstream_commit": "44e7891fc84f3de7e4caa52664cd864ca3715e91",
        }
        target = output_dir / "models" / name
        torch.save(payload, target)
        outputs.append(str(target))
    for filename in ASSET_FILES:
        source = base_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
    summary = {
        "model_type": "general_evidence_ties_merge",
        "base_dir": str(base_dir),
        "expert_dirs": [str(path) for path in expert_dirs],
        "keep_fraction": float(keep_fraction),
        "scale": float(scale),
        "prefixes": list(prefixes),
        "upstream_method": "TIES-Merging (NeurIPS 2023)",
        "upstream_commit": "44e7891fc84f3de7e4caa52664cd864ca3715e91",
        "checkpoints": outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge same-base directional retrieval experts with TIES-Merging.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--expert-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-fraction", type=float, default=0.2)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--prefix", action="append", default=["reaction_tower."])
    args = parser.parse_args()
    outputs = merge_directories(
        args.base_dir,
        args.expert_dir,
        args.output_dir,
        keep_fraction=args.keep_fraction,
        scale=args.scale,
        prefixes=tuple(dict.fromkeys(args.prefix)),
    )
    print(json.dumps({"checkpoints": outputs}, indent=2))


if __name__ == "__main__":
    main()
