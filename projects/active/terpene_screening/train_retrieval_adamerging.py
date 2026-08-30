from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.blend_general_evidence_models import ASSET_FILES, checkpoint_names
from projects.active.terpene_screening.rank_open_world import (
    load_feature_schema,
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig, TerpeneDualTower, seed_everything

UPSTREAM_REPOSITORY = "https://github.com/EnnengYang/AdaMerging"
UPSTREAM_PAPER = "AdaMerging: Adaptive Model Merging for Multi-Task Learning (ICLR 2024)"


def _tower_parameter_names(state: Mapping[str, torch.Tensor], prefix: str) -> tuple[str, ...]:
    return tuple(
        key for key, value in state.items()
        if key.startswith(prefix) and torch.is_floating_point(value)
    )


def _validate_directional_expert(
    base: Mapping[str, torch.Tensor],
    expert: Mapping[str, torch.Tensor],
    *,
    allowed_prefix: str,
) -> None:
    if base.keys() != expert.keys():
        raise ValueError("base and expert state dict keys differ")
    for key, value in base.items():
        if key.startswith(allowed_prefix):
            continue
        if not torch.equal(value, expert[key]):
            raise ValueError(f"directional expert drifted outside {allowed_prefix}: {key}")


def materialize_layerwise_merge(
    base: Mapping[str, torch.Tensor],
    protein_expert: Mapping[str, torch.Tensor],
    reaction_expert: Mapping[str, torch.Tensor],
    alphas: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    _validate_directional_expert(base, protein_expert, allowed_prefix="protein_tower.")
    _validate_directional_expert(base, reaction_expert, allowed_prefix="reaction_tower.")
    result: dict[str, torch.Tensor] = {}
    for key, value in base.items():
        if key.startswith("protein_tower.") and torch.is_floating_point(value):
            alpha = float(alphas[key])
            result[key] = value + alpha * (protein_expert[key] - value)
        elif key.startswith("reaction_tower.") and torch.is_floating_point(value):
            alpha = float(alphas[key])
            result[key] = value + alpha * (reaction_expert[key] - value)
        else:
            result[key] = value.clone()
    return result


def multipositive_bidirectional_loss(scores: torch.Tensor, positive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be aligned matrices")
    if positive_mask.dtype != torch.bool:
        raise ValueError("positive_mask must be boolean")
    row_keep = positive_mask.any(dim=1)
    col_keep = positive_mask.any(dim=0)
    if not row_keep.any() or not col_keep.any():
        raise ValueError("positive_mask must cover at least one row and one column")

    row_scores = scores[row_keep]
    row_positive = positive_mask[row_keep]
    row_positive_scores = row_scores.masked_fill(~row_positive, torch.finfo(scores.dtype).min)
    e2r = (torch.logsumexp(row_scores, dim=1) - torch.logsumexp(row_positive_scores, dim=1)).mean()

    col_scores = scores[:, col_keep].T
    col_positive = positive_mask[:, col_keep].T
    col_positive_scores = col_scores.masked_fill(~col_positive, torch.finfo(scores.dtype).min)
    r2e = (torch.logsumexp(col_scores, dim=1) - torch.logsumexp(col_positive_scores, dim=1)).mean()
    return (e2r + r2e) / 2, e2r, r2e


class LayerwiseAdaMerge(torch.nn.Module):
    """Retrieval adaptation of AdaMerging's layer-wise task-vector coefficients.

    Base weights and directional task vectors are immutable. One coefficient per
    floating parameter tensor is shared across matched production seeds. The original
    AdaMerging classifier-entropy objective is replaced by a labeled bidirectional
    retrieval objective because broad protein-reaction associations are available.
    """

    def __init__(
        self,
        base_states: list[dict[str, torch.Tensor]],
        protein_states: list[dict[str, torch.Tensor]],
        reaction_states: list[dict[str, torch.Tensor]],
        config: ModelConfig,
        *,
        initial_protein_alpha: float,
        initial_reaction_alpha: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not (len(base_states) == len(protein_states) == len(reaction_states)) or not base_states:
            raise ValueError("matched non-empty seed state lists are required")
        self.models = torch.nn.ModuleList([TerpeneDualTower(config).to(device).eval() for _ in base_states])
        for model in self.models:
            for parameter in model.parameters():
                parameter.requires_grad = False
        self.base_states: list[dict[str, torch.Tensor]] = []
        self.protein_deltas: list[dict[str, torch.Tensor]] = []
        self.reaction_deltas: list[dict[str, torch.Tensor]] = []
        self.protein_names = _tower_parameter_names(base_states[0], "protein_tower.")
        self.reaction_names = _tower_parameter_names(base_states[0], "reaction_tower.")
        self.names = self.protein_names + self.reaction_names
        for base, protein, reaction in zip(base_states, protein_states, reaction_states, strict=True):
            _validate_directional_expert(base, protein, allowed_prefix="protein_tower.")
            _validate_directional_expert(base, reaction, allowed_prefix="reaction_tower.")
            self.base_states.append({key: value.detach().to(device) for key, value in base.items()})
            self.protein_deltas.append({key: (protein[key] - base[key]).detach().to(device) for key in self.protein_names})
            self.reaction_deltas.append({key: (reaction[key] - base[key]).detach().to(device) for key in self.reaction_names})

        initial = [float(initial_protein_alpha)] * len(self.protein_names) + [float(initial_reaction_alpha)] * len(self.reaction_names)
        if any(value < 0 or value > 1 for value in initial):
            raise ValueError("initial AdaMerging coefficients must be in [0,1]")
        self.alpha_raw = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32, device=device))
        self.name_to_alpha = {name: index for index, name in enumerate(self.names)}

    def alphas(self) -> torch.Tensor:
        # Original AdaMerging clamps learned task coefficients to [0,1].
        return torch.clamp(self.alpha_raw, min=0.0, max=1.0)

    def alpha_dict(self) -> dict[str, float]:
        values = self.alphas().detach().cpu().tolist()
        return dict(zip(self.names, map(float, values), strict=True))

    def _tower_params(self, seed_index: int, prefix: str) -> dict[str, torch.Tensor]:
        base = self.base_states[seed_index]
        deltas = self.protein_deltas[seed_index] if prefix == "protein_tower." else self.reaction_deltas[seed_index]
        names = self.protein_names if prefix == "protein_tower." else self.reaction_names
        return {
            name.removeprefix(prefix): base[name] + self.alphas()[self.name_to_alpha[name]] * deltas[name]
            for name in names
        }

    def encode(self, seed_index: int, protein_features: torch.Tensor, reaction_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        model = self.models[seed_index]
        proteins = functional_call(model.protein_tower, self._tower_params(seed_index, "protein_tower."), (protein_features,))
        reactions = functional_call(model.reaction_tower, self._tower_params(seed_index, "reaction_tower."), (reaction_features,))
        return proteins, reactions


def _load_matched_states(base_dir: Path, protein_dir: Path, reaction_dir: Path) -> tuple[list[str], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], ModelConfig]:
    signatures = [checkpoint_names(path) for path in [base_dir, protein_dir, reaction_dir]]
    if not signatures[0] or any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(f"checkpoint sets differ: {signatures}")
    base_states: list[dict[str, torch.Tensor]] = []
    protein_states: list[dict[str, torch.Tensor]] = []
    reaction_states: list[dict[str, torch.Tensor]] = []
    config: ModelConfig | None = None
    for name in signatures[0]:
        payloads = [torch.load(path / "models" / name, map_location="cpu", weights_only=False) for path in [base_dir, protein_dir, reaction_dir]]
        local = ModelConfig(**payloads[0]["model_config"])
        if config is None:
            config = local
        elif local != config:
            raise ValueError("model configs differ across seeds")
        base_states.append(payloads[0]["model_state_dict"])
        protein_states.append(payloads[1]["model_state_dict"])
        reaction_states.append(payloads[2]["model_state_dict"])
    assert config is not None
    return list(signatures[0]), base_states, protein_states, reaction_states, config


def _batch_indices(
    pair_proteins: np.ndarray,
    pair_reactions: np.ndarray,
    batch_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_p = pair_proteins[batch_rows]
    selected_r = pair_reactions[batch_rows]
    proteins, p_inverse = np.unique(selected_p, return_inverse=True)
    reactions, r_inverse = np.unique(selected_r, return_inverse=True)
    positive = np.zeros((len(proteins), len(reactions)), dtype=bool)
    positive[p_inverse, r_inverse] = True
    return proteins, reactions, positive


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn layer-wise AdaMerging coefficients for bidirectional broad retrieval.")
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--protein-expert-dir", type=Path, required=True)
    parser.add_argument("--reaction-expert-dir", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--association-source", default="uniprot_rhea_cached")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-pairs", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--initial-protein-alpha", type=float, default=0.9)
    parser.add_argument("--initial-reaction-alpha", type=float, default=0.9)
    parser.add_argument("--alpha-prior-weight", type=float, default=0.0)
    parser.add_argument("--alpha-prior", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.temperature <= 0 or args.steps <= 0 or args.batch_pairs <= 0:
        raise ValueError("temperature, steps and batch-pairs must be positive")

    seed_everything(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    base_dir = args.base_dir.resolve(); protein_dir = args.protein_expert_dir.resolve(); reaction_dir = args.reaction_expert_dir.resolve(); universe = args.universe_dir.resolve(); output = args.output_dir.resolve()
    names, base_states, protein_states, reaction_states, config = _load_matched_states(base_dir, protein_dir, reaction_dir)

    protein_features, protein_ids = load_protein_library(universe / "proteins")
    schema = load_feature_schema(base_dir)
    reaction_features, reaction_ids = load_registered_reaction_feature_library(universe / "reaction_features/drfp_categorical_v1", schema)
    pindex = {value: i for i, value in enumerate(protein_ids)}
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    associations = pd.read_csv(universe / "associations.csv", dtype=str).fillna("")
    if args.association_source:
        associations = associations[associations["source"].eq(args.association_source)]
    associations = associations[associations["protein_id"].isin(pindex) & associations["reaction_id"].isin(rindex)].drop_duplicates(["protein_id", "reaction_id"])
    pair_proteins = associations["protein_id"].map(pindex).to_numpy(np.int64)
    pair_reactions = associations["reaction_id"].map(rindex).to_numpy(np.int64)
    if len(pair_proteins) < args.batch_pairs:
        raise ValueError("not enough broad associations for one batch")

    merger = LayerwiseAdaMerge(
        base_states, protein_states, reaction_states, config,
        initial_protein_alpha=args.initial_protein_alpha,
        initial_reaction_alpha=args.initial_reaction_alpha,
        device=device,
    )
    optimizer = torch.optim.Adam([merger.alpha_raw], lr=args.learning_rate)
    initial_alpha = merger.alphas().detach().clone()
    history: list[dict[str, object]] = []

    for step in range(args.steps):
        rows = rng.choice(len(pair_proteins), size=args.batch_pairs, replace=False)
        p_rows, r_rows, positive_np = _batch_indices(pair_proteins, pair_reactions, rows)
        p_batch = torch.as_tensor(protein_features[p_rows], dtype=torch.float32, device=device)
        r_batch = torch.as_tensor(reaction_features[r_rows], dtype=torch.float32, device=device)
        positive = torch.as_tensor(positive_np, dtype=torch.bool, device=device)
        seed_index = step % len(names)
        p_emb, r_emb = merger.encode(seed_index, p_batch, r_batch)
        scores = (p_emb @ r_emb.T) / args.temperature
        loss, e2r_loss, r2e_loss = multipositive_bidirectional_loss(scores, positive)
        prior_loss = torch.mean((merger.alphas() - float(args.alpha_prior)) ** 2)
        total = loss + float(args.alpha_prior_weight) * prior_loss
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        # Project raw coefficients so the optimizer cannot become trapped outside the
        # official AdaMerging [0,1] coefficient interval by clamp's zero gradient.
        with torch.no_grad():
            merger.alpha_raw.clamp_(0.0, 1.0)
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            record = {
                "step": step + 1,
                "seed_checkpoint": names[seed_index],
                "loss": float(total.detach().cpu()),
                "e2r_loss": float(e2r_loss.detach().cpu()),
                "r2e_loss": float(r2e_loss.detach().cpu()),
                "prior_loss": float(prior_loss.detach().cpu()),
                "alphas": merger.alpha_dict(),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    output.mkdir(parents=True, exist_ok=True); (output / "models").mkdir(exist_ok=True)
    alpha_dict = merger.alpha_dict()
    for i, name in enumerate(names):
        base_payload = torch.load(base_dir / "models" / name, map_location="cpu", weights_only=False)
        payload = {key: value for key, value in base_payload.items() if key != "model_state_dict"}
        payload["model_state_dict"] = materialize_layerwise_merge(base_states[i], protein_states[i], reaction_states[i], alpha_dict)
        payload["retrieval_adamerging"] = {
            "alphas": alpha_dict,
            "protein_expert": str((protein_dir / "models" / name).resolve()),
            "reaction_expert": str((reaction_dir / "models" / name).resolve()),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_paper": UPSTREAM_PAPER,
            "adaptation": "layer-wise task-vector coefficients optimized with bidirectional multi-positive retrieval contrastive loss",
        }
        torch.save(payload, output / "models" / name)
    for filename in ASSET_FILES:
        source = base_dir / filename
        if source.exists(): shutil.copy2(source, output / filename)
    pd.DataFrame([{k: v for k, v in item.items() if k != "alphas"} for item in history]).to_csv(output / "training_history.csv", index=False)
    summary = {
        "model_type": "retrieval_adapted_layerwise_adamerging",
        "upstream_method": UPSTREAM_PAPER,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "base_dir": str(base_dir),
        "protein_expert_dir": str(protein_dir),
        "reaction_expert_dir": str(reaction_dir),
        "universe_dir": str(universe),
        "association_source": args.association_source,
        "n_associations": int(len(associations)),
        "steps": args.steps,
        "batch_pairs": args.batch_pairs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "shared_across_seeds": True,
        "initial_alphas": dict(zip(merger.names, map(float, initial_alpha.cpu().tolist()), strict=True)),
        "learned_alphas": alpha_dict,
        "objective_adaptation": "AdaMerging coefficient learning retained; classification entropy replaced with supervised bidirectional retrieval loss because association labels are available.",
        "checkpoints": [str((output / "models" / name).resolve()) for name in names],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
