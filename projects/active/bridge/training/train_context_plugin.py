from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from projects.active.bridge.model import ExpertEvidence, ExpertSpec
from projects.active.bridge.model.assets import FibreAssetStore, ROOT
from projects.active.bridge.model.checkpoint import load_fibre_checkpoint
from projects.active.bridge.model.plugins import save_expert_plugin
from projects.active.bridge.training.train import (
    DEFAULT_PAIRS,
    positive_maps,
    reaction_neighbors,
    sample_negatives,
    split_double_cold,
)

DEFAULT_CHECKPOINT = ROOT / "results/fibre_relational_main_v1/best.pt"
DEFAULT_OUTPUT = ROOT / "results/fibre_relational_main_v1/plugins/context.pt"


class ContextSimilarity:
    """Leave-one-known-positive context matching existing runtime semantics."""

    def __init__(self, assets: FibreAssetStore, by_r: dict[str, set[str]], by_p: dict[str, set[str]]):
        self.assets = assets
        self.by_r = by_r
        self.by_p = by_p

    @staticmethod
    def _max_cosine(candidate: np.ndarray, seeds: np.ndarray) -> float:
        c = np.asarray(candidate, dtype=np.float32)
        s = np.asarray(seeds, dtype=np.float32)
        cn = max(float(np.linalg.norm(c)), 1e-8)
        sn = np.maximum(np.linalg.norm(s, axis=1), 1e-8)
        return float(np.max((s @ c) / (sn * cn)))

    def r2e(self, protein_id: str, reaction_id: str) -> float | None:
        seeds = set(self.by_r.get(str(reaction_id), set()))
        seeds.discard(str(protein_id))
        rows = [self.assets.protein_index[x] for x in seeds if x in self.assets.protein_index]
        row = self.assets.protein_index.get(str(protein_id))
        if row is None or not rows:
            return None
        return self._max_cosine(self.assets.protein[row], self.assets.protein[np.asarray(rows)])

    def e2r(self, protein_id: str, reaction_id: str) -> float | None:
        seeds = set(self.by_p.get(str(protein_id), set()))
        seeds.discard(str(reaction_id))
        rows = [self.assets.reaction_index[x] for x in seeds if x in self.assets.reaction_index]
        row = self.assets.reaction_index.get(str(reaction_id))
        if row is None or not rows:
            return None
        return self._max_cosine(
            self.assets.reaction_features[row],
            self.assets.reaction_features[np.asarray(rows)],
        )


def distances(
    model,
    assets: FibreAssetStore,
    pairs: list[tuple[str, str]],
    context: list[float],
    device: torch.device,
    *,
    chunk: int = 32,
) -> torch.Tensor:
    out: list[torch.Tensor] = []
    base_experts = set(model.expert_adapters.keys()) - {"context"}
    for start in range(0, len(pairs), chunk):
        part = pairs[start : start + chunk]
        values = np.asarray(context[start : start + len(part)], dtype=np.float32)
        kwargs = assets.batch(
            [p for p, _ in part],
            [r for _, r in part],
            device,
            expert_names=base_experts,
        )
        kwargs["evidence"]["context"] = ExpertEvidence(
            torch.as_tensor(values[:, None], device=device),
            torch.ones(len(part), dtype=torch.bool, device=device),
        )
        out.append(model.relation_distance(**kwargs))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the frozen-core BRIDGE context expert plug-in.")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hard-negatives", type=int, default=2)
    ap.add_argument("--random-negatives", type=int, default=1)
    ap.add_argument("--neighbor-k", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    assets = FibreAssetStore()
    pairs = pd.read_csv(args.pairs, dtype=str).fillna("")
    pairs = assets.supported_pairs(pairs)
    train, _, _ = split_double_cold(pairs)
    all_by_r, all_by_p = positive_maps(pairs)
    train_by_r, train_by_p = positive_maps(train)
    train_proteins = sorted(set(train.protein_id.astype(str)))
    train_reactions = sorted(set(train.reaction_id.astype(str)))
    neighbors = reaction_neighbors(
        assets,
        set(train_reactions),
        set(train_reactions),
        topk=args.neighbor_k,
        device=device,
    )
    context = ContextSimilarity(assets, train_by_r, train_by_p)

    model, payload = load_fibre_checkpoint(args.checkpoint, device=device, eval_mode=True)
    if "context" in model.expert_adapters:
        raise ValueError("base checkpoint already contains context")
    model.register_expert("context", ExpertSpec(1))
    model.expert_adapters["context"].to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.expert_adapters["context"].parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        model.expert_adapters["context"].parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(args.seed + epoch).permutation(len(train))
        epoch_losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            batch = train.iloc[order[start : start + args.batch_size]]
            r2e, e2r = sample_negatives(
                batch,
                rng=rng,
                by_r=train_by_r,
                by_p=train_by_p,
                all_by_r=all_by_r,
                all_by_p=all_by_p,
                neighbors=neighbors,
                proteins=train_proteins,
                reactions=train_reactions,
                hard=args.hard_negatives,
                random_count=args.random_negatives,
            )
            losses: list[torch.Tensor] = []
            for direction, negs in (("r2e", r2e), ("e2r", e2r)):
                if not negs:
                    continue
                pos_pairs: list[tuple[str, str]] = []
                neg_pairs: list[tuple[str, str]] = []
                pos_ctx: list[float] = []
                neg_ctx: list[float] = []
                for p, r, origin in negs:
                    source = batch.iloc[int(origin)]
                    pp, rr = str(source.protein_id), str(source.reaction_id)
                    pos_value = context.r2e(pp, rr) if direction == "r2e" else context.e2r(pp, rr)
                    neg_value = context.r2e(str(p), str(r)) if direction == "r2e" else context.e2r(str(p), str(r))
                    if pos_value is None or neg_value is None:
                        continue
                    pos_pairs.append((pp, rr)); neg_pairs.append((str(p), str(r)))
                    pos_ctx.append(pos_value); neg_ctx.append(neg_value)
                if not pos_pairs:
                    continue
                pos = distances(model, assets, pos_pairs, pos_ctx, device)
                neg = distances(model, assets, neg_pairs, neg_ctx, device)
                losses.append(F.softplus(args.margin + pos - neg).mean())
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        print({"epoch": epoch, "loss": float(np.mean(epoch_losses)) if epoch_losses else None}, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_expert_plugin(
        args.output,
        name="context",
        input_dim=1,
        model=model,
        metadata={
            "base_checkpoint": str(args.checkpoint),
            "base_epoch": payload.get("epoch"),
            "training": "leave-one-known-positive max-cosine context; frozen BRIDGE core",
            "epochs": args.epochs,
            "seed": args.seed,
        },
    )
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
