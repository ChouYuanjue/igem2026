from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from enzymecage.base import UID_COL, RXN_COL
from reproducibility.bime_rank.scripts.extract_enzymecage_family_response_v1 import (
    FAMILIES,
    FAMILY_CKPT,
    GENERIC_CKPT,
    PROJ_DIM,
    build_dataset,
    fixed_projection,
    make_model,
    resolve_config,
)

OUT = ROOT / "results/enzymecage_reaction_family_response_v1"
SEED = 20261002


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.float().unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


def attention_stats(attn: torch.Tensor, qmask: torch.Tensor, kmask: torch.Tensor):
    valid = qmask.unsqueeze(-1) & kmask.unsqueeze(1)
    weights = attn.masked_fill(~valid, 0.0)
    flat = weights.flatten(1)
    prob = flat / flat.sum(1, keepdim=True).clamp_min(1e-12)
    entropy = -(prob.clamp_min(1e-12).log() * prob).sum(1)
    valid_count = valid.flatten(1).sum(1).clamp_min(1)
    entropy = entropy / valid_count.float().log().clamp_min(1.0)
    peak = prob.max(1).values
    return entropy, peak


@torch.no_grad()
def run_reaction_only(conf, dataset, checkpoint: Path, device: torch.device, batch_size: int):
    model = make_model(conf, checkpoint, device)
    pre_proj = fixed_projection(256, device)
    post_proj = fixed_projection(256, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=["protein", "reaction_feature", "esm_feature", "substrates", "products"],
        num_workers=0,
    )
    blocks = []
    for batch in loader:
        batch = batch.to(device)
        sub, prod = model.encode_molecule(batch)
        sub_b, sub_m = to_dense_batch(sub, batch["substrates"].batch)
        prod_b, prod_m = to_dense_batch(prod, batch["products"].batch)

        pre = torch.cat([masked_mean(sub_b, sub_m), masked_mean(prod_b, prod_m)], dim=1)
        pre_norm = pre.float().norm(dim=1)
        pre_z = torch.nn.functional.normalize(pre.float() @ pre_proj, dim=1)

        attn_entropy = torch.zeros(len(pre), device=device)
        attn_peak = torch.zeros(len(pre), device=device)
        if model.rxn_inner_interaction:
            sub_rc, _ = to_dense_batch(
                batch["substrates"].reacting_center, batch["substrates"].batch
            )
            prod_rc, _ = to_dense_batch(
                batch["products"].reacting_center, batch["products"].batch
            )
            sw = (sub_rc * 0.5 + 0.1) * sub_m
            pw = (prod_rc * 0.5 + 0.1) * prod_m
            bias = torch.einsum("bi,bj->bij", sw, pw)
            sub2, attn = model.reaction_cross_attn(
                sub_b, prod_b, prod_b, sub_m, prod_m, bias
            )
            if model.use_prods_info:
                prod2, _ = model.reaction_cross_attn(
                    prod_b, sub2, sub2, prod_m, sub_m, bias.transpose(1, 2)
                )
            else:
                prod2 = prod_b
            attn_entropy, attn_peak = attention_stats(attn, sub_m, prod_m)
        else:
            sub2, prod2 = sub_b, prod_b

        post = torch.cat([masked_mean(sub2, sub_m), masked_mean(prod2, prod_m)], dim=1)
        post_norm = post.float().norm(dim=1)
        post_z = torch.nn.functional.normalize(post.float() @ post_proj, dim=1)

        blocks.append({
            "pre_proj": pre_z.cpu().numpy(),
            "pre_norm": pre_norm.cpu().numpy(),
            "post_proj": post_z.cpu().numpy(),
            "post_norm": post_norm.cpu().numpy(),
            "attn_entropy": attn_entropy.float().cpu().numpy(),
            "attn_peak": attn_peak.float().cpu().numpy(),
        })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}


def cosine_shift(base, adapted):
    return 1.0 - np.sum(base * adapted, axis=1)


def norm_log_ratio(base, adapted):
    return np.log(np.clip(adapted, 1e-8, None) / np.clip(base, 1e-8, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    conf = resolve_config(Path(args.config))
    # One row per reaction. Protein is only a carrier required by the dataset;
    # no protein-side tensor is used by the reaction-only extractor.
    df = pd.read_csv(conf.data_path, dtype=str).fillna("")
    if UID_COL not in df.columns and "enzyme" in df.columns:
        df[UID_COL] = df["enzyme"]
    if RXN_COL not in df.columns and "reaction" in df.columns:
        df[RXN_COL] = df["reaction"]
    unique = df.drop_duplicates(RXN_COL, keep="first").reset_index(drop=True)

    tmp = OUT / args.tag / "unique_queries.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    unique.to_csv(tmp, index=False)
    conf.data_path = str(tmp)

    filtered, dataset, audit = build_dataset(conf)
    device = torch.device(args.device)
    generic = run_reaction_only(conf, dataset, GENERIC_CKPT, device, args.batch_size)
    families = {
        fam: run_reaction_only(conf, dataset, FAMILY_CKPT[fam], device, args.batch_size)
        for fam in FAMILIES
    }

    out = filtered[[c for c in [RXN_COL, "reaction_id", "CANO_RXN_SMILES"] if c in filtered.columns]].copy()
    if "reaction_id" not in out.columns:
        out["reaction_id"] = ""
    if "CANO_RXN_SMILES" not in out.columns:
        out["CANO_RXN_SMILES"] = out[RXN_COL].astype(str)

    for fam in FAMILIES:
        cur = families[fam]
        out[f"{fam}_reaction_pre_cosine_shift"] = cosine_shift(
            generic["pre_proj"], cur["pre_proj"]
        )
        out[f"{fam}_reaction_pre_norm_log_ratio"] = norm_log_ratio(
            generic["pre_norm"], cur["pre_norm"]
        )
        out[f"{fam}_reaction_post_cosine_shift"] = cosine_shift(
            generic["post_proj"], cur["post_proj"]
        )
        out[f"{fam}_reaction_post_norm_log_ratio"] = norm_log_ratio(
            generic["post_norm"], cur["post_norm"]
        )
        out[f"{fam}_reaction_attention_entropy_delta"] = (
            cur["attn_entropy"] - generic["attn_entropy"]
        )
        out[f"{fam}_reaction_attention_peak_delta"] = (
            cur["attn_peak"] - generic["attn_peak"]
        )

    # Cross-family normalized response energy from reaction-side shifts only.
    primary = np.column_stack([
        out[f"{fam}_reaction_post_cosine_shift"].to_numpy(float) for fam in FAMILIES
    ])
    denom = np.clip(primary.sum(1, keepdims=True), 1e-8, None)
    prob = primary / denom
    ordered = np.sort(primary, axis=1)
    out["reaction_family_winner"] = [FAMILIES[i] for i in primary.argmax(1)]
    out["reaction_family_margin"] = ordered[:, -1] - ordered[:, -2]
    out["reaction_family_entropy"] = -(
        np.clip(prob, 1e-12, None) * np.log(np.clip(prob, 1e-12, None))
    ).sum(1)

    out.to_csv(OUT / args.tag / "query_features.csv", index=False)
    summary = {
        "schema": "enzymecage-reaction-family-response-v1",
        "status": "completed",
        "tag": args.tag,
        "queries_input": int(df[RXN_COL].nunique()),
        "queries_evaluable": int(len(out)),
        "coverage": float(len(out) / max(df[RXN_COL].nunique(), 1)),
        "protein_branch_used_for_features": False,
        "reaction_branch": "molecule_encoder + reaction_cross_attn",
        "labels_used": False,
        "generic_checkpoint": str(GENERIC_CKPT.relative_to(ROOT)),
        "family_checkpoints": {
            fam: str(FAMILY_CKPT[fam].relative_to(ROOT)) for fam in FAMILIES
        },
        "audit": audit,
    }
    (OUT / args.tag / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
