from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[3]
CAGE = ROOT / "external_repos/EnzymeCAGE"
if str(CAGE) not in sys.path:
    sys.path.insert(0, str(CAGE))

from enzymecage.base import UID_COL, RXN_COL
from enzymecage.dataset.geometric import load_geometric_dataset
from enzymecage.model import EnzymeCAGE

FAMILIES = ("p450", "phosphatase", "terpene")
GENERIC_CKPT = CAGE / "checkpoints/pretrain/seed_42/epoch_19.pth"
FAMILY_CKPT = {
    fam: CAGE / f"checkpoints/domain-specific-ft/{fam}/seed_42/epoch_9.pth"
    for fam in FAMILIES
}
OUT = ROOT / "results/enzymecage_family_response_v1"
PROJ_DIM = 32
SEED = 20261002


def resolve_config(path: Path) -> SimpleNamespace:
    raw = yaml.safe_load(path.read_text())
    path_keys = {
        "data_path", "rxn_fp", "mol_conformation", "reaction_center",
        "protein_gvp_feat", "esm_mean_feature", "esm_node_feature",
    }
    for key in path_keys:
        value = raw.get(key)
        if value:
            p = Path(value)
            raw[key] = str(p if p.is_absolute() else (CAGE / p).resolve())
    return SimpleNamespace(**raw)


def reuse_identical_sequence_structure_features(df, protein_gvp_feat, esm_node_feature):
    seq_df = df[[UID_COL, "sequence"]].drop_duplicates()
    seq_to_source = {}
    for uid, seq in seq_df.itertuples(index=False):
        if uid not in protein_gvp_feat or uid not in esm_node_feature:
            continue
        if len(protein_gvp_feat[uid][0]) != len(esm_node_feature[uid]):
            continue
        seq_to_source.setdefault(seq, uid)
    reused = 0
    for uid, seq in seq_df.itertuples(index=False):
        source = seq_to_source.get(seq)
        if source is None or source == uid:
            continue
        invalid = uid not in protein_gvp_feat or uid not in esm_node_feature
        if not invalid:
            invalid = len(protein_gvp_feat[uid][0]) != len(esm_node_feature[uid])
        if invalid:
            protein_gvp_feat[uid] = protein_gvp_feat[source]
            esm_node_feature[uid] = esm_node_feature[source]
            reused += 1
    return reused


def build_dataset(conf):
    df = pd.read_csv(conf.data_path, dtype=str).fillna("")
    if UID_COL not in df.columns and "enzyme" in df.columns:
        df[UID_COL] = df["enzyme"]
    if RXN_COL not in df.columns and "reaction" in df.columns:
        df[RXN_COL] = df["reaction"]
    if "Label" not in df.columns:
        df["Label"] = 0.0
    else:
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0.0).astype(float)

    gvp = torch.load(conf.protein_gvp_feat, map_location="cpu", weights_only=False)
    esm_node = torch.load(conf.esm_node_feature, map_location="cpu", weights_only=False)
    reused = reuse_identical_sequence_structure_features(df, gvp, esm_node)
    valid = {
        uid for uid in set(gvp) & set(esm_node)
        if int(gvp[uid][0].shape[0]) == int(esm_node[uid].shape[0])
    }
    filtered = df[df[UID_COL].isin(valid)].reset_index(drop=True)
    dataset = load_geometric_dataset(
        filtered,
        gvp,
        conf.rxn_fp,
        conf.mol_conformation,
        esm_node,
        conf.esm_mean_feature,
        conf.reaction_center,
    )
    return filtered, dataset, {
        "input_rows": int(len(df)),
        "evaluable_rows": int(len(filtered)),
        "coverage": float(len(filtered) / max(len(df), 1)),
        "reused_identical_sequence_structures": int(reused),
    }


def make_model(conf, checkpoint: Path, device: torch.device):
    model = EnzymeCAGE(
        use_esm=conf.use_esm,
        use_structure=conf.use_structure,
        use_drfp=conf.use_drfp,
        use_prods_info=conf.use_prods_info,
        esm_model=conf.esm_model,
        interaction_method=conf.interaction_method,
        rxn_inner_interaction=conf.rxn_inner_interaction,
        pocket_inner_interaction=conf.pocket_inner_interaction,
        device=str(device),
    )
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def fixed_projection(dim: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(SEED + dim)
    matrix = torch.randn(dim, PROJ_DIM, generator=gen, dtype=torch.float32)
    matrix /= math.sqrt(PROJ_DIM)
    return matrix.to(device)


class Capture:
    def __init__(self, model: EnzymeCAGE, device: torch.device):
        self.device = device
        self.data = {}
        self.masks = None
        self.proj = {
            "fused": fixed_projection(256, device),
            "hidden1": fixed_projection(2048, device),
            "hidden2": fixed_projection(1024, device),
        }
        self.handles = [
            model.interaction_model.register_forward_pre_hook(
                self._pre_interaction, with_kwargs=True
            ),
            model.interaction_model.register_forward_hook(
                self._hook_interaction, with_kwargs=True
            ),
            model.mlp[0].register_forward_hook(self._hook_hidden("hidden1")),
            model.mlp[1].register_forward_hook(self._hook_hidden("hidden2")),
        ]

    def _pre_interaction(self, module, args, kwargs):
        self.masks = (
            kwargs["enz_node_feature_mask"].detach(),
            kwargs["substrate_node_feature_mask"].detach(),
        )

    def _hook_interaction(self, module, args, kwargs, output):
        fused, attention = output
        self._store_repr("fused", fused)
        enz_mask, sub_mask = self.masks
        valid = enz_mask.unsqueeze(-1) & sub_mask.unsqueeze(1)
        weights = attention.masked_fill(~valid, 0.0)
        flat = weights.flatten(1)
        flat_sum = flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
        prob = flat / flat_sum
        entropy = -(prob.clamp_min(1e-12).log() * prob).sum(dim=1)
        valid_count = valid.flatten(1).sum(dim=1).clamp_min(1)
        entropy_norm = entropy / valid_count.float().log().clamp_min(1.0)
        self.data["attention_entropy"] = entropy_norm.detach().cpu().numpy()
        self.data["attention_peak"] = prob.max(dim=1).values.detach().cpu().numpy()

    def _hook_hidden(self, name):
        def hook(module, args, output):
            self._store_repr(name, output)
        return hook

    def _store_repr(self, name, x):
        x = x.float()
        self.data[f"{name}_norm"] = x.norm(dim=1).detach().cpu().numpy()
        projected = x @ self.proj[name]
        projected = torch.nn.functional.normalize(projected, dim=1)
        self.data[f"{name}_proj"] = projected.detach().cpu().numpy()

    def pop(self):
        data = self.data
        self.data = {}
        return data

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def run_checkpoint(conf, dataset, checkpoint: Path, device: torch.device, batch_size: int):
    model = make_model(conf, checkpoint, device)
    capture = Capture(model, device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=["protein", "reaction_feature", "esm_feature", "substrates", "products"],
    )
    blocks = []
    for batch in loader:
        logits = model(batch.to(device))
        cap = capture.pop()
        cap["logit"] = logits.detach().float().cpu().numpy()
        blocks.append(cap)
    capture.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {}
    for key in blocks[0]:
        out[key] = np.concatenate([b[key] for b in blocks], axis=0)
    return out


def cosine_shift(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.sum(a * b, axis=1)


def norm_log_ratio(base: np.ndarray, adapted: np.ndarray) -> np.ndarray:
    return np.log(np.clip(adapted, 1e-8, None) / np.clip(base, 1e-8, None))


def z_by_query(frame: pd.DataFrame, column: str) -> pd.Series:
    def transform(x):
        values = x.to_numpy(float)
        sd = values.std()
        if sd < 1e-8:
            return pd.Series(np.zeros(len(values)), index=x.index)
        return pd.Series((values - values.mean()) / sd, index=x.index)
    return frame.groupby("CANO_RXN_SMILES", sort=False)[column].transform(transform)


def top_decile_mean(values: pd.Series) -> float:
    x = np.sort(values.to_numpy(float))
    n = max(1, int(math.ceil(len(x) * 0.1)))
    return float(x[-n:].mean())


def aggregate_query_features(pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query, g in pair.groupby("CANO_RXN_SMILES", sort=False):
        row = {"CANO_RXN_SMILES": query, "candidate_count": int(len(g))}
        family_scores = []
        for fam in FAMILIES:
            response = g[f"{fam}_normalized_logit_response"]
            score = top_decile_mean(response)
            family_scores.append(score)
            row[f"{fam}_response_top10pct_mean"] = score
            row[f"{fam}_response_rms"] = float(np.sqrt(np.mean(np.square(response))))
            row[f"{fam}_fused_shift_mean"] = float(g[f"{fam}_fused_cosine_shift"].mean())
            row[f"{fam}_hidden2_shift_mean"] = float(g[f"{fam}_hidden2_cosine_shift"].mean())
            row[f"{fam}_attention_entropy_delta_mean"] = float(
                g[f"{fam}_attention_entropy_delta"].mean()
            )
            row[f"{fam}_attention_peak_delta_mean"] = float(
                g[f"{fam}_attention_peak_delta"].mean()
            )
        scores = np.asarray(family_scores, dtype=float)
        shifted = scores - scores.max()
        prob = np.exp(shifted)
        prob /= prob.sum()
        ordered = np.sort(scores)
        row["family_response_max"] = float(scores.max())
        row["family_response_margin"] = float(ordered[-1] - ordered[-2])
        row["family_response_entropy"] = float(
            -(prob * np.log(np.clip(prob, 1e-12, None))).sum()
        )
        row["family_response_winner"] = FAMILIES[int(scores.argmax())]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-family", choices=FAMILIES)
    ap.add_argument("--config")
    ap.add_argument("--tag")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.config:
        conf_path = Path(args.config).resolve()
        tag = args.tag or conf_path.stem
    else:
        if not args.dataset_family:
            raise ValueError("--dataset-family or --config is required")
        conf_path = CAGE / f"config/infer/external-test-set/{args.dataset_family}/wo-finetune.yaml"
        tag = args.dataset_family
    conf = resolve_config(conf_path)
    device = torch.device(args.device)
    frame, dataset, audit = build_dataset(conf)
    if args.limit:
        limit = min(args.limit, len(frame))
        frame = frame.iloc[:limit].reset_index(drop=True)
        dataset = dataset[:limit]
        audit["limited_rows"] = limit

    print(f"dataset={tag} rows={len(frame)}", flush=True)
    generic = run_checkpoint(conf, dataset, GENERIC_CKPT, device, args.batch_size)
    key_columns = list(dict.fromkeys([UID_COL, RXN_COL, "CANO_RXN_SMILES"]))
    pair = frame[key_columns].copy()
    pair["generic_logit"] = generic["logit"]
    pair["generic_attention_entropy"] = generic["attention_entropy"]
    pair["generic_attention_peak"] = generic["attention_peak"]

    for fam in FAMILIES:
        print(f"family={fam}", flush=True)
        adapted = run_checkpoint(conf, dataset, FAMILY_CKPT[fam], device, args.batch_size)
        pair[f"{fam}_logit"] = adapted["logit"]
        pair[f"{fam}_delta_logit"] = adapted["logit"] - generic["logit"]
        for name in ("fused", "hidden1", "hidden2"):
            pair[f"{fam}_{name}_cosine_shift"] = cosine_shift(
                generic[f"{name}_proj"], adapted[f"{name}_proj"]
            )
            pair[f"{fam}_{name}_norm_log_ratio"] = norm_log_ratio(
                generic[f"{name}_norm"], adapted[f"{name}_norm"]
            )
        pair[f"{fam}_attention_entropy_delta"] = (
            adapted["attention_entropy"] - generic["attention_entropy"]
        )
        pair[f"{fam}_attention_peak_delta"] = (
            adapted["attention_peak"] - generic["attention_peak"]
        )
        pair[f"{fam}_z_logit"] = z_by_query(pair, f"{fam}_logit")

    pair["generic_z_logit"] = z_by_query(pair, "generic_logit")
    for fam in FAMILIES:
        pair[f"{fam}_normalized_logit_response"] = (
            pair[f"{fam}_z_logit"] - pair["generic_z_logit"]
        )

    query = aggregate_query_features(pair)
    out = OUT / tag
    if args.limit:
        out = OUT / f"{tag}_smoke{args.limit}"
    out.mkdir(parents=True, exist_ok=True)
    pair.to_csv(out / "pair_features.csv.gz", index=False, compression="gzip")
    query.to_csv(out / "query_features.csv", index=False)
    summary = {
        "schema": "enzymecage-family-response-v1",
        "dataset_tag": tag,
        "source_config": str(conf_path),
        "status": "completed",
        "audit": audit,
        "generic_checkpoint": str(GENERIC_CKPT.relative_to(ROOT)),
        "family_checkpoints": {
            fam: str(FAMILY_CKPT[fam].relative_to(ROOT)) for fam in FAMILIES
        },
        "projection_dim": PROJ_DIM,
        "pair_features": [
            "query-normalized logit response",
            "projected interaction-fused cosine shift and exact norm log-ratio",
            "projected hidden-2048 cosine shift and exact norm log-ratio",
            "projected hidden-1024 cosine shift and exact norm log-ratio",
            "pocket-substrate attention entropy and peak deltas",
        ],
        "query_features": [
            "family response top-decile mean",
            "family response RMS",
            "mean interaction/hidden/attention shifts",
            "cross-family winner, margin and entropy",
        ],
        "labels_used_for_extraction": False,
        "external_labels_allowed_for_router_fit": False,
        "rows": int(len(pair)),
        "queries": int(query["CANO_RXN_SMILES"].nunique()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
