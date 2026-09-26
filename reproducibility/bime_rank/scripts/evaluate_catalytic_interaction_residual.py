from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.fibre.kernel.interaction import BoundedBilinearInteraction
from projects.active.fibre.runtime.base_model import ModelConfig, TerpeneDualTower
from projects.active.fibre.runtime.cli import (
    load_protein_library,
    load_registered_reaction_feature_library,
)
from projects.active.fibre.runtime.evaluation_metrics import evaluate_ranking_frame

DEFAULT_BASE = ROOT / "results/cleanroom_internal_reaction_center_bounded_v3/base"
DEFAULT_OUTPUT = ROOT / "results/fibre_catalytic_interaction_residual_dev_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def encode_selected(
    model: TerpeneDualTower,
    values: np.ndarray,
    source_rows: np.ndarray,
    *,
    side: str,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    if side not in {"reaction", "protein"}:
        raise ValueError(side)
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source_rows), chunk_size):
            batch = torch.as_tensor(
                values[source_rows[start : start + chunk_size]],
                dtype=torch.float32,
                device=device,
            )
            encoded = (
                model.encode_reactions(batch)
                if side == "reaction"
                else model.encode_proteins(batch)
            )
            rows.append(encoded.detach().cpu().numpy().astype(np.float32, copy=False))
    if not rows:
        raise ValueError(f"no {side} rows selected")
    return np.concatenate(rows, axis=0)


def positive_mask_for_batch(
    batch_reactions: list[str],
    batch_enzymes: list[str],
    positive_by_reaction: dict[str, set[str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    cols: dict[str, list[int]] = {}
    for col, enzyme_id in enumerate(batch_enzymes):
        cols.setdefault(enzyme_id, []).append(col)
    mask = torch.zeros(
        (len(batch_reactions), len(batch_enzymes)),
        dtype=torch.bool,
        device=device,
    )
    for row, reaction_id in enumerate(batch_reactions):
        for enzyme_id in positive_by_reaction.get(reaction_id, ()):
            for col in cols.get(enzyme_id, ()):
                mask[row, col] = True
    if not bool(mask.diagonal().all()):
        raise AssertionError("positive minibatch diagonal was lost")
    return mask


def multi_positive_score_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float,
    reaction_loss_weight: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(reaction_loss_weight) <= 1.0:
        raise ValueError("reaction_loss_weight must be within [0, 1]")
    logits = scores / float(temperature)
    neg_inf = torch.finfo(logits.dtype).min
    positive = logits.masked_fill(~positive_mask, neg_inf)
    row_loss = (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive, dim=1)
    ).mean()
    col_loss = (
        torch.logsumexp(logits, dim=0) - torch.logsumexp(positive, dim=0)
    ).mean()
    return (
        float(reaction_loss_weight) * row_loss
        + (1.0 - float(reaction_loss_weight)) * col_loss
    )


def train_interaction(
    reaction_embeddings: np.ndarray,
    enzyme_embeddings: np.ndarray,
    train_pairs: pd.DataFrame,
    *,
    max_frobenius_norm: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    reaction_loss_weight: float,
    score_preservation_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[BoundedBilinearInteraction, list[dict[str, float]]]:
    reaction_ids = sorted(train_pairs.reaction_id.astype(str).unique())
    enzyme_ids = sorted(train_pairs.protein_id.astype(str).unique())
    reaction_index = {value: i for i, value in enumerate(reaction_ids)}
    enzyme_index = {value: i for i, value in enumerate(enzyme_ids)}
    pair_r = np.asarray(
        [reaction_index[x] for x in train_pairs.reaction_id.astype(str)],
        dtype=np.int64,
    )
    pair_e = np.asarray(
        [enzyme_index[x] for x in train_pairs.protein_id.astype(str)],
        dtype=np.int64,
    )
    positive_by_reaction: dict[str, set[str]] = {}
    for reaction_id, protein_id in train_pairs[["reaction_id", "protein_id"]].itertuples(
        index=False
    ):
        positive_by_reaction.setdefault(str(reaction_id), set()).add(str(protein_id))

    layer = BoundedBilinearInteraction(
        reaction_embeddings.shape[1],
        max_frobenius_norm=max_frobenius_norm,
    ).to(device)
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    identity_probe_r = torch.as_tensor(
        reaction_embeddings[: min(64, len(reaction_embeddings))],
        dtype=torch.float32,
        device=device,
    )
    identity_probe_e = torch.as_tensor(
        enzyme_embeddings[: min(64, len(enzyme_embeddings))],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        identity_diff = (
            layer.score_matrix(identity_probe_r, identity_probe_e)
            - identity_probe_r @ identity_probe_e.T
        ).abs().max()
    if float(identity_diff.cpu()) != 0.0:
        raise AssertionError("bilinear interaction is not exact identity before training")

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_pairs))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            if len(selected) < 2:
                continue
            local_r = pair_r[selected]
            local_e = pair_e[selected]
            r = torch.as_tensor(
                reaction_embeddings[local_r], dtype=torch.float32, device=device
            )
            e = torch.as_tensor(
                enzyme_embeddings[local_e], dtype=torch.float32, device=device
            )
            batch_reaction_ids = [reaction_ids[int(x)] for x in local_r]
            batch_enzyme_ids = [enzyme_ids[int(x)] for x in local_e]
            positive_mask = positive_mask_for_batch(
                batch_reaction_ids,
                batch_enzyme_ids,
                positive_by_reaction,
                device=device,
            )
            scores = layer.score_matrix(r, e)
            loss = multi_positive_score_loss(
                scores,
                positive_mask,
                temperature=temperature,
                reaction_loss_weight=reaction_loss_weight,
            )
            if score_preservation_weight > 0:
                base_scores = r @ e.T
                normalized_shift = (scores - base_scores) / float(max_frobenius_norm)
                loss = loss + float(score_preservation_weight) * normalized_shift.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        row = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            **layer.diagnostics(),
        }
        history.append(row)
        print(json.dumps({"stage": "train", **row}), flush=True)
    return layer, history


def score_rows(
    frame: pd.DataFrame,
    *,
    reaction_ids: list[str],
    enzyme_ids: list[str],
    reaction_embeddings: np.ndarray,
    enzyme_embeddings: np.ndarray,
    layer: BoundedBilinearInteraction,
    device: torch.device,
    batch_size: int = 16384,
) -> tuple[np.ndarray, np.ndarray]:
    rindex = {value: i for i, value in enumerate(reaction_ids)}
    eindex = {value: i for i, value in enumerate(enzyme_ids)}
    reaction_rows = np.asarray(
        [rindex[x] for x in frame.reaction_id.astype(str)], dtype=np.int64
    )
    enzyme_rows = np.asarray(
        [eindex[x] for x in frame.protein_id.astype(str)], dtype=np.int64
    )
    base_out: list[np.ndarray] = []
    corrected_out: list[np.ndarray] = []
    layer.eval()
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            rr = reaction_rows[start : start + batch_size]
            ee = enzyme_rows[start : start + batch_size]
            r = torch.as_tensor(
                reaction_embeddings[rr], dtype=torch.float32, device=device
            )
            e = torch.as_tensor(
                enzyme_embeddings[ee], dtype=torch.float32, device=device
            )
            base = (r * e).sum(dim=1)
            corrected = layer.score_pairs(r, e)
            base_out.append(base.cpu().numpy())
            corrected_out.append(corrected.cpu().numpy())
    return np.concatenate(base_out), np.concatenate(corrected_out)


def ranking_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    direction: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    local = frame.copy()
    local["score"] = np.asarray(scores, dtype=np.float64)
    if direction == "r2e":
        query_col, candidate_col = "reaction_id", "protein_id"
    elif direction == "e2r":
        query_col, candidate_col = "protein_id", "reaction_id"
    else:
        raise ValueError(direction)
    return evaluate_ranking_frame(
        local,
        query_col=query_col,
        candidate_col=candidate_col,
        score_col="score",
        label_col="label",
    )


def delta_metrics(
    base: dict[str, object],
    corrected: dict[str, object],
) -> dict[str, float]:
    keys = (
        "mrr",
        "map",
        "macro_roc_auc",
        "ndcg_at_10",
        "ndcg_at_20",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_20",
        "hit_at_50",
    )
    out = {}
    for key in keys:
        if base.get(key) is None or corrected.get(key) is None:
            continue
        out[key] = float(corrected[key]) - float(base[key])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Fit only an identity-preserving bilinear catalytic interaction correction "
            "on frozen strict-double-cold training pairs and evaluate on the exact held-out "
            "candidate rows already used by the cleanroom development protocol."
        )
    )
    ap.add_argument("--base-root", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--max-frobenius-norm", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--learning-rate", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--reaction-loss-weight", type=float, default=0.98)
    ap.add_argument("--score-preservation-weight", type=float, default=0.05)
    ap.add_argument("--feature-chunk-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    if not folds:
        raise ValueError("at least one fold is required")
    base_root = args.base_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    device = torch.device(args.device)

    first_summary = json.loads(
        (base_root / f"fold{folds[0]}" / "summary.json").read_text()
    )
    protein_dir = Path(first_summary["protein_feature_dir"]).resolve()
    reaction_dir = Path(first_summary["reaction_feature_dir"]).resolve()
    protein_features, all_protein_ids = load_protein_library(protein_dir)
    reaction_schema = json.loads(
        (reaction_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    reaction_features, all_reaction_ids = load_registered_reaction_feature_library(
        reaction_dir, reaction_schema
    )
    all_pindex = {value: i for i, value in enumerate(all_protein_ids)}
    all_rindex = {value: i for i, value in enumerate(all_reaction_ids)}

    fold_records: list[dict[str, object]] = []
    pooled: dict[str, list[pd.DataFrame]] = {
        "r2e_base": [],
        "r2e_corrected": [],
        "e2r_base": [],
        "e2r_corrected": [],
    }

    for fold in folds:
        fold_root = base_root / f"fold{fold}"
        summary = json.loads((fold_root / "summary.json").read_text())
        if Path(summary["protein_feature_dir"]).resolve() != protein_dir:
            raise ValueError("protein feature directory differs across folds")
        if Path(summary["reaction_feature_dir"]).resolve() != reaction_dir:
            raise ValueError("reaction feature directory differs across folds")
        train = pd.read_csv(fold_root / "training_pairs.csv", dtype=str).fillna("")
        train = train[["protein_id", "reaction_id"]].drop_duplicates().reset_index(drop=True)
        r2e = pd.read_csv(fold_root / "dev_pair_scores.csv", dtype=str).fillna("")
        e2r = pd.read_csv(fold_root / "dev_pair_scores_e2r.csv", dtype=str).fillna("")
        for frame in (r2e, e2r):
            frame["label"] = pd.to_numeric(frame["label"]).astype(int)
            frame["score"] = pd.to_numeric(frame["score"]).astype(float)

        checkpoint = next((fold_root / "models").glob("production_seed*.pt"))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = ModelConfig(**payload["model_config"])
        model = TerpeneDualTower(config).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        train_reactions = sorted(train.reaction_id.unique())
        train_enzymes = sorted(train.protein_id.unique())
        eval_reactions = sorted(set(r2e.reaction_id) | set(e2r.reaction_id))
        eval_enzymes = sorted(set(r2e.protein_id) | set(e2r.protein_id))
        needed_reactions = sorted(set(train_reactions) | set(eval_reactions))
        needed_enzymes = sorted(set(train_enzymes) | set(eval_enzymes))
        missing_r = [x for x in needed_reactions if x not in all_rindex]
        missing_e = [x for x in needed_enzymes if x not in all_pindex]
        if missing_r or missing_e:
            raise ValueError(
                f"fold{fold} feature coverage gap: reactions={missing_r[:5]}, proteins={missing_e[:5]}"
            )

        encoded_reactions = encode_selected(
            model,
            reaction_features,
            np.asarray([all_rindex[x] for x in needed_reactions], dtype=np.int64),
            side="reaction",
            device=device,
            chunk_size=args.feature_chunk_size,
        )
        encoded_enzymes = encode_selected(
            model,
            protein_features,
            np.asarray([all_pindex[x] for x in needed_enzymes], dtype=np.int64),
            side="protein",
            device=device,
            chunk_size=args.feature_chunk_size,
        )
        local_rindex = {value: i for i, value in enumerate(needed_reactions)}
        local_eindex = {value: i for i, value in enumerate(needed_enzymes)}
        train_r_embeddings = encoded_reactions[
            np.asarray([local_rindex[x] for x in train_reactions], dtype=np.int64)
        ]
        train_e_embeddings = encoded_enzymes[
            np.asarray([local_eindex[x] for x in train_enzymes], dtype=np.int64)
        ]

        interaction, history = train_interaction(
            train_r_embeddings,
            train_e_embeddings,
            train,
            max_frobenius_norm=args.max_frobenius_norm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            temperature=args.temperature,
            reaction_loss_weight=args.reaction_loss_weight,
            score_preservation_weight=args.score_preservation_weight,
            seed=args.seed + fold,
            device=device,
        )
        model_path = output / "models" / f"fold{fold}.pt"
        torch.save(
            {
                "schema": "fibre-catalytic-interaction-residual-v1",
                "fold": fold,
                "base_checkpoint": str(checkpoint),
                "base_checkpoint_sha256": sha256_file(checkpoint),
                "dimension": int(config.embedding_dim),
                "max_frobenius_norm": float(args.max_frobenius_norm),
                "interaction_state_dict": {
                    key: value.detach().cpu()
                    for key, value in interaction.state_dict().items()
                },
                "history": history,
                "training_pair_count": int(len(train)),
                "training_pairs_sha256": sha256_file(fold_root / "training_pairs.csv"),
                "target_benchmark_labels_read_for_training": False,
                "target_benchmark_metadata_used_for_training": False,
            },
            model_path,
        )

        fold_result: dict[str, object] = {
            "fold": fold,
            "base_checkpoint": str(checkpoint),
            "base_checkpoint_sha256": sha256_file(checkpoint),
            "training_pairs": int(len(train)),
            "interaction": interaction.diagnostics(),
            "history": history,
            "directions": {},
        }
        for direction, frame in (("r2e", r2e), ("e2r", e2r)):
            base_recomputed, corrected = score_rows(
                frame,
                reaction_ids=needed_reactions,
                enzyme_ids=needed_enzymes,
                reaction_embeddings=encoded_reactions,
                enzyme_embeddings=encoded_enzymes,
                layer=interaction,
                device=device,
            )
            recorded = frame["score"].to_numpy(dtype=np.float64)
            parity = float(np.max(np.abs(base_recomputed.astype(np.float64) - recorded)))
            base_q, base_metrics = ranking_metrics(frame, recorded, direction=direction)
            corrected_q, corrected_metrics = ranking_metrics(
                frame, corrected, direction=direction
            )
            score_shift = corrected.astype(np.float64) - base_recomputed.astype(np.float64)
            fold_result["directions"][direction] = {
                "recorded_base_recompute_max_abs": parity,
                "score_shift_max_abs": float(np.max(np.abs(score_shift))),
                "score_shift_mean_abs": float(np.mean(np.abs(score_shift))),
                "base": base_metrics,
                "corrected": corrected_metrics,
                "delta": delta_metrics(base_metrics, corrected_metrics),
            }
            tag = f"fold{fold}:"
            base_frame = frame.copy()
            corrected_frame = frame.copy()
            if direction == "r2e":
                base_frame["query_id"] = tag + base_frame.reaction_id.astype(str)
                base_frame["candidate_id"] = base_frame.protein_id.astype(str)
                corrected_frame["query_id"] = tag + corrected_frame.reaction_id.astype(str)
                corrected_frame["candidate_id"] = corrected_frame.protein_id.astype(str)
            else:
                base_frame["query_id"] = tag + base_frame.protein_id.astype(str)
                base_frame["candidate_id"] = base_frame.reaction_id.astype(str)
                corrected_frame["query_id"] = tag + corrected_frame.protein_id.astype(str)
                corrected_frame["candidate_id"] = corrected_frame.reaction_id.astype(str)
            base_frame["score"] = recorded
            corrected_frame["score"] = corrected
            pooled[f"{direction}_base"].append(
                base_frame[["query_id", "candidate_id", "score", "label"]]
            )
            pooled[f"{direction}_corrected"].append(
                corrected_frame[["query_id", "candidate_id", "score", "label"]]
            )
            base_q.to_csv(output / f"fold{fold}_{direction}_base_query_metrics.csv", index=False)
            corrected_q.to_csv(
                output / f"fold{fold}_{direction}_corrected_query_metrics.csv", index=False
            )
        fold_records.append(fold_result)
        (output / f"fold{fold}.json").write_text(
            json.dumps(fold_result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "stage": "fold_done",
                    "fold": fold,
                    "r2e_delta_mrr": fold_result["directions"]["r2e"]["delta"]["mrr"],
                    "e2r_delta_mrr": fold_result["directions"]["e2r"]["delta"]["mrr"],
                }
            ),
            flush=True,
        )

        del model, interaction, encoded_reactions, encoded_enzymes
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_summary: dict[str, object] = {}
    for direction in ("r2e", "e2r"):
        base_frame = pd.concat(pooled[f"{direction}_base"], ignore_index=True)
        corrected_frame = pd.concat(pooled[f"{direction}_corrected"], ignore_index=True)
        _, base_metrics = evaluate_ranking_frame(base_frame)
        _, corrected_metrics = evaluate_ranking_frame(corrected_frame)
        pooled_summary[direction] = {
            "base": base_metrics,
            "corrected": corrected_metrics,
            "delta": delta_metrics(base_metrics, corrected_metrics),
        }

    summary = {
        "schema": "fibre-catalytic-interaction-residual-development-v1",
        "scope": "fixed clean2023 strict protein+reaction double-cold development folds only",
        "folds": folds,
        "base_root": str(base_root),
        "training": {
            "only_parameter_family": "bounded bilinear interaction B",
            "initialization": "B=0, exact base score identity",
            "max_frobenius_norm": float(args.max_frobenius_norm),
            "operator_norm_upper_bound": float(args.max_frobenius_norm),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "temperature": float(args.temperature),
            "reaction_loss_weight": float(args.reaction_loss_weight),
            "score_preservation_weight": float(args.score_preservation_weight),
            "seed": int(args.seed),
            "base_encoders_frozen": True,
            "application_data_used": False,
            "heldout_labels_used_for_training": False,
        },
        "universal_coverage_contract": {
            "candidate_filter_added": False,
            "abstention_added": False,
            "new_input_requirement_added": False,
            "base_encoder_domain_changed": False,
            "fallback": "B=0 recovers exact frozen base pairing",
        },
        "fold_results": fold_records,
        "pooled": pooled_summary,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"stage": "complete", "pooled": pooled_summary}, indent=2))


if __name__ == "__main__":
    main()
