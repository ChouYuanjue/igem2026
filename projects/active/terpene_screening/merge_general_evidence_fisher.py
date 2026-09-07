from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.terpene_screening.blend_general_evidence_models import ASSET_FILES
from projects.active.terpene_screening.rank_open_world import (
    load_feature_schema,
    load_protein_library,
    load_reaction_library,
    load_registered_reaction_feature_library,
)
from projects.active.terpene_screening.third_party.fusionbench_fisher import (
    get_param_squared_gradients,
    merging_with_fisher_weights,
)
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig, TerpeneDualTower
from projects.active.terpene_screening.train_general_evidence_retriever import (
    _directional_full_candidate_loss,
    _encode_chunks,
    _query_positive_rows,
)

DEFAULT_SOURCE_PROTEINS = ROOT / "data/terpene_embeddings/esmc600m_mean"
DEFAULT_SOURCE_POSITIVES = ROOT / "data/terpene/enzyme_terpene_synthase.tsv"
DEFAULT_BROAD_UNIVERSE = ROOT / "data/catalyst_candidate_universes/general_merged"


def _load_one_model(model_dir: Path, device: torch.device) -> tuple[TerpeneDualTower, dict[str, object], Path]:
    checkpoints = sorted((model_dir / "models").glob("production_seed*.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"Fisher screening expects exactly one checkpoint under {model_dir}; found {len(checkpoints)}")
    checkpoint = checkpoints[0]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"])
    model = TerpeneDualTower(config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, checkpoint


def _sample_queries(query_ids: list[str], max_queries: int, seed: int) -> list[str]:
    ids = sorted(set(query_ids))
    if max_queries <= 0 or len(ids) <= max_queries:
        return ids
    rng = random.Random(seed)
    return sorted(rng.sample(ids, int(max_queries)))


def _directional_fisher(
    model: TerpeneDualTower,
    *,
    direction: str,
    query_features: np.ndarray,
    query_ids: list[str],
    candidate_features: np.ndarray,
    candidate_ids: list[str],
    associations: pd.DataFrame,
    parameter_prefix: str,
    max_queries: int,
    sample_seed: int,
    batch_size: int,
    feature_chunk_size: int,
    temperature: float,
    topk_k: int,
    topk_weight: float,
    all_positive_weight: float,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    query_index = {value: index for index, value in enumerate(query_ids)}
    candidate_index = {value: index for index, value in enumerate(candidate_ids)}
    valid_query_ids, positives = _query_positive_rows(
        associations,
        direction=direction,
        query_index=query_index,
        candidate_index=candidate_index,
    )
    positive_by_query = dict(zip(valid_query_ids, positives, strict=True))
    selected = _sample_queries(valid_query_ids, max_queries, sample_seed)
    if not selected:
        raise ValueError("No Fisher queries remain after association/universe filtering")

    candidate_kind = "protein" if direction == "r2e" else "reaction"
    candidate_embeddings = _encode_chunks(
        model,
        candidate_features,
        kind=candidate_kind,
        device=device,
        chunk_size=feature_chunk_size,
    ).detach()
    parameter_names = [
        name for name, parameter in model.named_parameters()
        if name.startswith(parameter_prefix) and parameter.requires_grad
    ]
    if not parameter_names:
        # Fisher merging does not require params to have been marked trainable by a prior trainer.
        parameter_names = [name for name, _ in model.named_parameters() if name.startswith(parameter_prefix)]
    fisher = {
        name: torch.zeros_like(dict(model.named_parameters())[name], device="cpu")
        for name in parameter_names
    }

    total_examples = 0
    model.eval()
    for start in range(0, len(selected), batch_size):
        batch_ids = selected[start : start + batch_size]
        rows = np.asarray([query_index[value] for value in batch_ids], dtype=np.int64)
        batch = torch.as_tensor(query_features[rows], dtype=torch.float32, device=device)
        query_embeddings = model.encode_reactions(batch) if direction == "r2e" else model.encode_proteins(batch)
        batch_positives = [positive_by_query[value] for value in batch_ids]
        loss, _ = _directional_full_candidate_loss(
            query_embeddings,
            candidate_embeddings,
            batch_positives,
            temperature=temperature,
            topk_k=topk_k,
            topk_weight=topk_weight,
            topk_margin=0.0,
            all_positive_weight=all_positive_weight,
        )
        model.zero_grad(set_to_none=True)
        loss.backward()
        squared = get_param_squared_gradients(model, parameter_names)
        n = len(batch_ids)
        for name, value in squared.items():
            fisher[name] += value.detach().cpu() * n
        total_examples += n

    if total_examples <= 0:
        raise ValueError("Fisher accumulation saw zero examples")
    for name in fisher:
        fisher[name] /= float(total_examples)
    diagnostics = {
        "direction": direction,
        "parameter_prefix": parameter_prefix,
        "n_queries": len(selected),
        "sample_seed": int(sample_seed),
        "mean_fisher": {name: float(value.mean()) for name, value in fisher.items()},
        "max_fisher": {name: float(value.max()) for name, value in fisher.items()},
    }
    return fisher, diagnostics


def _source_r2e_data(source_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray, list[str], pd.DataFrame]:
    protein_features, protein_ids = load_protein_library(DEFAULT_SOURCE_PROTEINS)
    schema = load_feature_schema(source_dir)
    reaction_features, reaction_ids = load_reaction_library(source_dir, schema)
    positives = pd.read_csv(DEFAULT_SOURCE_POSITIVES, sep="\t", dtype=str).fillna("")
    associations = positives[["Entry", "rhea_id"]].rename(
        columns={"Entry": "protein_id", "rhea_id": "reaction_id"}
    ).drop_duplicates()
    return reaction_features, reaction_ids, protein_features, protein_ids, associations


def _target_r2e_data(
    source_dir: Path,
    universe_dir: Path,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], pd.DataFrame]:
    protein_features, protein_ids = load_protein_library(universe_dir / "proteins")
    schema = load_feature_schema(source_dir)
    reaction_features, reaction_ids = load_registered_reaction_feature_library(
        universe_dir / "reaction_features" / "drfp_categorical_v1",
        schema,
    )
    associations = pd.read_csv(universe_dir / "associations.csv", dtype=str).fillna("")
    associations = associations[["protein_id", "reaction_id"]].drop_duplicates()
    historical = set(pd.read_csv(source_dir / "reaction_registry.csv", dtype=str).fillna("")["reaction_id"].astype(str))
    associations = associations[~associations["reaction_id"].isin(historical)].copy()
    return reaction_features, reaction_ids, protein_features, protein_ids, associations


def _scale_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse a broad R2E continuation back into production with FusionBench Fisher weights.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--adapted-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--universe-dir", type=Path, default=DEFAULT_BROAD_UNIVERSE)
    parser.add_argument("--source-scales", default="1,2,5,10")
    parser.add_argument("--source-max-queries", type=int, default=0, help="0 uses all current R2E reaction queries")
    parser.add_argument("--target-max-queries", type=int, default=1024)
    parser.add_argument("--sample-seed", type=int, default=20260723)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-chunk-size", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--topk-k", type=int, default=10)
    parser.add_argument("--topk-weight", type=float, default=0.10)
    parser.add_argument("--all-positive-weight", type=float, default=0.05)
    parser.add_argument("--minimal-fisher-weight", type=float, default=1e-6)
    parser.add_argument("--no-normalize-fisher", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    adapted_dir = args.adapted_dir.resolve()
    output_root = args.output_root.resolve()
    universe_dir = args.universe_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    source_model, source_payload, source_checkpoint = _load_one_model(source_dir, device)
    adapted_model, adapted_payload, adapted_checkpoint = _load_one_model(adapted_dir, device)
    if source_payload["model_config"] != adapted_payload["model_config"]:
        raise ValueError("Source and adapted model configs differ")
    if source_checkpoint.name != adapted_checkpoint.name:
        raise ValueError("Source and adapted checkpoint seeds differ")

    s_query, s_query_ids, s_candidates, s_candidate_ids, s_assoc = _source_r2e_data(source_dir)
    source_fisher, source_diag = _directional_fisher(
        source_model,
        direction="r2e",
        query_features=s_query,
        query_ids=s_query_ids,
        candidate_features=s_candidates,
        candidate_ids=s_candidate_ids,
        associations=s_assoc,
        parameter_prefix="reaction_tower.",
        max_queries=args.source_max_queries,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        feature_chunk_size=args.feature_chunk_size,
        temperature=args.temperature,
        topk_k=args.topk_k,
        topk_weight=args.topk_weight,
        all_positive_weight=args.all_positive_weight,
        device=device,
    )
    del s_query, s_candidates
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t_query, t_query_ids, t_candidates, t_candidate_ids, t_assoc = _target_r2e_data(source_dir, universe_dir)
    target_fisher, target_diag = _directional_fisher(
        adapted_model,
        direction="r2e",
        query_features=t_query,
        query_ids=t_query_ids,
        candidate_features=t_candidates,
        candidate_ids=t_candidate_ids,
        associations=t_assoc,
        parameter_prefix="reaction_tower.",
        max_queries=args.target_max_queries,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        feature_chunk_size=args.feature_chunk_size,
        temperature=args.temperature,
        topk_k=args.topk_k,
        topk_weight=args.topk_weight,
        all_positive_weight=args.all_positive_weight,
        device=device,
    )

    source_state = {name: value.detach().cpu() for name, value in source_model.state_dict().items()}
    adapted_state = {name: value.detach().cpu() for name, value in adapted_model.state_dict().items()}
    parameter_names = sorted(source_fisher)
    models_to_merge_param_dict = {
        name: [source_state[name], adapted_state[name]] for name in parameter_names
    }
    fisher_list = [source_fisher, target_fisher]
    source_scales = [float(value) for value in args.source_scales.split(",") if value.strip()]
    outputs = []
    for source_scale in source_scales:
        merged = merging_with_fisher_weights(
            models_to_merge_param_dict=models_to_merge_param_dict,
            models_to_merge_fisher_weights_list=fisher_list,
            fisher_scaling_coefficients=torch.tensor([source_scale, 1.0], dtype=torch.float32),
            normalize_fisher_weight=not args.no_normalize_fisher,
            minimal_fisher_weight=args.minimal_fisher_weight,
        )
        state = {name: value.clone() for name, value in source_state.items()}
        for name, value in merged.items():
            state[name] = value.to(dtype=state[name].dtype)
        tag = _scale_tag(source_scale)
        output_dir = output_root / f"source_{tag}"
        (output_dir / "models").mkdir(parents=True, exist_ok=True)
        payload = {key: value for key, value in source_payload.items() if key != "model_state_dict"}
        payload["model_state_dict"] = state
        payload["general_evidence_fisher_merge"] = {
            "source_checkpoint": str(source_checkpoint),
            "adapted_checkpoint": str(adapted_checkpoint),
            "source_scale": source_scale,
            "target_scale": 1.0,
            "normalize_fisher_weight": not args.no_normalize_fisher,
            "minimal_fisher_weight": args.minimal_fisher_weight,
            "upstream": "https://github.com/tanganke/fusion_bench",
            "upstream_commit": "54c9e8c9d9621620c720452cd8533332a32d3689",
        }
        target_checkpoint = output_dir / "models" / source_checkpoint.name
        torch.save(payload, target_checkpoint)
        for filename in ASSET_FILES:
            src = source_dir / filename
            if src.exists():
                shutil.copy2(src, output_dir / filename)
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "model_type": "general_evidence_fisher_consolidation",
                    "source_scale": source_scale,
                    "source_fisher": source_diag,
                    "target_fisher": target_diag,
                    "checkpoint": str(target_checkpoint),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        outputs.append(str(output_dir))

    summary = {
        "source_checkpoint": str(source_checkpoint),
        "adapted_checkpoint": str(adapted_checkpoint),
        "source_fisher": source_diag,
        "target_fisher": target_diag,
        "outputs": outputs,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
