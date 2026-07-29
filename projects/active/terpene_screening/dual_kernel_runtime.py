from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from projects.active.terpene_screening.evaluate_dual_kernel_collaborative_retrieval import (
    normalized_adjacency,
    normalize_rows,
    topk_affinity,
)
from projects.active.terpene_screening.evaluate_zero_shot_retrieval_cold import (
    build_reaction_similarity_matrix,
)


LOCKED_REACTION_K = 50
LOCKED_PROTEIN_K = 5
LOCKED_TEMPERATURE = 0.03
LOCKED_DEGREE_POWER = 1.0


@dataclass(frozen=True)
class DualKernelAssets:
    root: Path
    reaction_ids: tuple[str, ...]
    protein_ids: tuple[str, ...]
    train_protein_rows: np.ndarray
    protein_features: np.ndarray
    reaction_protein_support: csr_matrix
    metadata: dict[str, object]

    @property
    def reaction_to_row(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.reaction_ids)}

    @property
    def protein_to_row(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.protein_ids)}


def _validate_unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        duplicates = pd.Series(values)[pd.Series(values).duplicated()].unique().tolist()
        raise ValueError(f"Duplicate {label} identifiers: {duplicates[:10]}")


def build_assets(
    *,
    reaction_registry: pd.DataFrame,
    protein_ids: list[str],
    protein_features: np.ndarray,
    training_pairs: pd.DataFrame,
    output_dir: Path,
    reaction_k: int = LOCKED_REACTION_K,
    protein_k: int = LOCKED_PROTEIN_K,
    temperature: float = LOCKED_TEMPERATURE,
    degree_power: float = LOCKED_DEGREE_POWER,
) -> dict[str, object]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reaction_registry = reaction_registry.copy().fillna("")
    required_reaction_columns = {"reaction_id", "reaction_smiles"}
    if not required_reaction_columns.issubset(reaction_registry.columns):
        raise ValueError(
            f"Reaction registry must contain {sorted(required_reaction_columns)}"
        )
    reaction_ids = reaction_registry["reaction_id"].astype(str).tolist()
    protein_ids = [str(value) for value in protein_ids]
    _validate_unique(reaction_ids, "reaction")
    _validate_unique(protein_ids, "protein")
    if len(protein_features) != len(protein_ids):
        raise ValueError("Protein feature rows do not match protein identifiers")
    protein_features = normalize_rows(np.asarray(protein_features, dtype=np.float32))

    pairs = training_pairs[["Entry", "rhea_id"]].astype(str).drop_duplicates().copy()
    reaction_set = set(reaction_ids)
    protein_set = set(protein_ids)
    missing_reactions = sorted(set(pairs.rhea_id) - reaction_set)
    missing_proteins = sorted(set(pairs.Entry) - protein_set)
    if missing_reactions or missing_proteins:
        raise ValueError(
            "Training-pair registries are incomplete: "
            f"missing_reactions={missing_reactions[:10]}, "
            f"missing_proteins={missing_proteins[:10]}"
        )

    reaction_to_row = {value: index for index, value in enumerate(reaction_ids)}
    protein_to_row = {value: index for index, value in enumerate(protein_ids)}
    train_reaction_rows = np.asarray(
        sorted({reaction_to_row[value] for value in pairs.rhea_id}), dtype=np.int64
    )
    train_protein_rows = np.asarray(
        sorted({protein_to_row[value] for value in pairs.Entry}), dtype=np.int64
    )

    similarity_input = reaction_registry[["reaction_id", "reaction_smiles"]].rename(
        columns={"reaction_id": "rhea_id", "reaction_smiles": "smiles_seq"}
    )
    reaction_similarity, _ = build_reaction_similarity_matrix(similarity_input)
    reaction_affinity = topk_affinity(
        reaction_similarity,
        train_reaction_rows,
        reaction_k,
        temperature,
    )
    adjacency = normalized_adjacency(
        pairs,
        reaction_to_row,
        protein_to_row,
        (len(reaction_ids), len(protein_ids)),
        degree_power,
    ).tocsr()
    reaction_protein_support = (reaction_affinity @ adjacency).tocsr()

    pd.DataFrame({"reaction_id": reaction_ids}).to_csv(
        output_dir / "reaction_ids.csv", index=False
    )
    pd.DataFrame(
        {
            "protein_id": protein_ids,
            "is_training_protein": [
                int(index in set(train_protein_rows.tolist()))
                for index in range(len(protein_ids))
            ],
        }
    ).to_csv(output_dir / "protein_ids.csv", index=False)
    np.save(output_dir / "protein_features.npy", protein_features)
    np.save(output_dir / "train_protein_rows.npy", train_protein_rows)
    save_npz(output_dir / "reaction_protein_support.npz", reaction_protein_support)
    np.save(output_dir / "reaction_similarity.npy", reaction_similarity.astype(np.float32))
    save_npz(output_dir / "reaction_affinity.npz", reaction_affinity)
    save_npz(output_dir / "normalized_adjacency.npz", adjacency)

    support_row_sums = np.asarray(reaction_protein_support.sum(axis=1)).reshape(-1)
    metadata: dict[str, object] = {
        "method": "locked_e2r_dual_kernel_collaborative_support",
        "reaction_k": reaction_k,
        "protein_k": protein_k,
        "temperature": temperature,
        "degree_power": degree_power,
        "n_reactions": len(reaction_ids),
        "n_proteins": len(protein_ids),
        "n_training_pairs": len(pairs),
        "n_train_reactions": len(train_reaction_rows),
        "n_train_proteins": len(train_protein_rows),
        "support_shape": list(reaction_protein_support.shape),
        "support_nnz": int(reaction_protein_support.nnz),
        "support_zero_rows": int(np.sum(support_row_sums == 0)),
        "query_self_exclusion": True,
        "outputs": {
            "reaction_ids": str(output_dir / "reaction_ids.csv"),
            "protein_ids": str(output_dir / "protein_ids.csv"),
            "protein_features": str(output_dir / "protein_features.npy"),
            "train_protein_rows": str(output_dir / "train_protein_rows.npy"),
            "reaction_protein_support": str(
                output_dir / "reaction_protein_support.npz"
            ),
            "reaction_similarity": str(output_dir / "reaction_similarity.npy"),
            "reaction_affinity": str(output_dir / "reaction_affinity.npz"),
            "normalized_adjacency": str(output_dir / "normalized_adjacency.npz"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_assets(path: Path) -> DualKernelAssets:
    root = path.resolve()
    metadata = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    reaction_ids = tuple(
        pd.read_csv(root / "reaction_ids.csv", dtype=str)
        .fillna("")["reaction_id"]
        .astype(str)
    )
    protein_ids = tuple(
        pd.read_csv(root / "protein_ids.csv", dtype=str)
        .fillna("")["protein_id"]
        .astype(str)
    )
    protein_features = np.load(root / "protein_features.npy").astype(np.float32)
    train_protein_rows = np.load(root / "train_protein_rows.npy").astype(np.int64)
    support = load_npz(root / "reaction_protein_support.npz").tocsr().astype(
        np.float32
    )
    if protein_features.ndim != 2 or protein_features.shape[0] != len(protein_ids):
        raise ValueError("Dual-kernel protein asset shape mismatch")
    if int(metadata["n_proteins"]) != len(protein_ids):
        raise ValueError("Dual-kernel protein metadata count mismatch")
    if support.shape != (len(reaction_ids), len(protein_ids)):
        raise ValueError("Dual-kernel support asset shape mismatch")
    if np.any(train_protein_rows < 0) or np.any(train_protein_rows >= len(protein_ids)):
        raise ValueError("Dual-kernel training protein rows are out of bounds")
    return DualKernelAssets(
        root=root,
        reaction_ids=reaction_ids,
        protein_ids=protein_ids,
        train_protein_rows=train_protein_rows,
        protein_features=normalize_rows(protein_features),
        reaction_protein_support=support,
        metadata=metadata,
    )


def protein_affinity(
    query_feature: np.ndarray,
    assets: DualKernelAssets,
    *,
    query_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature = normalize_rows(np.asarray(query_feature, dtype=np.float32).reshape(1, -1))[0]
    if feature.shape[0] != assets.protein_features.shape[1]:
        raise ValueError("Dual-kernel query feature dimension mismatch")
    allowed = assets.train_protein_rows
    if query_id:
        query_row = assets.protein_to_row.get(str(query_id))
        if query_row is not None:
            allowed = allowed[allowed != query_row]
    if not len(allowed):
        raise ValueError("No training proteins remain for dual-kernel retrieval")
    similarities = assets.protein_features[allowed] @ feature
    k = min(int(assets.metadata["protein_k"]), len(allowed))
    selected_local = np.argpartition(similarities, -k)[-k:]
    selected_local = selected_local[
        np.argsort(-similarities[selected_local], kind="stable")
    ]
    selected_rows = allowed[selected_local]
    selected_scores = similarities[selected_local].astype(np.float64)
    temperature = float(assets.metadata["temperature"])
    weights = np.exp((selected_scores - selected_scores.max()) / temperature)
    denominator = weights.sum()
    if denominator <= 0 or not np.isfinite(denominator):
        weights = np.ones(len(selected_rows), dtype=np.float64) / len(selected_rows)
    else:
        weights /= denominator
    return selected_rows.astype(np.int64), weights.astype(np.float32)


def score_query(
    query_feature: np.ndarray,
    assets: DualKernelAssets,
    *,
    query_id: str | None = None,
) -> np.ndarray:
    selected_rows, weights = protein_affinity(
        query_feature, assets, query_id=query_id
    )
    scores = assets.reaction_protein_support[:, selected_rows] @ weights
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def align_reaction_scores(
    scores: np.ndarray,
    assets: DualKernelAssets,
    reaction_ids: list[str] | tuple[str, ...],
) -> np.ndarray:
    reaction_ids = tuple(str(value) for value in reaction_ids)
    if len(reaction_ids) != len(assets.reaction_ids) or set(reaction_ids) != set(
        assets.reaction_ids
    ):
        raise ValueError("Dual-kernel reaction registry set mismatch")
    order = np.asarray(
        [assets.reaction_to_row[value] for value in reaction_ids], dtype=np.int64
    )
    values = np.asarray(scores)
    if values.shape[-1] != len(assets.reaction_ids):
        raise ValueError("Dual-kernel score width does not match asset reactions")
    return values[..., order]


def score_batch(
    query_features: np.ndarray,
    assets: DualKernelAssets,
    *,
    query_ids: list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    features = np.asarray(query_features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("Dual-kernel batch features must be two-dimensional")
    if query_ids is None:
        query_ids = [""] * len(features)
    if len(query_ids) != len(features):
        raise ValueError("Dual-kernel batch query IDs do not match feature rows")
    return np.stack(
        [
            score_query(feature, assets, query_id=str(query_id) or None)
            for feature, query_id in zip(features, query_ids)
        ]
    ).astype(np.float32)
