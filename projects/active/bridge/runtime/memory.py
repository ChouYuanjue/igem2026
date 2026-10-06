from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch


EPISODIC_FEATURE_NAMES = (
    "log_support_count",
    "effective_support_count",
    "support_query_mean",
    "support_query_max",
    "support_query_std",
    "support_coherence",
    "support_seen_entity_fraction",
    "base_margin_1_2",
    "base_top10_spread",
    "memory_margin_1_2",
    "base_memory_corr",
    "top10_overlap",
    "top100_overlap",
    "train_memory_weight",
)


@dataclass(frozen=True)
class EpisodicMemoryResult:
    score: np.ndarray
    features: np.ndarray
    effective_support_ids: tuple[str, ...]
    ignored_training_ids: tuple[str, ...]
    missing_support_ids: tuple[str, ...]
    attention: np.ndarray
    effective_support_count: float
    support_coherence: float


def _top_rows(values: np.ndarray, k: int) -> np.ndarray:
    kk = min(max(int(k), 0), len(values))
    if kk == 0:
        return np.empty(0, dtype=np.int64)
    rows = np.argpartition(-values, kk - 1)[:kk]
    return rows[np.argsort(-values[rows], kind="stable")]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _margin(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    pair = np.partition(values, -2)[-2:]
    return float(pair.max() - pair.min())


def _standardize(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return (x - float(x.mean())) / max(float(x.std()), 1e-8)


def _attention_weights(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Parameter-free query-to-support attention.

    Confirmed supports are all positives.  We therefore only use the frozen Broad
    geometry to decide which supports are most relevant to this query.  Centering
    and scaling the logits makes the rule query-local and avoids another tuned
    temperature hyperparameter.
    """
    logits = support @ query
    if len(logits) == 1:
        return torch.ones_like(logits)
    logits = (logits - logits.mean()) / logits.std(unbiased=False).clamp_min(1e-6)
    return torch.softmax(logits, dim=0)


def _effective_count(attention: np.ndarray) -> float:
    p = np.asarray(attention, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    entropy = -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
    return float(np.exp(entropy))


def _coherence(support: torch.Tensor, attention: torch.Tensor) -> float:
    if len(support) == 0:
        return 0.0
    mean = (attention[:, None] * support).sum(0)
    return float(mean.norm().item())


def build_episodic_memory(
    *,
    query_embedding: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_ids: list[str],
    candidate_index: dict[str, int],
    requested_support_ids: Iterable[str],
    training_positive_ids: set[str],
    entity_seen_mask: np.ndarray,
    base_score: np.ndarray,
    train_memory_weight: float,
    external_support_embeddings: dict[str, torch.Tensor] | None = None,
) -> EpisodicMemoryResult | None:
    """Build runtime-only episodic memory from relations unseen by model training.

    Training positives are intentionally removed: they belong to the long-term
    graph memory and must not vote a second time as runtime evidence.
    """
    requested = tuple(dict.fromkeys(str(x) for x in requested_support_ids if str(x)))
    ignored_training = tuple(x for x in requested if x in training_positive_ids)
    episodic = tuple(x for x in requested if x not in training_positive_ids)
    external_support_embeddings = dict(external_support_embeddings or {})
    effective = tuple(
        x for x in episodic
        if x in candidate_index or x in external_support_embeddings
    )
    missing = tuple(
        x for x in episodic
        if x not in candidate_index and x not in external_support_embeddings
    )
    if not effective:
        return None

    support_rows: list[torch.Tensor] = []
    seen_flags: list[bool] = []
    seen_array = np.asarray(entity_seen_mask, dtype=bool)
    for value in effective:
        row = candidate_index.get(value)
        if row is not None:
            support_rows.append(candidate_embeddings[int(row)])
            seen_flags.append(bool(seen_array[int(row)]))
            continue
        external = external_support_embeddings[value].to(
            device=candidate_embeddings.device,
            dtype=candidate_embeddings.dtype,
        )
        if external.ndim != 1 or external.shape[0] != candidate_embeddings.shape[1]:
            raise ValueError(
                f"external episodic support {value} has incompatible embedding shape "
                f"{tuple(external.shape)}"
            )
        support_rows.append(external)
        seen_flags.append(False)
    support = torch.stack(support_rows, dim=0)
    attention_t = _attention_weights(query_embedding, support)
    memory = (attention_t[:, None] * support).sum(0)
    raw = (candidate_embeddings @ memory).float().detach().cpu().numpy().astype(np.float64)
    score = _standardize(raw)

    att = attention_t.detach().cpu().numpy().astype(np.float64)
    eff_n = _effective_count(att)
    coherence = _coherence(support, attention_t)

    qsim = (support @ query_embedding).float().detach().cpu().numpy().astype(np.float64)
    base = np.asarray(base_score, dtype=np.float64)
    base100 = _top_rows(base, 100)
    base10 = base100[: min(10, len(base100))]
    memory100 = _top_rows(score, 100)
    memory10 = memory100[: min(10, len(memory100))]
    base_values = np.sort(base[base100])[::-1]
    memory_values = np.sort(score[memory100])[::-1]
    seen_fraction = float(np.mean(np.asarray(seen_flags, dtype=np.float64)))

    features = np.asarray(
        [
            math.log1p(len(effective)),
            eff_n,
            float(qsim.mean()),
            float(qsim.max()),
            float(qsim.std()),
            coherence,
            seen_fraction,
            _margin(base_values),
            float(base_values[0] - base_values[min(9, len(base_values) - 1)]),
            _margin(memory_values),
            _corr(base, score),
            float(len(set(base10.tolist()) & set(memory10.tolist())) / max(1, len(base10))),
            float(len(set(base100.tolist()) & set(memory100.tolist())) / max(1, len(base100))),
            float(train_memory_weight),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(EPISODIC_FEATURE_NAMES),):
        raise AssertionError((features.shape, len(EPISODIC_FEATURE_NAMES)))

    return EpisodicMemoryResult(
        score=score,
        features=features,
        effective_support_ids=effective,
        ignored_training_ids=ignored_training,
        missing_support_ids=missing,
        attention=att,
        effective_support_count=eff_n,
        support_coherence=coherence,
    )


def episodic_authority(
    gate: dict[str, Any],
    direction: str,
    features: np.ndarray,
) -> tuple[float, float]:
    """Return the query-specific trust in runtime episodic memory.

    The gate is trained only on firewalled validation episodes.  Its output has
    an intrinsic scale: 0 preserves the frozen BRIDGE prior; 1 replaces only
    the Broad query component with the support-memory component.
    """
    asset = gate["directions"][direction]
    if tuple(asset["feature_names"]) != EPISODIC_FEATURE_NAMES:
        raise RuntimeError("episodic memory gate feature contract mismatch")
    x = asset["scaler"].transform(np.asarray(features, dtype=np.float64)[None, :])
    raw = float(asset["gate_model"].predict(x)[0])
    return float(np.clip(raw, 0.0, 1.0)), raw
