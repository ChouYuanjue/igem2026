"""Frozen, query-conditioned E2R evidence authority.

The feature extractor sees only query scores, candidate availability, and the
clean2023 training graph. It does not accept ground-truth labels. Model fitting
uses independent Rhea relationships and is performed by the reproducibility
script. The candidate universe and all frozen evidence embeddings are unchanged.
"""

from __future__ import annotations

import numpy as np

CHANNELS = ("functional", "structure", "relation")
GRID = {
    "functional": (0.0, 0.5, 1.0),
    "structure": (0.0, 0.5, 1.0),
    "relation": (0.0, 0.35, 1.5, 4.0),
}
TOPK = 1000
MAX_AUTHORITY = (1.0, 1.0, 4.0)
LOCAL_FEATURE_NAMES = (
    "coverage",
    "support_top10",
    "std",
    "mean",
    "top1_margin",
    "top10_spread",
    "top10_overlap",
    "top100_overlap",
    "broad_corr",
)
RELATION_FEATURE_NAMES = (
    "log_query_degree",
    "query_neighbor_coherence",
    "context_coverage",
    "base_margin_1_2",
    "base_top10_spread",
    "base_top100_spread",
    "context_margin_1_2",
    "context_top10_mean",
    "context_top100_mean",
    "base_context_corr",
    "top10_overlap",
    "top100_overlap",
    "component_agreement",
    "context_top20_broad_logrank_mean",
)
FEATURE_NAMES = (
    RELATION_FEATURE_NAMES
    + tuple(f"{ch}_{name}" for ch in CHANNELS for name in LOCAL_FEATURE_NAMES)
)


def feature_vector(
    *,
    core: np.ndarray,
    functional: np.ndarray,
    structure: np.ndarray,
    relation: np.ndarray,
    top: np.ndarray,
    f_available: np.ndarray,
    s_available: np.ndarray,
    relation_available: np.ndarray,
    relation_features: np.ndarray,
) -> np.ndarray:
    if len(top) == 0:
        raise ValueError("Empty E2R candidate head")
    base = core[top]
    order10 = set(range(min(10, len(top))))
    order100 = set(range(min(100, len(top))))
    output = list(np.asarray(relation_features, dtype=np.float64))
    for raw, full_av in (
        (functional, f_available),
        (structure, s_available),
        (relation, relation_available),
    ):
        score = raw[top]
        available = full_av[top]
        supported = score[available]
        coverage = float(available.mean())
        std = float(supported.std()) if len(supported) >= 2 else 0.0
        mean = float(supported.mean()) if len(supported) else 0.0
        active = score.copy()
        active[~available] = -np.inf
        supported_order = np.argsort(-active, kind="stable")
        chosen = [i for i in supported_order if available[i]]
        margin = float(score[chosen[0]] - score[chosen[1]]) if len(chosen) >= 2 else 0.0
        spread = (
            float(score[chosen[0]] - score[chosen[min(9, len(chosen) - 1)]])
            if len(chosen) >= 2 else 0.0
        )
        overlap10 = len(order10.intersection(chosen[:10])) / max(1, min(10, len(top)))
        overlap100 = len(order100.intersection(chosen[:100])) / max(1, min(100, len(top)))
        if available.sum() >= 3 and float(base[available].std()) > 1e-8 and std > 1e-8:
            corr = float(np.corrcoef(base[available], score[available])[0, 1])
        else:
            corr = 0.0
        output.extend((
            coverage, float(available[:10].mean()), std, mean,
            margin, spread, overlap10, overlap100, corr,
        ))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise RuntimeError("Invalid E2R query gate feature contract")
    return features


def predict_authority(asset: dict, features: np.ndarray) -> np.ndarray:
    """Independent non-negative permission and strength per evidence channel."""
    if tuple(asset["feature_names"]) != FEATURE_NAMES:
        raise ValueError("E2R query gate feature names are incompatible with runtime")
    x = np.asarray(features, dtype=np.float64).reshape(1, -1)
    result = []
    for channel, maximum in zip(CHANNELS, MAX_AUTHORITY):
        models = asset["channels"][channel]
        classifier = models["permission"]
        strength = models["strength"]
        if isinstance(classifier, float):
            p = float(classifier)
        else:
            p = float(classifier.predict_proba(x)[0, 1])
        raw = float(strength) if isinstance(strength, float) else float(strength.predict(x)[0])
        # Exact fallback for unsupported/no-permission channels.
        result.append(0.0 if p < float(asset["permission_threshold"]) else
                      float(np.clip(p * raw, 0.0, maximum)))
    return np.asarray(result, dtype=np.float64)


def reorder_head(
    *,
    broad_order: np.ndarray,
    lexical: np.ndarray,
    core: np.ndarray,
    functional: np.ndarray,
    structure: np.ndarray,
    relation: np.ndarray,
    authority: np.ndarray,
    k: int = TOPK,
) -> np.ndarray:
    """Keep the full Broad candidate universe, only reorder its first k members."""
    top = broad_order[:k]
    values = (
        core[top]
        + float(authority[0]) * functional[top]
        + float(authority[1]) * structure[top]
        + float(authority[2]) * relation[top]
    )
    local = np.lexsort((lexical[top], -values))
    return np.concatenate((top[local], broad_order[k:]))


def filtered_target_ranks(order: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # Clean2023-masked reaction rows are absent from order but original
    # reaction indices remain sparse in the full candidate universe.
    ranks = np.empty(int(np.max(order)) + 1, dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    values = ranks[targets].astype(np.int64, copy=True)
    return values - np.sum(values[:, None] > values[None, :], axis=1)


def monotone(ranks: np.ndarray) -> bool:
    r = np.asarray(ranks)
    return (r <= 3).mean() <= (r <= 10).mean() <= (r <= 100).mean()


class FrozenE2RSurface:
    """Reuse the production Broad/EnzGFM/CLIPZyme tensors and clean2023 graph.

    This score path does not reference any validation or test labels.
    """

    def __init__(self, runtime: object):
        import torch

        self.rt = runtime
        self.lex = runtime._reaction_lex
        self.count = len(runtime.index.reaction_ids)
        self.frows = runtime._e2r_functional_r_row
        self.crows = runtime._clip_r_row
        self.fmask = self.frows >= 0
        self.smask = (self.crows >= 0) & runtime.clip.r_supported[np.maximum(self.crows, 0)]
        self.f_index = torch.as_tensor(
            self.frows[self.fmask], dtype=torch.long, device=runtime.index.device
        )
        self.s_index = torch.as_tensor(
            self.crows[self.smask], dtype=torch.long, device=runtime.index.device
        )
        self.member_f = runtime.e2r_members["enzgfm_e2r"]
        self.member_s = runtime.e2r_members["clipzyme_structure"]
        self.scale = float(runtime.e2r_core_cal["scale"])
        self.center = float(runtime.e2r_core_cal["center"])

    def score(self, protein_id: str) -> dict:
        import torch

        rt = self.rt
        with torch.no_grad():
            prow = rt.index.protein_index[protein_id]
            broad = (
                rt.index.protein_embeddings[prow]
                @ rt.index.reaction_embeddings.T
            ).float().cpu().numpy().astype(np.float64, copy=False)

            broad_std = max(float(broad.std()), 1e-8)
            broad_z = (broad - float(broad.mean())) / broad_std
            ctx, ca, cb, ctx_av, ctx_both = rt._relation_components("e2r", protein_id)
            # Stabilize only gate features under near-tied fp32 GPU scores.
            # Neither physical evidence scores nor candidate ranks are rounded.
            rel_features = rt._relation_features(
                "e2r", protein_id, np.round(broad_z, 4), np.round(ctx, 4),
                np.round(ca, 4), np.round(cb, 4), ctx_av, ctx_both,
            )
            f_query = rt.functional_e2r.p_index.get(protein_id, -1)
            f_available = self.fmask.copy()
            f_score = np.zeros(self.count, dtype=np.float64)
            if f_query < 0:
                f_available[:] = False
            else:
                values = (
                    rt.functional_e2r.r.index_select(0, self.f_index)
                    @ rt.functional_e2r.p[f_query]
                ).float().cpu().numpy()
                cal = self.member_f["calibration"]
                f_score[f_available] = (
                    values - float(cal["score_center"])
                ) / max(float(cal["score_scale"]), 1e-8)

            s_query = rt.clip.p_index.get(protein_id, -1)
            s_available = self.smask.copy()
            s_score = np.zeros(self.count, dtype=np.float64)
            if s_query < 0 or not bool(rt.clip.p_supported[s_query]):
                s_available[:] = False
            else:
                values = (
                    rt.clip.r_device.index_select(0, self.s_index)
                    @ rt.clip.p_device[s_query]
                ).float().cpu().numpy()
                cal = self.member_s["calibration"]
                s_score[s_available] = (
                    values - float(cal["score_center"])
                ) / max(float(cal["score_scale"]), 1e-8)

        f = float(self.member_f["strength"]) * f_score
        s = float(self.member_s["strength"]) * s_score
        relation = (broad_std / self.scale) * ctx
        core = (broad - self.center) / self.scale
        known_rows = np.asarray([
            rt.index.reaction_index[r]
            for r in rt.known_by_protein.get(protein_id, set())
            if r in rt.index.reaction_index
        ], dtype=np.int64)
        valid = broad.copy()
        valid[known_rows] = -np.inf
        order = np.lexsort((self.lex, -valid))
        order = order[np.isfinite(valid[order])]
        top = order[:TOPK]
        features = feature_vector(
            core=np.round(core, 4),
            functional=np.round(f, 4),
            structure=np.round(s, 4),
            relation=np.round(relation, 4),
            top=top,
            f_available=f_available,
            s_available=s_available,
            relation_available=ctx_av,
            relation_features=rel_features,
        )
        return {
            "core": core,
            "broad": broad,
            "functional": f,
            "structure": s,
            "relation": relation,
            "features": features,
            "order": order,
            "top": top,
            "known_rows": known_rows,
            "functional_av": bool(f_available[top].any()),
            "structure_av": bool(s_available[top].any()),
            "relation_av": bool(ctx_av[top].any()),
        }


# Predeclared interpretable joint evidence configurations. The optimizer never
# searches/test-tunes this set; it predicts one route per E2R enzyme query.
JOINT_ROUTES = {
    "broad": (0.0, 0.0, 0.0),
    "functional": (1.0, 0.0, 0.0),
    "structure": (0.0, 1.0, 0.0),
    "general": (1.0, 1.0, 0.0),
    "relation": (0.0, 0.0, 1.0),
    "all_evidence": (1.0, 1.0, 1.0),
    "relation_emphasis": (1.0, 1.0, 3.0),
}


def predict_joint_route(
    asset: dict,
    features: np.ndarray,
    evidence: dict,
) -> tuple[str, np.ndarray, dict[str, float]]:
    """One query-level evidence route; broad fallback if no admissible channel.

    All inputs are unlabeled query scores, training-graph support features,
    and true expert availability. Never requires labels of candidate reactions.
    """
    if tuple(asset["feature_names"]) != tuple(FEATURE_NAMES):
        raise ValueError("Feature vocabulary changed")
    if asset["route_authorities"] != JOINT_ROUTES:
        raise ValueError("Versioned joint evidence route schema mismatch")
    x = np.asarray(features, np.float64).reshape(1, -1)
    prediction: dict[str, float] = {"broad": 0.0}
    for name, model in asset["predictors"].items():
        if name not in JOINT_ROUTES or name == "broad":
            raise ValueError(f"Unrecognized E2R route predictor: {name}")
        prediction[name] = float(model.predict(x)[0])
    availability = np.asarray(
        [evidence["functional_av"], evidence["structure_av"], evidence["relation_av"]],
        dtype=bool,
    )
    for name, weights in JOINT_ROUTES.items():
        if bool(np.any((np.asarray(weights) > 0.0) & (~availability))):
            prediction[name] = -np.inf
    selection = max(JOINT_ROUTES, key=lambda name: prediction[name])
    if prediction[selection] <= float(asset["baseline_guard"]):
        selection = "broad"
    return selection, np.asarray(JOINT_ROUTES[selection], np.float64), prediction
