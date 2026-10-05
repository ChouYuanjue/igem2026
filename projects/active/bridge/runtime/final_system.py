from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.evidence.experts import ReactionCenterPairEvidence
from projects.active.bridge.evidence.pair_scores import (
    ClipzymePairEvidence,
    enzgfm_pair_evidence,
)
from projects.active.bridge.core.taxonomy_scope import filter_candidate_ids, validate_scope
from projects.active.bridge.kernel.evidence_fusion import query_standardize
from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import FibreCandidateIndex

TOPK_R2E = 1000
RELATION_CONTEXT_ALPHA = 20.0
RELATION_CONTEXT_PROTECTED = 5
RELATION_CONTEXT_END = 100
POCKET_ALPHA = 0.35
POCKET_PREFIX = 20

RUNTIME_ASSETS = ROOT / "projects/active/bridge/release/runtime/final_bridge_v1"
ROUTER = RUNTIME_ASSETS / "router.production.pkl"
TPS_ROOT = ROOT / "results/fibre_tps_specialist_response_v2"
TPS_GATE = RUNTIME_ASSETS / "tps_gate.production.pkl"
FAMILY_GATE = RUNTIME_ASSETS / "family_gates.production.pkl"
FAMILY_APP = ROOT / "results/fibre_family_applicability_router_v1/full_outer_predictions.csv"
FAMILY_REACTION = ROOT / "results/enzymecage_reaction_family_response_v1/full_outer/query_features.csv"
FAMILY_PAIR = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/pair_features.csv.gz"
FAMILY_QUERY = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/query_features_calibrated.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
TRAIN_RELATIONS = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
POCKET_PAIR_SCORES = ROOT / "results/bridge_pocket_loo_expert_v10/outer_pair_scores.csv.gz"
E2R_BUNDLE = ROOT / "projects/active/bridge/release/manifests/score_evidence_v1/e2r_bundle.json"
E2R_CORE_CAL = ROOT / "projects/active/bridge/release/manifests/score_evidence_v1/e2r_core_calibration.json"

FAMILIES = ("p450", "phosphatase", "terpene")
FAMILY_SCALE = {"p450": 0.5, "phosphatase": 0.25, "terpene": 0.75}
TPS_FEATURES = (
    "tps_ref_max_cosine",
    "tps_ref_top5_mean",
    "tps_ref_top20_mean",
    "tps_ref_margin_1_2",
    "tps_ref_softmax_entropy",
    "tps_score_mean",
    "tps_score_std",
    "tps_score_top1_margin",
    "tps_score_top20_mean",
)


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)


def _z_masked(values: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    """Query-standardize available entries; unavailable evidence is neutral zero."""
    out = torch.zeros_like(values)
    count = available.sum(1, keepdim=True)
    valid = count[:, 0] > 1
    if valid.any():
        selected = values[valid]
        mask = available[valid]
        n = mask.sum(1, keepdim=True).float()
        mean = (selected * mask).sum(1, keepdim=True) / n
        var = (((selected - mean) * mask) ** 2).sum(1, keepdim=True) / n
        standardized = (selected - mean) / var.sqrt().clamp_min(1e-6)
        standardized[~mask] = 0
        out[valid] = standardized
    return out


def _build_relation_prototypes(
    index: FibreCandidateIndex,
    train: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the frozen clean2023 reciprocal relation prototypes used by BRIDGE."""
    protein = index.protein_embeddings
    reaction = index.reaction_embeddings
    protein_sum = torch.zeros_like(protein)
    protein_count = torch.zeros((len(index.protein_ids), 1), device=index.device)
    reaction_sum = torch.zeros_like(reaction)
    reaction_count = torch.zeros((len(index.reaction_ids), 1), device=index.device)

    protein_rows: list[int] = []
    reaction_rows: list[int] = []
    for protein_id, reaction_id in train[["protein_id", "reaction_id"]].itertuples(index=False):
        p = index.protein_index.get(str(protein_id))
        r = index.reaction_index.get(str(reaction_id))
        if p is None or r is None:
            continue
        protein_rows.append(p)
        reaction_rows.append(r)

    if protein_rows:
        pidx = torch.as_tensor(protein_rows, dtype=torch.long, device=index.device)
        ridx = torch.as_tensor(reaction_rows, dtype=torch.long, device=index.device)
        ones = torch.ones((len(pidx), 1), device=index.device)
        protein_sum.index_add_(0, pidx, reaction.index_select(0, ridx))
        protein_count.index_add_(0, pidx, ones)
        reaction_sum.index_add_(0, ridx, protein.index_select(0, pidx))
        reaction_count.index_add_(0, ridx, ones)

    protein_mask = protein_count[:, 0] > 0
    reaction_mask = reaction_count[:, 0] > 0
    protein_proto = torch.zeros_like(protein)
    reaction_proto = torch.zeros_like(reaction)
    protein_proto[protein_mask] = _normalize_rows(
        protein_sum[protein_mask] / protein_count[protein_mask]
    )
    reaction_proto[reaction_mask] = _normalize_rows(
        reaction_sum[reaction_mask] / reaction_count[reaction_mask]
    )
    return protein_proto, protein_mask, reaction_proto, reaction_mask


def _z(values: np.ndarray, available: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if available is None:
        available = np.ones(len(x), dtype=bool)
    return query_standardize(
        torch.as_tensor(x, dtype=torch.float64),
        torch.as_tensor(available, dtype=torch.bool),
    ).cpu().numpy()


def _corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 3:
        return 0.0
    x, y = a[mask], b[mask]
    if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _margin(x: np.ndarray, mask: np.ndarray) -> float:
    v = x[mask]
    if len(v) < 2:
        return 0.0
    pair = np.partition(v, -2)[-2:]
    return float(pair.max() - pair.min())


def _overlap(core: np.ndarray, expert: np.ndarray, mask: np.ndarray, k: int = 20) -> float:
    if not mask.any():
        return 0.0
    rows = np.flatnonzero(mask)
    kk = min(int(k), len(rows))
    a = set(rows[np.argsort(core[rows])[-kk:]].tolist())
    b = set(rows[np.argsort(expert[rows])[-kk:]].tolist())
    return float(len(a & b) / max(len(a | b), 1))


def _family_features(
    core: np.ndarray,
    expert: np.ndarray,
    available: np.ndarray,
    pocket: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            float(available.mean()),
            float(expert[available].std()) if available.any() else 0.0,
            _margin(expert, available),
            _corr(core, expert, available),
            _overlap(core, expert, available),
            _margin(core, np.ones(len(core), dtype=bool)),
            float(core.std()),
            float(pocket.mean()),
        ],
        dtype=np.float32,
    )


def _predict_weight(bundle: dict[str, Any], matrix: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    pred: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family in ("functional", "geometry"):
        classifier, regressor = bundle["models"][family]
        probability = classifier.predict_proba(matrix)[:, 1]
        strength = np.clip(regressor.predict(matrix), 0.0, 0.75)
        weight = np.where(
            probability >= bundle["thresholds"][family],
            strength * bundle["scales"][family],
            0.0,
        )
        pred[family] = (probability, np.clip(weight, 0.0, 0.75))
    return pred


def _family_feature_columns(family: str, suffixes: list[str] | tuple[str, ...]) -> list[str]:
    return [f"{family}_{suffix}" for suffix in suffixes] + [
        "family_calibrated_max_percentile",
        "family_calibrated_margin",
        "family_calibrated_entropy",
        "candidate_count",
    ]


def _calibrated(raw: np.ndarray, available: np.ndarray, center: float, scale: float) -> np.ndarray:
    out = np.zeros(len(raw), dtype=np.float64)
    if available.any():
        out[available] = (raw[available] - float(center)) / max(float(scale), 1e-8)
    return out


class FinalBridgeRuntime:
    """Production wrapper for the frozen final BRIDGE system.

    It intentionally mirrors the canonical relation-unseen evaluation semantics:
    Broad owns the full candidate universe; admitted experts make bounded
    corrections; reciprocal clean2023 relation context can only reorder ranks
    6..100. Raw external entities remain the responsibility of the existing
    open-world encoder path until all final expert views define external inputs.
    """

    version = "bridge-final-v1"

    def __init__(self, *, device: str | None = None) -> None:
        self.device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._lock = threading.RLock()
        self.index = FibreCandidateIndex(device=self.device)
        self.train = pd.read_csv(TRAIN_RELATIONS, dtype=str).fillna("").drop_duplicates(
            ["protein_id", "reaction_id"]
        )
        self.pproto, self.pmask, self.rproto, self.rmask = _build_relation_prototypes(
            self.index, self.train
        )
        self.known_by_reaction = (
            self.train.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
        )
        self.known_by_protein = (
            self.train.groupby("protein_id")["reaction_id"].apply(lambda x: set(map(str, x))).to_dict()
        )
        self._protein_lex = self._lex(self.index.protein_ids)
        self._reaction_lex = self._lex(self.index.reaction_ids)

        self.functional_r2e = enzgfm_pair_evidence("r2e", device=self.device)
        self.functional_e2r = enzgfm_pair_evidence("e2r", device=self.device)
        self.clip = ClipzymePairEvidence(device=self.device)
        self.mechanism = ReactionCenterPairEvidence(device=self.device)

        with open(ROUTER, "rb") as handle:
            self.router = pickle.load(handle)
        with open(TPS_GATE, "rb") as handle:
            self.tps_gate = pickle.load(handle)
        with open(FAMILY_GATE, "rb") as handle:
            self.family_gate = pickle.load(handle)

        self.family_app = pd.read_csv(FAMILY_APP, dtype={"reaction_id": str}).fillna("").set_index(
            "reaction_id", drop=False
        )
        self.family_reaction = pd.read_csv(
            FAMILY_REACTION, dtype={"reaction_id": str}
        ).fillna("").set_index("reaction_id", drop=False)
        pair = pd.read_csv(FAMILY_PAIR, dtype=str).fillna("")
        self.family_pair_groups = {
            str(key): group for key, group in pair.groupby("CANO_RXN_SMILES", sort=False)
        }
        self.family_query = pd.read_csv(FAMILY_QUERY, dtype=str).fillna("").set_index(
            "CANO_RXN_SMILES", drop=False
        )
        reaction_meta = pd.read_csv(REACTIONS, dtype=str).fillna("")
        self.reaction_smiles = dict(
            zip(reaction_meta["reaction_id"].astype(str), reaction_meta["reaction_smiles"].astype(str))
        )

        self.tps_q = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str).fillna("").set_index(
            "query_id", drop=False
        )
        self.tps_ids = pd.read_csv(TPS_ROOT / "full_outer/query_features.csv", dtype=str)[
            "query_id"
        ].astype(str).tolist()
        self.tps_z = np.load(TPS_ROOT / "full_outer/reaction_tps.npy", mmap_mode="r")
        self.tps_index = {query: i for i, query in enumerate(self.tps_ids)}
        self.tps_protein = np.load(TPS_ROOT / "broad_protein_tps.npy", mmap_mode="r")

        pocket = pd.read_csv(POCKET_PAIR_SCORES, dtype=str).fillna("")
        pocket["pocket_interaction_score"] = pd.to_numeric(
            pocket["pocket_interaction_score"], errors="coerce"
        )
        self.pocket_groups = {
            str(key): group[["protein_id", "pocket_interaction_score"]].dropna()
            for key, group in pocket.groupby("reaction_id", sort=False)
        }

        self.e2r_bundle = json.loads(E2R_BUNDLE.read_text())
        self.e2r_core_cal = json.loads(E2R_CORE_CAL.read_text())
        self.e2r_members = {
            member["descriptor"]["name"]: member for member in self.e2r_bundle["members"]
        }

        self._r2e_functional_p_row = np.asarray(
            [self.functional_r2e.p_index.get(pid, -1) for pid in self.index.protein_ids],
            dtype=np.int64,
        )
        self._clip_p_row = np.asarray(
            [self.clip.p_index.get(pid, -1) for pid in self.index.protein_ids],
            dtype=np.int64,
        )
        self._mechanism_p_row = np.asarray(
            [self.mechanism.p_index.get(pid, -1) for pid in self.index.protein_ids],
            dtype=np.int64,
        )
        self._e2r_functional_r_row = np.asarray(
            [self.functional_e2r.r_index.get(rid, -1) for rid in self.index.reaction_ids],
            dtype=np.int64,
        )
        self._clip_r_row = np.asarray(
            [self.clip.r_index.get(rid, -1) for rid in self.index.reaction_ids],
            dtype=np.int64,
        )

    @staticmethod
    def _lex(values: list[str]) -> np.ndarray:
        lex = np.empty(len(values), dtype=np.int64)
        order = np.argsort(np.asarray(values, dtype=object), kind="stable")
        lex[order] = np.arange(len(order))
        return lex

    def contains(self, command: str, query_id: str) -> bool:
        if command == "rank-enzymes":
            return str(query_id) in self.index.reaction_index
        if command == "rank-reactions":
            return str(query_id) in self.index.protein_index
        return False

    def status(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "version": self.version,
            "device": self.device,
            "protein_candidates": len(self.index.protein_ids),
            "reaction_candidates": len(self.index.reaction_ids),
            "r2e_experts": [
                "Broad",
                "EnzGFM",
                "CLIPZyme",
                "reaction_center",
                "family_specialists",
                "TPS",
                "pocket_interaction",
                "reciprocal_relation_context",
            ],
            "e2r_experts": [
                "Broad",
                "EnzGFM",
                "CLIPZyme",
                "reciprocal_relation_context",
            ],
        }

    def _mask_rows(
        self,
        score: torch.Tensor,
        ids: list[str],
        index: dict[str, int],
    ) -> None:
        rows = [index[value] for value in ids if value in index]
        if rows:
            score[torch.as_tensor(rows, dtype=torch.long, device=score.device)] = -torch.inf

    def _relation_context(self, direction: str, query_id: str) -> np.ndarray:
        P, R = self.index.protein_embeddings, self.index.reaction_embeddings
        if direction == "r2e":
            qrow = self.index.reaction_index[query_id]
            qemb = R[qrow : qrow + 1]
            qa = self.rproto[qrow : qrow + 1] @ P.T
            qa_av = self.rmask[qrow : qrow + 1, None].expand_as(qa)
            qb = qemb @ self.pproto.T
            qb_av = self.pmask[None, :].expand_as(qb)
        else:
            qrow = self.index.protein_index[query_id]
            qemb = P[qrow : qrow + 1]
            qa = self.pproto[qrow : qrow + 1] @ R.T
            qa_av = self.pmask[qrow : qrow + 1, None].expand_as(qa)
            qb = qemb @ self.rproto.T
            qb_av = self.rmask[None, :].expand_as(qb)
        za = _z_masked(qa, qa_av)
        zb = _z_masked(qb, qb_av)
        den = qa_av.float() + qb_av.float()
        ctx = (za + zb) / den.clamp_min(1.0)
        ctx[den == 0] = 0
        return ctx[0].detach().cpu().numpy().astype(np.float64, copy=False)

    def _apply_relation_context(
        self,
        ordered_rows: np.ndarray,
        base_score: np.ndarray,
        ctx: np.ndarray,
        lex: np.ndarray,
    ) -> np.ndarray:
        if len(ordered_rows) <= RELATION_CONTEXT_PROTECTED:
            return ordered_rows
        end = min(RELATION_CONTEXT_END, len(ordered_rows))
        head = ordered_rows.copy()
        rows = head[RELATION_CONTEXT_PROTECTED:end]
        mix = base_score[rows] + RELATION_CONTEXT_ALPHA * ctx[rows]
        head[RELATION_CONTEXT_PROTECTED:end] = rows[
            np.lexsort((lex[rows], -mix))
        ]
        return head

    def _r2e_general_weights(
        self,
        query_id: str,
        core: np.ndarray,
        functional_score: np.ndarray,
        functional_available: np.ndarray,
        geometry_score: np.ndarray,
        geometry_available: np.ndarray,
    ) -> tuple[float, float]:
        pocket = np.zeros(len(core), dtype=np.float32)
        base_x = np.concatenate(
            [
                _family_features(core, functional_score, functional_available, pocket),
                _family_features(core, geometry_score, geometry_available, pocket),
            ]
        )
        reaction = self.index.reaction_embeddings[
            self.index.reaction_index[query_id]
        ].detach().cpu().numpy()[None, :]
        rpca = self.router["pca"].transform(reaction).astype(np.float32)[0]
        matrix = np.concatenate(
            [base_x, np.zeros(4, dtype=np.float32), rpca]
        )[None, :].astype(np.float32)
        pred = _predict_weight(self.router, matrix)
        return float(pred["functional"][1][0]), float(pred["geometry"][1][0])

    def _r2e_specialists(
        self,
        query_id: str,
        candidate_ids: list[str],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        score = np.zeros(len(candidate_ids), dtype=np.float64)
        audit: dict[str, Any] = {"family": "none", "family_weight": 0.0, "tps_weight": 0.0}
        if query_id in self.family_app.index and query_id in self.family_reaction.index:
            app = self.family_app.loc[query_id]
            if isinstance(app, pd.DataFrame):
                app = app.iloc[0]
            reaction = self.family_reaction.loc[query_id]
            if isinstance(reaction, pd.DataFrame):
                reaction = reaction.iloc[0]
            family = str(app["pred"])
            confidence = float(app["confidence"])
            smiles = self.reaction_smiles.get(query_id, "")
            if (
                family in FAMILIES
                and confidence >= 0.50
                and str(reaction["reaction_family_winner"]) == family
                and smiles in self.family_query.index
                and smiles in self.family_pair_groups
            ):
                query = self.family_query.loc[smiles]
                if isinstance(query, pd.DataFrame):
                    query = query.iloc[0]
                if str(query["family_calibrated_winner"]) == family:
                    group = self.family_pair_groups[smiles]
                    local = {pid: i for i, pid in enumerate(candidate_ids)}
                    residual = np.zeros(len(candidate_ids), dtype=np.float64)
                    available = np.zeros(len(candidate_ids), dtype=bool)
                    column = f"{family}_normalized_logit_response"
                    for pid, raw in group[["protein_id", column]].itertuples(index=False):
                        row = local.get(str(pid))
                        if row is None:
                            continue
                        residual[row] = float(raw)
                        available[row] = True
                    residual = _z(residual, available)
                    columns = _family_feature_columns(
                        family, self.family_gate["feature_suffixes"]
                    )
                    features = [float(query[column]) for column in columns] + [
                        float(available.mean())
                    ]
                    raw_weight = float(
                        np.clip(
                            self.family_gate["models"][family]
                            .predict(np.asarray(features, dtype=float)[None, :])[0],
                            0.0,
                            0.50,
                        )
                    )
                    weight = raw_weight * FAMILY_SCALE[family]
                    if weight > 0:
                        score += weight * residual
                        audit.update({"family": family, "family_weight": weight})

        if query_id in self.tps_q.index and query_id in self.tps_index:
            query = self.tps_q.loc[query_id]
            if isinstance(query, pd.DataFrame):
                query = query.iloc[0]
            if (
                float(query["tps_ref_max_cosine"])
                >= float(self.tps_gate["semantic_threshold"])
                and audit["family"] not in ("p450", "phosphatase")
            ):
                qz = np.asarray(self.tps_z[self.tps_index[query_id]], dtype=np.float32)
                rows = np.asarray(
                    [self.index.protein_index[pid] for pid in candidate_ids], dtype=np.int64
                )
                raw = (np.asarray(self.tps_protein[rows], dtype=np.float32) @ qz).astype(
                    np.float64
                )
                sorted_values = np.sort(raw)
                values = {
                    "tps_ref_max_cosine": float(query["tps_ref_max_cosine"]),
                    "tps_ref_top5_mean": float(query["tps_ref_top5_mean"]),
                    "tps_ref_top20_mean": float(query["tps_ref_top20_mean"]),
                    "tps_ref_margin_1_2": float(query["tps_ref_margin_1_2"]),
                    "tps_ref_softmax_entropy": float(query["tps_ref_softmax_entropy"]),
                    "tps_score_mean": float(raw.mean()),
                    "tps_score_std": float(raw.std()),
                    "tps_score_top1_margin": float(sorted_values[-1] - sorted_values[-2]),
                    "tps_score_top20_mean": float(sorted_values[-20:].mean()),
                }
                features = np.asarray(
                    [values[key] for key in TPS_FEATURES], dtype=float
                )[None, :]
                raw_weight = float(
                    np.clip(self.tps_gate["regressor"].predict(features)[0], 0.0, 0.5)
                )
                weight = raw_weight * float(self.tps_gate["strength_scale"])
                if weight > 0:
                    score += weight * _z(raw)
                    audit["tps_weight"] = weight
        return score, audit

    def _pocket_reorder(
        self,
        query_id: str,
        candidate_ids: list[str],
        base_score: np.ndarray,
    ) -> np.ndarray:
        order = np.argsort(-base_score, kind="stable")
        group = self.pocket_groups.get(str(query_id))
        if group is None or len(group) < 2:
            return order
        inverse = np.empty(len(order), dtype=np.int32)
        inverse[order] = np.arange(1, len(order) + 1)
        local = {pid: i for i, pid in enumerate(candidate_ids)}
        rows: list[int] = []
        values: list[float] = []
        for pid, raw in group[["protein_id", "pocket_interaction_score"]].itertuples(index=False):
            row = local.get(str(pid))
            if row is None or int(inverse[row]) <= POCKET_PREFIX:
                continue
            rows.append(row)
            values.append(float(raw))
        if len(rows) < 2:
            return order
        locs = np.asarray(rows, dtype=np.int64)
        mix = _z(base_score[locs]) + POCKET_ALPHA * _z(np.asarray(values, dtype=np.float64))
        slots = np.sort(inverse[locs])
        ranked = locs[np.argsort(-mix, kind="stable")]
        final_inverse = inverse.copy()
        final_inverse[ranked] = slots
        return np.argsort(final_inverse, kind="stable")

    def rank_enzymes(self, payload: dict[str, Any]) -> dict[str, Any]:
        query_id = str(payload.get("reaction_id") or "").strip()
        if query_id not in self.index.reaction_index:
            raise KeyError(query_id)
        top_k = max(1, int(payload.get("top_k") or 10))
        with self._lock, torch.no_grad():
            qrow = self.index.reaction_index[query_id]
            full = (
                self.index.reaction_embeddings[qrow] @ self.index.protein_embeddings.T
            ).float()
            masks = list(payload.get("mask_enzyme_ids") or [])
            masks.extend(
                self.known_by_reaction.get(query_id, set())
                if payload.get("mask_clean2023", False)
                else []
            )
            self._mask_rows(full, list(dict.fromkeys(map(str, masks))), self.index.protein_index)

            taxonomy_scope = validate_scope(
                str(payload.get("enzyme_taxonomy_scope") or "all")
            )
            taxonomy_rows, taxonomy_audit = filter_candidate_ids(
                self.index.protein_ids, taxonomy_scope
            )
            if taxonomy_scope != "all":
                taxonomy_allowed = torch.zeros(
                    len(self.index.protein_ids), dtype=torch.bool, device=full.device
                )
                if taxonomy_rows:
                    taxonomy_allowed[
                        torch.as_tensor(
                            taxonomy_rows, dtype=torch.long, device=full.device
                        )
                    ] = True
                full[~taxonomy_allowed] = -torch.inf

            subset = [str(x) for x in payload.get("candidate_ids") or [] if str(x)]
            if subset:
                allowed = torch.zeros(len(self.index.protein_ids), dtype=torch.bool, device=full.device)
                rows = [self.index.protein_index[x] for x in subset if x in self.index.protein_index]
                if rows:
                    allowed[torch.as_tensor(rows, dtype=torch.long, device=full.device)] = True
                full[~allowed] = -torch.inf
            effective_candidates = int(torch.isfinite(full).sum().item())
            if effective_candidates <= 0:
                raise ValueError("No enzyme candidates remain after product-layer constraints")
            k = min(TOPK_R2E, effective_candidates)
            values, indices = torch.topk(full, k=k, largest=True, sorted=True)
            top = indices.detach().cpu().numpy().astype(np.int64, copy=False)
            candidates = [self.index.protein_ids[int(row)] for row in top]
            core = values.detach().cpu().numpy().astype(np.float64, copy=False)
            core_z = (core - core.mean()) / max(float(core.std()), 1e-6)

            frows = self._r2e_functional_p_row[top]
            fr = self.functional_r2e.r_index.get(query_id, -1)
            fa = frows >= 0
            fraw = np.zeros(k, dtype=np.float64)
            if fr < 0:
                fa[:] = False
            elif fa.any():
                rows_t = torch.as_tensor(frows[fa], dtype=torch.long, device=self.index.device)
                fraw[fa] = (
                    self.functional_r2e.p.index_select(0, rows_t)
                    @ self.functional_r2e.r[fr]
                ).float().cpu().numpy()
            fz = _z(fraw, fa)

            crows = self._clip_p_row[top]
            cr = self.clip.r_index.get(query_id, -1)
            ca = crows >= 0
            if cr >= 0 and bool(self.clip.r_supported[cr]):
                ca &= self.clip.p_supported[np.maximum(crows, 0)]
            else:
                ca[:] = False
            craw = np.zeros(k, dtype=np.float64)
            if ca.any():
                rows_t = torch.as_tensor(crows[ca], dtype=torch.long, device=self.index.device)
                craw[ca] = (
                    self.clip.p_device.index_select(0, rows_t) @ self.clip.r_device[cr]
                ).float().cpu().numpy()
            cz = _z(craw, ca)

            mrows = self._mechanism_p_row[top]
            mr = self.mechanism.r_index.get(query_id, -1)
            ma = mrows >= 0
            mraw = np.zeros(k, dtype=np.float64)
            if mr < 0:
                ma[:] = False
            elif ma.any():
                rows_t = torch.as_tensor(mrows[ma], dtype=torch.long, device=self.index.device)
                mraw[ma] = (
                    self.mechanism.p.index_select(0, rows_t) @ self.mechanism.r[mr]
                ).float().cpu().numpy()
            mz = _z(mraw, ma)
            ga = ca | ma
            geometry = np.zeros(k, dtype=np.float64)
            count = np.zeros(k, dtype=np.int32)
            for score, available in ((cz, ca), (mz, ma)):
                geometry[available] += score[available]
                count[available] += 1
            geometry[ga] /= count[ga]

            functional_weight, geometry_weight = self._r2e_general_weights(
                query_id, core, fz, fa, geometry, ga
            )
            specialist, specialist_audit = self._r2e_specialists(query_id, candidates)
            final_score = core_z + functional_weight * fz + geometry_weight * geometry + specialist
            local_order = self._pocket_reorder(query_id, candidates, final_score)
            ordered_rows = top[local_order]
            full_score = np.full(len(self.index.protein_ids), -np.inf, dtype=np.float64)
            full_score[top] = final_score
            ctx = self._relation_context("r2e", query_id)
            ordered_rows = self._apply_relation_context(
                ordered_rows, full_score, ctx, self._protein_lex
            )
            selected = ordered_rows[:top_k]
            selected_local = {int(row): i for i, row in enumerate(top)}
            result = []
            for rank, row in enumerate(selected, start=1):
                local_row = selected_local[int(row)]
                result.append(
                    {
                        "rank": rank,
                        "candidate_id": self.index.protein_ids[int(row)],
                        "score": float(final_score[local_row]),
                        "selection_source": "bridge_final",
                    }
                )
        return {
            "query": {
                "query_id": query_id,
                "direction": "reaction_to_enzyme",
                "route_id": "bridge-final-r2e-v1",
                "route_version": "bridge-final-production-v1",
                "ranking_objective": str(
                    payload.get("ranking_objective")
                    or ("top3" if top_k <= 3 else "top10" if top_k <= 10 else "top20")
                ),
                "model_bundle_version": self.version,
                "score_source": "broad+gated_experts+reciprocal_context",
                "candidate_universe": "general_merged",
                "candidate_universe_size": len(self.index.protein_ids),
                "candidate_universe_pre_taxonomy_size": taxonomy_audit["pre_filter_size"],
                "candidate_universe_post_taxonomy_size": taxonomy_audit["post_filter_size"],
                "enzyme_taxonomy_scope": taxonomy_scope,
                "taxonomy_eukaryote_count": taxonomy_audit["eukaryote_count"],
                "taxonomy_prokaryote_count": taxonomy_audit["prokaryote_count"],
                "taxonomy_other_count": taxonomy_audit["other_count"],
                "taxonomy_unknown_count": taxonomy_audit["unknown_count"],
                "taxonomy_excluded_count": taxonomy_audit["excluded_count"],
                "candidate_subset_applied": bool(subset),
                "candidate_subset_requested_count": len(subset),
                "candidate_subset_effective_count": effective_candidates if subset else 0,
                "requested_top_k": top_k,
                "shot_mode": "frozen_reciprocal_relation_context",
                "functional_weight": functional_weight,
                "geometry_weight": geometry_weight,
                **specialist_audit,
            },
            "candidates": result,
        }

    def rank_reactions(self, payload: dict[str, Any]) -> dict[str, Any]:
        query_id = str(payload.get("enzyme_id") or "").strip()
        if query_id not in self.index.protein_index:
            raise KeyError(query_id)
        top_k = max(1, int(payload.get("top_k") or 10))
        functional_member = self.e2r_members["enzgfm_e2r"]
        clip_member = self.e2r_members["clipzyme_structure"]
        with self._lock, torch.no_grad():
            prow = self.index.protein_index[query_id]
            broad = (
                self.index.protein_embeddings[prow] @ self.index.reaction_embeddings.T
            ).float().cpu().numpy().astype(np.float64, copy=False)
            core = (
                broad - float(self.e2r_core_cal["center"])
            ) / float(self.e2r_core_cal["scale"])

            fp = self.functional_e2r.p_index.get(query_id, -1)
            frows = self._e2r_functional_r_row
            fa = frows >= 0
            fraw = np.zeros(len(self.index.reaction_ids), dtype=np.float64)
            if fp < 0:
                fa = np.zeros_like(fa)
            elif fa.any():
                rows_t = torch.as_tensor(frows[fa], dtype=torch.long, device=self.index.device)
                fraw[fa] = (
                    self.functional_e2r.r.index_select(0, rows_t)
                    @ self.functional_e2r.p[fp]
                ).float().cpu().numpy()
            fcal = _calibrated(
                fraw,
                fa,
                functional_member["calibration"]["score_center"],
                functional_member["calibration"]["score_scale"],
            )

            cp = self.clip.p_index.get(query_id, -1)
            crows = self._clip_r_row
            ca = crows >= 0
            craw = np.zeros(len(self.index.reaction_ids), dtype=np.float64)
            if cp < 0 or not bool(self.clip.p_supported[cp]):
                ca = np.zeros_like(ca)
            else:
                ca &= self.clip.r_supported[np.maximum(crows, 0)]
            if ca.any():
                rows_t = torch.as_tensor(crows[ca], dtype=torch.long, device=self.index.device)
                craw[ca] = (
                    self.clip.r_device.index_select(0, rows_t) @ self.clip.p_device[cp]
                ).float().cpu().numpy()
            ccal = _calibrated(
                craw,
                ca,
                clip_member["calibration"]["score_center"],
                clip_member["calibration"]["score_scale"],
            )
            score = (
                core
                + float(functional_member["strength"]) * fcal
                + float(clip_member["strength"]) * ccal
            )
            masks = list(payload.get("mask_reaction_ids") or [])
            if payload.get("mask_clean2023", False):
                masks.extend(self.known_by_protein.get(query_id, set()))
            for reaction_id in dict.fromkeys(map(str, masks)):
                row = self.index.reaction_index.get(reaction_id)
                if row is not None:
                    score[row] = -np.inf
            subset = [str(x) for x in payload.get("candidate_ids") or [] if str(x)]
            if subset:
                allowed = np.zeros(len(score), dtype=bool)
                rows = [self.index.reaction_index[x] for x in subset if x in self.index.reaction_index]
                allowed[np.asarray(rows, dtype=np.int64)] = True
                score[~allowed] = -np.inf

            order = np.lexsort((self._reaction_lex, -score))
            order = order[np.isfinite(score[order])]
            ctx = self._relation_context("e2r", query_id)
            order = self._apply_relation_context(order, score, ctx, self._reaction_lex)
            selected = order[:top_k]
            result = [
                {
                    "rank": rank,
                    "candidate_id": self.index.reaction_ids[int(row)],
                    "score": float(score[int(row)]),
                    "selection_source": "bridge_final",
                }
                for rank, row in enumerate(selected, start=1)
            ]
        return {
            "query": {
                "query_id": query_id,
                "direction": "enzyme_to_reaction",
                "route_id": "bridge-final-e2r-v1",
                "route_version": "bridge-final-production-v1",
                "ranking_objective": str(
                    payload.get("ranking_objective")
                    or ("top3" if top_k <= 3 else "top10" if top_k <= 10 else "top20")
                ),
                "model_bundle_version": self.version,
                "score_source": "broad+gated_experts+reciprocal_context",
                "candidate_universe": "general_merged",
                "candidate_universe_size": len(self.index.reaction_ids),
                "requested_top_k": top_k,
                "shot_mode": "frozen_reciprocal_relation_context",
            },
            "candidates": result,
        }

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "rank-enzymes":
            return self.rank_enzymes(payload)
        if command == "rank-reactions":
            return self.rank_reactions(payload)
        raise ValueError(f"unsupported BRIDGE command: {command}")
