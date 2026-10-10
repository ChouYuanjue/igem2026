from __future__ import annotations

import json
import math
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
from projects.active.bridge.runtime.memory import build_episodic_memory, episodic_authority

TOPK_R2E = 1000
POCKET_ALPHA = 0.35
POCKET_PREFIX = 20

RUNTIME_ASSETS = ROOT / "projects/active/bridge/release/runtime/final_bridge_v1"
E2R_JOINT_GATE_V4 = RUNTIME_ASSETS / "e2r_joint_query_v4.production.pkl"
E2R_INDUCTIVE_CONTEXT = RUNTIME_ASSETS / "e2r_inductive_relation.production.json"
ROUTER = RUNTIME_ASSETS / "router.production.pkl"
RELATION_GATE = RUNTIME_ASSETS / "adaptive_relation_gate.production.pkl"
EPISODIC_GATE = RUNTIME_ASSETS / "episodic_memory_gate.production.pkl"
TPS_ROOT = ROOT / "results/fibre_tps_specialist_response_v2"
TPS_GATE = RUNTIME_ASSETS / "tps_gate.production.pkl"
FAMILY_GATE = RUNTIME_ASSETS / "family_gates.production.pkl"
FAMILY_APP = ROOT / "results/fibre_family_applicability_router_v1/full_outer_predictions.csv"
FAMILY_REACTION = ROOT / "results/enzymecage_reaction_family_response_v1/full_outer/query_features.csv"
FAMILY_PAIR = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/pair_features.csv.gz"
FAMILY_QUERY = ROOT / "results/enzymecage_family_response_v1/full_outer_specialists/query_features_calibrated.csv"
REACTIONS = ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv"
REACTION_FEATURE_MANIFEST = (
    ROOT
    / "data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_v1/manifest.json"
)
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

RELATION_FEATURES = (
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
    corrections. clean2023 relation context is retained as validation-frozen
    long-term graph memory. Runtime positives that were not training relations
    form a separate episodic support memory and receive a query-specific 0..1
    trust value learned only on validation episodes. Raw external entities remain
    the responsibility of the existing open-world encoder path until all final
    expert views define external inputs.
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
        self._protein_relation_seen = self.pmask.detach().cpu().numpy().astype(bool)
        self._reaction_relation_seen = self.rmask.detach().cpu().numpy().astype(bool)

        self.functional_r2e = enzgfm_pair_evidence("r2e", device=self.device)
        self.functional_e2r = enzgfm_pair_evidence("e2r", device=self.device)
        self.clip = ClipzymePairEvidence(device=self.device)
        self.mechanism = ReactionCenterPairEvidence(device=self.device)

        with open(ROUTER, "rb") as handle:
            self.router = pickle.load(handle)
        with open(RELATION_GATE, "rb") as handle:
            self.relation_gate = pickle.load(handle)
        with open(EPISODIC_GATE, "rb") as handle:
            self.episodic_gate = pickle.load(handle)
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
        self.reaction_feature_schema = dict(
            json.loads(REACTION_FEATURE_MANIFEST.read_text())["contract"]
        )

        self.tps_q = pd.read_csv(
            TPS_ROOT / "production/query_features.csv", dtype=str
        ).fillna("").set_index("query_id", drop=False)
        self.tps_ids = pd.read_csv(
            TPS_ROOT / "production/query_features.csv", dtype=str
        )["query_id"].astype(str).tolist()
        self.tps_z = np.load(
            TPS_ROOT / "production/reaction_tps.npy", mmap_mode="r"
        )
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
                "episodic_memory",
            ],
            "e2r_experts": [
                "Broad",
                "EnzGFM",
                "CLIPZyme",
                "reciprocal_relation_context",
                "episodic_memory",
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

    def _external_protein_support_embeddings(
        self,
        payload: dict[str, Any],
        requested_ids: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Encode runtime protein supports in the same frozen Broad space.

        This path is intentionally lazy: ordinary registered-entity requests do
        not import or load ESM-C.  Failures stay neutral so an optional external
        support cannot break the already-valid BRIDGE prior.
        """
        missing_ids = tuple(
            value for value in requested_ids if value not in self.index.protein_index
        )
        path_value = str(payload.get("external_enzymes_csv") or "").strip()
        if not missing_ids or not path_value:
            return {}, []
        try:
            from projects.active.bridge.runtime.cli import (
                encode_external_enzymes_with_audit,
                load_external_enzyme_rows,
            )

            frame = load_external_enzyme_rows(Path(path_value))
            frame = frame[frame["enzyme_id"].astype(str).isin(set(missing_ids))].copy()
            if frame.empty:
                return {}, []
            features, _ = encode_external_enzymes_with_audit(
                frame,
                self.device,
                "esmc_600m",
                input_policy="warn",
            )
            encoded = self.index.model.encode_proteins(
                torch.as_tensor(
                    features,
                    dtype=torch.float32,
                    device=self.index.device,
                )
            )
            return {
                str(identifier): encoded[row]
                for row, identifier in enumerate(frame["enzyme_id"].astype(str).tolist())
            }, []
        except Exception as exc:
            return {}, [f"{type(exc).__name__}:{exc}"]

    def _external_reaction_support_embeddings(
        self,
        payload: dict[str, Any],
        requested_ids: tuple[str, ...],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Encode runtime reaction supports with the frozen Broad reaction schema."""
        missing_ids = tuple(
            value for value in requested_ids if value not in self.index.reaction_index
        )
        path_value = str(payload.get("external_reactions_csv") or "").strip()
        if not missing_ids or not path_value:
            return {}, []
        try:
            from projects.active.bridge.runtime.cli import (
                encode_reaction_with_audit,
                load_external_reaction_rows,
            )

            frame = load_external_reaction_rows(Path(path_value))
            frame = frame[frame["reaction_id"].astype(str).isin(set(missing_ids))].copy()
            if frame.empty:
                return {}, []
            features = []
            for reaction_smiles in frame["reaction_smiles"].astype(str):
                values, _ = encode_reaction_with_audit(
                    reaction_smiles,
                    self.reaction_feature_schema,
                    failure_policy="warn",
                )
                features.append(values)
            encoded = self.index.model.encode_reactions(
                torch.as_tensor(
                    np.stack(features).astype(np.float32, copy=False),
                    dtype=torch.float32,
                    device=self.index.device,
                )
            )
            return {
                str(identifier): encoded[row]
                for row, identifier in enumerate(frame["reaction_id"].astype(str).tolist())
            }, []
        except Exception as exc:
            return {}, [f"{type(exc).__name__}:{exc}"]

    @staticmethod
    def _relation_top_rows(values: np.ndarray, k: int) -> np.ndarray:
        kk = min(int(k), len(values))
        if kk <= 0:
            return np.empty(0, dtype=np.int64)
        rows = np.argpartition(-values, kk - 1)[:kk]
        return rows[np.argsort(-values[rows], kind="stable")]

    @staticmethod
    def _relation_safe_corr(
        a: np.ndarray,
        b: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        if int(mask.sum()) < 3:
            return 0.0
        x, y = a[mask], b[mask]
        if float(x.std()) < 1e-8 or float(y.std()) < 1e-8:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _relation_components(
        self,
        direction: str,
        query_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        return (
            ctx[0].detach().cpu().numpy().astype(np.float64, copy=False),
            za[0].detach().cpu().numpy().astype(np.float64, copy=False),
            zb[0].detach().cpu().numpy().astype(np.float64, copy=False),
            (den[0] > 0).detach().cpu().numpy(),
            (qa_av[0] & qb_av[0]).detach().cpu().numpy(),
        )

    def _relation_query_rows(self, direction: str, query_id: str) -> list[int]:
        if direction == "r2e":
            return [
                self.index.protein_index[value]
                for value in self.known_by_reaction.get(query_id, set())
                if value in self.index.protein_index
            ]
        return [
            self.index.reaction_index[value]
            for value in self.known_by_protein.get(query_id, set())
            if value in self.index.reaction_index
        ]

    def _relation_query_coherence(
        self,
        direction: str,
        rows: list[int],
    ) -> float:
        if not rows:
            return 0.0
        matrix = (
            self.index.protein_embeddings
            if direction == "r2e"
            else self.index.reaction_embeddings
        )
        selected = matrix.index_select(
            0, torch.as_tensor(rows, dtype=torch.long, device=self.index.device)
        )
        return float(selected.mean(0).norm().item())

    def _relation_features(
        self,
        direction: str,
        query_id: str,
        broad_z: np.ndarray,
        ctx: np.ndarray,
        comp_a: np.ndarray,
        comp_b: np.ndarray,
        available: np.ndarray,
        both_available: np.ndarray,
    ) -> np.ndarray:
        base = np.asarray(broad_z, dtype=np.float64).copy()
        context = np.asarray(ctx, dtype=np.float64).copy()
        rows = self._relation_query_rows(direction, query_id)
        if rows:
            idx = np.asarray(rows, dtype=np.int64)
            floor = float(np.min(base) - 100.0)
            base[idx] = floor
            context[idx] = 0.0

        order100 = self._relation_top_rows(base, 100)
        order10 = order100[: min(10, len(order100))]
        ctx100 = self._relation_top_rows(context, 100)
        ctx10 = ctx100[: min(10, len(ctx100))]
        ctx20 = ctx100[: min(20, len(ctx100))]

        base_sorted = np.sort(base[order100])[::-1]
        context_values = (
            np.sort(context[available])[::-1]
            if bool(available.any())
            else np.asarray([0.0], dtype=np.float64)
        )
        broad_rank = np.empty(len(base), dtype=np.int64)
        broad_order = np.argsort(-base, kind="stable")
        broad_rank[broad_order] = np.arange(1, len(base) + 1)
        ctx20_logrank = (
            float(np.mean(np.log1p(broad_rank[ctx20])))
            if len(ctx20)
            else math.log1p(len(base))
        )
        coherence = self._relation_query_coherence(direction, rows)
        features = np.asarray(
            [
                math.log1p(len(rows)),
                coherence,
                float(available.mean()),
                float(base_sorted[0] - base_sorted[1]) if len(base_sorted) > 1 else 0.0,
                float(base_sorted[0] - base_sorted[min(9, len(base_sorted) - 1)]),
                float(base_sorted[0] - base_sorted[-1]),
                (
                    float(context_values[0] - context_values[1])
                    if len(context_values) > 1
                    else 0.0
                ),
                float(np.mean(context_values[: min(10, len(context_values))])),
                float(np.mean(context_values[: min(100, len(context_values))])),
                self._relation_safe_corr(base, context, available),
                float(
                    len(set(order10.tolist()) & set(ctx10.tolist()))
                    / max(1, len(order10))
                ),
                float(
                    len(set(order100.tolist()) & set(ctx100.tolist()))
                    / max(1, len(order100))
                ),
                self._relation_safe_corr(comp_a, comp_b, both_available),
                ctx20_logrank,
            ],
            dtype=np.float64,
        )
        return features

    def _relation_authority(
        self,
        direction: str,
        query_id: str,
        broad_z: np.ndarray,
        ctx: np.ndarray,
        comp_a: np.ndarray,
        comp_b: np.ndarray,
        available: np.ndarray,
        both_available: np.ndarray,
    ) -> tuple[float, float, float]:
        asset = self.relation_gate["directions"][direction]
        if tuple(asset["feature_names"]) != RELATION_FEATURES:
            raise RuntimeError("adaptive relation gate feature contract mismatch")
        features = self._relation_features(
            direction,
            query_id,
            broad_z,
            ctx,
            comp_a,
            comp_b,
            available,
            both_available,
        )
        x = asset["scaler"].transform(features[None, :])
        probability = float(asset["permission_model"].predict_proba(x)[0, 1])
        strength = float(np.expm1(asset["strength_model"].predict(x)[0]))
        max_alpha = float(max(asset["alpha_support"]))
        strength = float(np.clip(strength, 0.0, max_alpha))
        alpha = float(
            np.clip(
                probability ** float(asset["permission_power"]) * strength,
                0.0,
                max_alpha,
            )
        )
        return alpha, probability, strength

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
            broad_tensor = (
                self.index.reaction_embeddings[qrow] @ self.index.protein_embeddings.T
            ).float()
            broad_full = (
                broad_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            broad_std = max(float(broad_full.std()), 1e-8)
            broad_z_full = (broad_full - float(broad_full.mean())) / broad_std
            ctx, ctx_a, ctx_b, ctx_available, ctx_both = self._relation_components(
                "r2e", query_id
            )
            relation_weight, relation_permission, relation_strength = (
                self._relation_authority(
                    "r2e",
                    query_id,
                    broad_z_full,
                    ctx,
                    ctx_a,
                    ctx_b,
                    ctx_available,
                    ctx_both,
                )
            )

            training_positive_ids = set(self.known_by_reaction.get(query_id, set()))
            requested_support_ids = tuple(
                dict.fromkeys(
                    str(value)
                    for value in (payload.get("known_enzyme_ids") or [])
                    if str(value)
                )
            )
            episodic_requested_ids = tuple(
                value for value in requested_support_ids
                if value not in training_positive_ids
            )
            external_support_embeddings, episodic_external_errors = (
                self._external_protein_support_embeddings(
                    payload,
                    episodic_requested_ids,
                )
            )
            episodic_effective_ids = tuple(
                value for value in episodic_requested_ids
                if value in self.index.protein_index
                or value in external_support_embeddings
            )
            episodic_missing_ids = tuple(
                value for value in episodic_requested_ids
                if value not in self.index.protein_index
                and value not in external_support_embeddings
            )
            train_score_z = broad_z_full + relation_weight * ctx
            episodic_memory = build_episodic_memory(
                query_embedding=self.index.reaction_embeddings[qrow],
                candidate_embeddings=self.index.protein_embeddings,
                candidate_ids=self.index.protein_ids,
                candidate_index=self.index.protein_index,
                requested_support_ids=requested_support_ids,
                training_positive_ids=training_positive_ids,
                entity_seen_mask=self._protein_relation_seen,
                base_score=train_score_z,
                train_memory_weight=relation_weight,
                external_support_embeddings=external_support_embeddings,
            )
            episodic_weight = 0.0
            episodic_raw_weight = 0.0
            episodic_delta = np.zeros_like(broad_z_full)
            episodic_effective_count = 0.0
            episodic_support_coherence = 0.0
            if episodic_memory is not None:
                episodic_weight, episodic_raw_weight = episodic_authority(
                    self.episodic_gate,
                    "r2e",
                    episodic_memory.features,
                )
                episodic_delta = episodic_memory.score - broad_z_full
                episodic_effective_count = episodic_memory.effective_support_count
                episodic_support_coherence = episodic_memory.support_coherence

            retrieval_score = broad_full + broad_std * (
                relation_weight * ctx + episodic_weight * episodic_delta
            )
            full = torch.as_tensor(
                retrieval_score,
                dtype=broad_tensor.dtype,
                device=broad_tensor.device,
            )
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
            _, indices = torch.topk(full, k=k, largest=True, sorted=True)
            top = indices.detach().cpu().numpy().astype(np.int64, copy=False)
            candidates = [self.index.protein_ids[int(row)] for row in top]
            core = broad_full[top]
            core_std = max(float(core.std()), 1e-6)
            core_z = (core - float(core.mean())) / core_std
            relation_local = relation_weight * (broad_std / core_std) * ctx[top]
            episodic_local = (
                episodic_weight
                * (broad_std / core_std)
                * episodic_delta[top]
            )

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
            final_score = (
                core_z
                + relation_local
                + episodic_local
                + functional_weight * fz
                + geometry_weight * geometry
                + specialist
            )
            local_order = self._pocket_reorder(query_id, candidates, final_score)
            ordered_rows = top[local_order]
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
                "route_id": (
                    "bridge-final-r2e-v1+fewshot"
                    if episodic_effective_ids
                    else "bridge-final-r2e-v1"
                ),
                "route_version": "bridge-final-production-v1",
                "ranking_objective": str(
                    payload.get("ranking_objective")
                    or ("top3" if top_k <= 3 else "top10" if top_k <= 10 else "top20")
                ),
                "model_bundle_version": self.version,
                "score_source": (
                    "broad+gated_experts+reciprocal_context+episodic_memory"
                    if episodic_effective_ids
                    else "broad+gated_experts+reciprocal_context"
                ),
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
                "shot_mode": (
                    "episodic_few_shot"
                    if episodic_effective_ids
                    else "validation_frozen_adaptive_relation_context"
                ),
                "relation_context_weight": relation_weight,
                "relation_permission_probability": relation_permission,
                "relation_conditional_strength": relation_strength,
                "train_memory_weight": relation_weight,
                "train_memory_relation_count": len(training_positive_ids),
                "requested_seed_count": len(requested_support_ids),
                "training_seed_count": len(requested_support_ids) - len(episodic_requested_ids),
                "episodic_support_count": len(episodic_effective_ids),
                "episodic_external_support_count": len(external_support_embeddings),
                "episodic_missing_support_count": len(episodic_missing_ids),
                "episodic_external_errors": episodic_external_errors,
                "episodic_memory_applied": bool(
                    episodic_effective_ids and episodic_weight > 0.0
                ),
                "episodic_memory_weight": episodic_weight,
                "episodic_memory_raw_weight": episodic_raw_weight,
                "episodic_effective_support_count": episodic_effective_count,
                "episodic_support_coherence": episodic_support_coherence,
                "functional_weight": functional_weight,
                "geometry_weight": geometry_weight,
                **specialist_audit,
            },
            "candidates": result,
        }

    def _rank_reactions_joint_query_v4(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Opt-in, frozen E2R route with the same protocol as its heldout test.

        The existing production E2R and episodic few-shot paths are unaffected.
        This route masks clean2023 relations and reorders only the full-universe
        Broad head of 1000; unsupported scenarios fail explicitly rather than
        silently falling back to a different ranking model.
        """
        from projects.active.bridge.runtime.e2r_query_gate import (
            FrozenE2RSurface, predict_joint_route, reorder_head,
        )

        query_id = str(payload.get("enzyme_id") or "").strip()
        if query_id not in self.index.protein_index:
            raise KeyError(query_id)
        top_k = max(1, int(payload.get("top_k") or 10))
        if top_k > 1000:
            raise ValueError("joint-query-v4 supports top_k up to Broad Top1000 only")
        if payload.get("mask_clean2023") is not True:
            raise ValueError("joint-query-v4 requires explicit mask_clean2023=true")
        for name in ("known_reaction_ids", "mask_reaction_ids", "candidate_ids", "external_reaction_support_path"):
            if payload.get(name):
                raise ValueError(f"joint-query-v4 does not support {name}")
        with self._lock, torch.no_grad():
            if not hasattr(self, "_joint_e2r_v4"):
                if not E2R_JOINT_GATE_V4.exists():
                    raise FileNotFoundError(
                        f"joint-query-v4 model asset unavailable: {E2R_JOINT_GATE_V4}"
                    )
                with E2R_JOINT_GATE_V4.open("rb") as handle:
                    self._joint_e2r_v4 = pickle.load(handle)
                if self._joint_e2r_v4.get("schema") != "bridge-e2r-joint-query-route-v4":
                    raise RuntimeError("Unsupported E2R joint query gate asset schema")
                self._joint_e2r_surface_v4 = FrozenE2RSurface(self)
            surface = self._joint_e2r_surface_v4
            item = surface.score(query_id)
            route, authority, predictions = predict_joint_route(
                self._joint_e2r_v4, item["features"], item
            )
            order = reorder_head(
                broad_order=item["order"],
                lexical=surface.lex,
                core=item["core"],
                functional=item["functional"],
                structure=item["structure"],
                relation=item["relation"],
                authority=authority,
            )
            score = (
                item["core"]
                + float(authority[0]) * item["functional"]
                + float(authority[1]) * item["structure"]
                + float(authority[2]) * item["relation"]
            )
            result = [
                {
                    "rank": int(rank),
                    "candidate_id": self.index.reaction_ids[int(row)],
                    "score": float(score[int(row)]),
                    "selection_source": "bridge_joint_query_v4",
                }
                for rank, row in enumerate(order[:top_k], start=1)
            ]
        return {
            "query": {
                "query_id": query_id,
                "direction": "enzyme_to_reaction",
                "route_id": "bridge-e2r-joint-query-v4",
                "route_version": "bridge-e2r-joint-query-v4-experimental",
                "ranking_objective": str(payload.get("ranking_objective") or
                                         ("top3" if top_k<=3 else "top10" if top_k<=10 else "top20")),
                "model_bundle_version": self._joint_e2r_v4["schema"],
                "score_source": "frozen_broad+query_level_joint_evidence_authority",
                "candidate_universe": "general_merged",
                "candidate_universe_size": len(self.index.reaction_ids),
                "broad_head_reranked": 1000,
                "mask_clean2023": True,
                "requested_top_k": top_k,
                "evidence_route": route,
                "evidence_authority": {
                    "functional":float(authority[0]),
                    "structure":float(authority[1]),
                    "long_term_relation":float(authority[2]),
                },
                "predicted_route_gain": float(predictions[route]),
                "production_default_unchanged": True,
                "episodic_memory_applied": False,
            },
            "candidates": result,
        }

    def _rank_reactions_inductive_relation(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Production E2R ranking for previously unreported catalytic links.

        The query gate, biochemical feature experts, candidate universe and
        filtered Broad-Top1000 protocol are frozen. The two-sided catalytic
        evidence is read only from the clean2023 association graph.
        """
        from projects.active.bridge.runtime.e2r_query_gate import (
            FrozenE2RSurface, predict_joint_route,
        )
        from projects.active.bridge.runtime.inductive_relation import (
            InductiveE2RRelation, InductiveRelationConfig,
        )

        query_id = str(payload.get("enzyme_id") or "").strip()
        if query_id not in self.index.protein_index:
            raise KeyError(query_id)
        top_k = max(1, int(payload.get("top_k") or 10))
        if top_k > 1000:
            raise ValueError("E2R catalytic relation ranking is defined over Broad Top1000")
        if payload.get("mask_clean2023") is not True:
            raise ValueError("Discovery ranking requires mask_clean2023=true")
        for name in (
            "known_reaction_ids", "mask_reaction_ids", "candidate_ids",
            "external_reaction_support_path",
        ):
            if payload.get(name):
                raise ValueError(f"Catalytic discovery route does not support {name}")
        with self._lock, torch.no_grad():
            if not hasattr(self, "_joint_e2r_v4"):
                if not E2R_JOINT_GATE_V4.is_file():
                    raise FileNotFoundError(E2R_JOINT_GATE_V4)
                with E2R_JOINT_GATE_V4.open("rb") as handle:
                    self._joint_e2r_v4 = pickle.load(handle)
                if self._joint_e2r_v4.get("schema") != "bridge-e2r-joint-query-route-v4":
                    raise RuntimeError("Unexpected joint biochemical expert gate schema")
                self._joint_e2r_surface_v4 = FrozenE2RSurface(self)
            if not hasattr(self, "_inductive_e2r_context"):
                if not E2R_INDUCTIVE_CONTEXT.is_file():
                    raise FileNotFoundError(E2R_INDUCTIVE_CONTEXT)
                config = json.loads(E2R_INDUCTIVE_CONTEXT.read_text())
                if config.get("schema") != "bridge-e2r-inductive-bipartite-graph-production-v1":
                    raise RuntimeError("Unexpected inductive biochemical evidence asset schema")
                weights = config["coefficients"]
                controls = config["neighborhood"]
                self._inductive_e2r_asset = config
                self._inductive_e2r_context = InductiveE2RRelation(
                    self, InductiveRelationConfig(
                        observed_coefficient=float(weights["observed_links"]),
                        inferred_coefficient=float(weights["reaction_analogy"]),
                        nearest_proteins=int(controls["protein_neighbors"]),
                        nearest_reactions=int(controls["reaction_neighbors"]),
                        reaction_neighbor_pool=int(controls["reaction_retrieval_pool"]),
                        protein_attention_temperature=float(controls["protein_temperature"]),
                        reaction_attention_temperature=float(controls["reaction_temperature"]),
                        prior_normalizer=float(controls["observation_prior_factor"]),
                        reaction_min_cosine=float(controls["reaction_confidence_floor"]),
                    )
                )
            surface = self._joint_e2r_surface_v4
            item = surface.score(query_id)
            route, authority, _ = predict_joint_route(
                self._joint_e2r_v4, item["features"], item
            )
            top = item["top"]
            base = (
                item["core"][top]
                + float(authority[0]) * item["functional"][top]
                + float(authority[1]) * item["structure"][top]
                + float(authority[2]) * item["relation"][top]
            )
            relation_evidence = self._inductive_e2r_context.correction(
                query_id, top, float(np.std(item["broad"]))
            )
            local_score = base + relation_evidence
            local_order = np.lexsort((surface.lex[top], -local_score))
            ordered_rows = top[local_order]
            selected = [
                {
                    "rank": i + 1,
                    "candidate_id": self.index.reaction_ids[int(row)],
                    "score": float(local_score[int(local)]),
                    "selection_source": "bridge_inductive_relation",
                }
                for i, (row, local) in enumerate(
                    zip(ordered_rows[:top_k], local_order[:top_k])
                )
            ]
        return {
            "query": {
                "query_id": query_id,
                "direction": "enzyme_to_reaction",
                "route_id": "bridge-e2r-inductive-bipartite-graph",
                "route_version": "bridge-e2r-inductive-relation-production-v1",
                "model_bundle_version": self._inductive_e2r_asset["schema"],
                "score_source": (
                    "broad+query_gated_biochemical_experts"
                    "+documented_catalyst_neighbors+reaction_analogy"
                ),
                "candidate_universe": "general_merged",
                "candidate_universe_size": len(self.index.reaction_ids),
                "broad_head_reranked": 1000,
                "mask_clean2023": True,
                "requested_top_k": top_k,
                "ranking_objective": str(
                    payload.get("ranking_objective") or
                    ("top3" if top_k <= 3 else "top10" if top_k <= 10 else "top20")
                ),
                "evidence_route": route,
                "evidence_authority": {
                    "functional": float(authority[0]),
                    "structure": float(authority[1]),
                    "long_term_relation": float(authority[2]),
                },
                "observed_enzyme_neighbors": self._inductive_e2r_asset[
                    "neighborhood"
                ]["protein_neighbors"],
                "inductive_reaction_neighbors": self._inductive_e2r_asset[
                    "neighborhood"
                ]["reaction_neighbors"],
                "episodic_memory_applied": False,
            },
            "candidates": selected,
        }

    def rank_reactions(self, payload: dict[str, Any]) -> dict[str, Any]:
        gate_profile = str(payload.get("e2r_gate_profile") or "").strip()
        if gate_profile == "joint-query-v4":
            return self._rank_reactions_joint_query_v4(payload)
        if gate_profile == "inductive-bipartite":
            return self._rank_reactions_inductive_relation(payload)
        if gate_profile:
            raise ValueError(f"Unknown E2R gate profile: {gate_profile}")
        # Registered-enzyme discovery queries receive the validated relation
        # model automatically. Existing annotation/episodic/custom-candidate
        # workflows continue to use their full-universe ranking contract.
        can_discover = (
            payload.get("mask_clean2023") is True
            and max(1, int(payload.get("top_k") or 10)) <= 1000
            and not any(payload.get(k) for k in (
                "known_reaction_ids", "mask_reaction_ids",
                "candidate_ids", "external_reaction_support_path"
            ))
        )
        if can_discover:
            return self._rank_reactions_inductive_relation(payload)
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
            broad_std = max(float(broad.std()), 1e-8)
            broad_z = (broad - float(broad.mean())) / broad_std
            ctx, ctx_a, ctx_b, ctx_available, ctx_both = self._relation_components(
                "e2r", query_id
            )
            relation_weight, relation_permission, relation_strength = (
                self._relation_authority(
                    "e2r",
                    query_id,
                    broad_z,
                    ctx,
                    ctx_a,
                    ctx_b,
                    ctx_available,
                    ctx_both,
                )
            )
            training_positive_ids = set(self.known_by_protein.get(query_id, set()))
            requested_support_ids = tuple(
                dict.fromkeys(
                    str(value)
                    for value in (payload.get("known_reaction_ids") or [])
                    if str(value)
                )
            )
            episodic_requested_ids = tuple(
                value for value in requested_support_ids
                if value not in training_positive_ids
            )
            external_support_embeddings, episodic_external_errors = (
                self._external_reaction_support_embeddings(
                    payload,
                    episodic_requested_ids,
                )
            )
            episodic_effective_ids = tuple(
                value for value in episodic_requested_ids
                if value in self.index.reaction_index
                or value in external_support_embeddings
            )
            episodic_missing_ids = tuple(
                value for value in episodic_requested_ids
                if value not in self.index.reaction_index
                and value not in external_support_embeddings
            )
            train_score_z = broad_z + relation_weight * ctx
            episodic_memory = build_episodic_memory(
                query_embedding=self.index.protein_embeddings[prow],
                candidate_embeddings=self.index.reaction_embeddings,
                candidate_ids=self.index.reaction_ids,
                candidate_index=self.index.reaction_index,
                requested_support_ids=requested_support_ids,
                training_positive_ids=training_positive_ids,
                entity_seen_mask=self._reaction_relation_seen,
                base_score=train_score_z,
                train_memory_weight=relation_weight,
                external_support_embeddings=external_support_embeddings,
            )
            episodic_weight = 0.0
            episodic_raw_weight = 0.0
            episodic_delta = np.zeros_like(broad_z)
            episodic_effective_count = 0.0
            episodic_support_coherence = 0.0
            if episodic_memory is not None:
                episodic_weight, episodic_raw_weight = episodic_authority(
                    self.episodic_gate,
                    "e2r",
                    episodic_memory.features,
                )
                episodic_delta = episodic_memory.score - broad_z
                episodic_effective_count = episodic_memory.effective_support_count
                episodic_support_coherence = episodic_memory.support_coherence

            core_scale = float(self.e2r_core_cal["scale"])
            core = (
                broad - float(self.e2r_core_cal["center"])
            ) / core_scale
            relation_core = relation_weight * (broad_std / core_scale) * ctx
            episodic_core = (
                episodic_weight
                * (broad_std / core_scale)
                * episodic_delta
            )

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
                + relation_core
                + episodic_core
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
                "route_id": (
                    "bridge-final-e2r-v1+fewshot"
                    if episodic_effective_ids
                    else "bridge-final-e2r-v1"
                ),
                "route_version": "bridge-final-production-v1",
                "ranking_objective": str(
                    payload.get("ranking_objective")
                    or ("top3" if top_k <= 3 else "top10" if top_k <= 10 else "top20")
                ),
                "model_bundle_version": self.version,
                "score_source": (
                    "broad+gated_experts+reciprocal_context+episodic_memory"
                    if episodic_effective_ids
                    else "broad+gated_experts+reciprocal_context"
                ),
                "candidate_universe": "general_merged",
                "candidate_universe_size": len(self.index.reaction_ids),
                "requested_top_k": top_k,
                "shot_mode": (
                    "episodic_few_shot"
                    if episodic_effective_ids
                    else "validation_frozen_adaptive_relation_context"
                ),
                "relation_context_weight": relation_weight,
                "relation_permission_probability": relation_permission,
                "relation_conditional_strength": relation_strength,
                "train_memory_weight": relation_weight,
                "train_memory_relation_count": len(training_positive_ids),
                "requested_seed_count": len(requested_support_ids),
                "training_seed_count": len(requested_support_ids) - len(episodic_requested_ids),
                "episodic_support_count": len(episodic_effective_ids),
                "episodic_external_support_count": len(external_support_embeddings),
                "episodic_missing_support_count": len(episodic_missing_ids),
                "episodic_external_errors": episodic_external_errors,
                "episodic_memory_applied": bool(
                    episodic_effective_ids and episodic_weight > 0.0
                ),
                "episodic_memory_weight": episodic_weight,
                "episodic_memory_raw_weight": episodic_raw_weight,
                "episodic_effective_support_count": episodic_effective_count,
                "episodic_support_coherence": episodic_support_coherence,
            },
            "candidates": result,
        }

    def rank(self, command: str, payload: dict[str, Any]) -> dict[str, Any]:
        if command == "rank-enzymes":
            return self.rank_enzymes(payload)
        if command == "rank-reactions":
            return self.rank_reactions(payload)
        raise ValueError(f"unsupported BRIDGE command: {command}")
