from __future__ import annotations

import argparse
import json
from types import MethodType

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime import final_system as fs
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
R2E_MEMBERSHIP = ROOT / "results/bridge_layered_v4_r2e_cage/pair_membership.csv.gz"
E2R_MEMBERSHIP = ROOT / "results/bridge_layered_v4_e2r_cage/pair_membership.csv.gz"
R2E_BUDGET = ROOT / "results/bridge_layered_v4_candidates/r2e_query_budget.csv"
E2R_BUDGET = ROOT / "results/bridge_layered_v4_candidates/e2r_query_budget.csv"
PROTEIN_SEQUENCES = ROOT / "data/catalyst_candidate_universes/general_merged/protein_sequences.tsv"
EXTERNAL_ESMC = ROOT / "results/bridge_layered_v4_cage_gate_bridge/external_protein_esmc"
OUT = ROOT / "results/bridge_layered_v4_cage_gate_bridge"


def parse_targets(value: str) -> set[str]:
    return {x for x in str(value).split(";") if x}


def clean_sequence(value: object) -> str:
    return "".join(str(value).upper().split()).rstrip("*")


def metric_row(
    order: list[str],
    positive_candidates: set[str],
    positive_total: int,
) -> dict[str, float]:
    ranks = [i + 1 for i, cid in enumerate(order) if cid in positive_candidates]
    best = min(ranks) if ranks else None
    return {
        "rr": 0.0 if best is None else 1.0 / best,
        "hit3": float(best is not None and best <= 3),
        "hit10": float(best is not None and best <= 10),
        "hit100": float(best is not None and best <= 100),
        "positive_recall": (
            float(len(positive_candidates) / positive_total)
            if positive_total
            else 0.0
        ),
    }


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "queries": int(len(frame)),
        "mrr": float(frame.rr.mean()),
        "hit3": float(frame.hit3.mean()),
        "hit10": float(frame.hit10.mean()),
        "hit100": float(frame.hit100.mean()),
        "macro_positive_recall": float(frame.positive_recall.mean()),
        "query_hit": float((frame.positive_recall > 0).mean()),
    }


def _safe_r2e_specialists(
    self: FinalBridgeRuntime,
    query_id: str,
    candidate_ids: list[str],
    *,
    original_protein_count: int,
    alias_source: dict[str, str],
) -> tuple[np.ndarray, dict[str, object]]:
    """Current specialist semantics with missing external TPS evidence kept neutral."""

    score = np.zeros(len(candidate_ids), dtype=np.float64)
    audit: dict[str, object] = {
        "family": "none",
        "family_weight": 0.0,
        "tps_weight": 0.0,
    }

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
            family in fs.FAMILIES
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
                residual = np.zeros(len(candidate_ids), dtype=np.float64)
                available = np.zeros(len(candidate_ids), dtype=bool)
                column = f"{family}_normalized_logit_response"
                family_value = {
                    str(pid): float(raw)
                    for pid, raw in group[["protein_id", column]].itertuples(index=False)
                }
                for row, candidate_id in enumerate(candidate_ids):
                    evidence_id = alias_source.get(str(candidate_id), str(candidate_id))
                    raw = family_value.get(evidence_id)
                    if raw is None:
                        continue
                    residual[row] = raw
                    available[row] = True
                residual = fs._z(residual, available)
                columns = fs._family_feature_columns(
                    family,
                    self.family_gate["feature_suffixes"],
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
                weight = raw_weight * fs.FAMILY_SCALE[family]
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
            evidence_ids = [
                alias_source.get(str(pid), str(pid))
                for pid in candidate_ids
            ]
            available = np.asarray(
                [
                    self.index.protein_index.get(pid, original_protein_count)
                    < original_protein_count
                    for pid in evidence_ids
                ],
                dtype=bool,
            )
            if int(available.sum()) >= 2:
                qz = np.asarray(
                    self.tps_z[self.tps_index[query_id]],
                    dtype=np.float32,
                )
                supported_ids = [
                    pid
                    for pid, ok in zip(evidence_ids, available, strict=True)
                    if ok
                ]
                rows = np.asarray(
                    [self.index.protein_index[pid] for pid in supported_ids],
                    dtype=np.int64,
                )
                raw = np.zeros(len(candidate_ids), dtype=np.float64)
                raw[available] = (
                    np.asarray(self.tps_protein[rows], dtype=np.float32) @ qz
                ).astype(np.float64)
                observed = raw[available]
                sorted_values = np.sort(observed)
                values = {
                    "tps_ref_max_cosine": float(query["tps_ref_max_cosine"]),
                    "tps_ref_top5_mean": float(query["tps_ref_top5_mean"]),
                    "tps_ref_top20_mean": float(query["tps_ref_top20_mean"]),
                    "tps_ref_margin_1_2": float(query["tps_ref_margin_1_2"]),
                    "tps_ref_softmax_entropy": float(
                        query["tps_ref_softmax_entropy"]
                    ),
                    "tps_score_mean": float(observed.mean()),
                    "tps_score_std": float(observed.std()),
                    "tps_score_top1_margin": float(
                        sorted_values[-1] - sorted_values[-2]
                    ),
                    "tps_score_top20_mean": float(
                        sorted_values[-min(20, len(sorted_values)) :].mean()
                    ),
                }
                features = np.asarray(
                    [values[key] for key in fs.TPS_FEATURES],
                    dtype=float,
                )[None, :]
                raw_weight = float(
                    np.clip(
                        self.tps_gate["regressor"].predict(features)[0],
                        0.0,
                        0.5,
                    )
                )
                weight = raw_weight * float(
                    self.tps_gate["strength_scale"]
                )
                if weight > 0:
                    score += weight * fs._z(raw, available)
                    audit["tps_weight"] = weight
    return score, audit


def _safe_pocket_reorder(
    self: FinalBridgeRuntime,
    query_id: str,
    candidate_ids: list[str],
    base_score: np.ndarray,
    *,
    alias_source: dict[str, str],
) -> np.ndarray:
    order = np.argsort(-base_score, kind="stable")
    group = self.pocket_groups.get(str(query_id))
    if group is None or len(group) < 2:
        return order
    inverse = np.empty(len(order), dtype=np.int32)
    inverse[order] = np.arange(1, len(order) + 1)
    pocket_value = {
        str(pid): float(raw)
        for pid, raw in group[["protein_id", "pocket_interaction_score"]].itertuples(index=False)
    }
    rows: list[int] = []
    values: list[float] = []
    for row, candidate_id in enumerate(candidate_ids):
        evidence_id = alias_source.get(str(candidate_id), str(candidate_id))
        raw = pocket_value.get(evidence_id)
        if raw is None or int(inverse[row]) <= fs.POCKET_PREFIX:
            continue
        rows.append(row)
        values.append(raw)
    if len(rows) < 2:
        return order
    locs = np.asarray(rows, dtype=np.int64)
    mix = fs._z(base_score[locs]) + fs.POCKET_ALPHA * fs._z(
        np.asarray(values, dtype=np.float64)
    )
    slots = np.sort(inverse[locs])
    ranked = locs[np.argsort(-mix, kind="stable")]
    final_inverse = inverse.copy()
    final_inverse[ranked] = slots
    return np.argsort(final_inverse, kind="stable")


def extend_r2e_runtime(
    rt: FinalBridgeRuntime,
    membership: pd.DataFrame,
) -> dict[str, object]:
    """Append CAGE-only proteins to the frozen Broad space without changing the gate."""

    original_count = len(rt.index.protein_ids)
    unique = membership[
        ["logical_candidate_id", "sequence"]
    ].drop_duplicates("logical_candidate_id")
    missing = unique[
        ~unique.logical_candidate_id.astype(str).isin(rt.index.protein_index)
    ].copy()
    if missing.empty:
        return {
            "original_bridge_proteins": original_count,
            "appended_exact_sequence_aliases": 0,
            "appended_open_world_proteins": 0,
            "external_feature_failures": 0,
        }

    base = pd.read_csv(
        PROTEIN_SEQUENCES,
        sep="\t",
        dtype=str,
    ).fillna("")
    sequence_to_ids: dict[str, list[str]] = {}
    for protein_id, sequence in base[
        ["protein_id", "sequence"]
    ].itertuples(index=False):
        protein_id = str(protein_id)
        if protein_id not in rt.index.protein_index:
            continue
        seq = clean_sequence(sequence)
        if seq:
            sequence_to_ids.setdefault(seq, []).append(protein_id)

    alias_source: dict[str, str] = {}
    external_ids: list[str] = []
    external_sequence: dict[str, str] = {}
    for candidate_id, sequence in missing.itertuples(index=False):
        candidate_id = str(candidate_id)
        seq = clean_sequence(sequence)
        matches = sorted(set(sequence_to_ids.get(seq, [])))
        if matches:
            alias_source[candidate_id] = matches[0]
        else:
            external_ids.append(candidate_id)
            external_sequence[candidate_id] = seq

    external_features: dict[str, np.ndarray] = {}
    if external_ids:
        entries_path = EXTERNAL_ESMC / "entries.csv"
        matrix_path = EXTERNAL_ESMC / "embeddings.npy"
        if not entries_path.exists() or not matrix_path.exists():
            raise RuntimeError(
                "open-world ESM-C extraction is incomplete: "
                f"{entries_path} / {matrix_path}"
            )
        entries = pd.read_csv(entries_path, dtype=str).fillna("")
        if "row" in entries.columns:
            entries["row"] = pd.to_numeric(entries.row).astype(int)
            entries = entries.sort_values("row")
        id_col = "Entry" if "Entry" in entries.columns else "protein_id"
        matrix = np.load(matrix_path).astype(np.float32)
        if len(entries) != len(matrix):
            raise RuntimeError(
                "external ESM-C entries/matrix length mismatch"
            )
        for row, identifier in enumerate(entries[id_col].astype(str)):
            if identifier in external_sequence:
                value = np.asarray(matrix[row], dtype=np.float32)
                norm = max(float(np.linalg.norm(value)), 1e-12)
                external_features[identifier] = value / norm
        absent = sorted(set(external_ids) - set(external_features))
        if absent:
            raise RuntimeError(
                "open-world ESM-C extraction misses CAGE candidates: "
                f"{absent[:20]} (n={len(absent)})"
            )

    append_ids = sorted(alias_source) + sorted(external_ids)
    latent: list[torch.Tensor] = []

    if alias_source:
        alias_rows = torch.as_tensor(
            [
                rt.index.protein_index[alias_source[identifier]]
                for identifier in sorted(alias_source)
            ],
            dtype=torch.long,
            device=rt.index.device,
        )
        latent.append(
            rt.index.protein_embeddings.index_select(0, alias_rows)
        )

    if external_ids:
        ordered_external = sorted(external_ids)
        feature_matrix = np.stack(
            [external_features[identifier] for identifier in ordered_external]
        ).astype(np.float32)
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(feature_matrix), 1024):
                values = torch.as_tensor(
                    feature_matrix[start : start + 1024],
                    dtype=torch.float32,
                    device=rt.index.device,
                )
                chunks.append(rt.index.model.encode_proteins(values))
        latent.append(torch.cat(chunks, dim=0))

    appended = torch.cat(latent, dim=0)
    rt.index.protein_embeddings = torch.cat(
        [rt.index.protein_embeddings, appended],
        dim=0,
    )
    rt.index.protein_ids = list(rt.index.protein_ids) + append_ids
    rt.index.protein_index = {
        identifier: i
        for i, identifier in enumerate(rt.index.protein_ids)
    }

    relation_dim = rt.pproto.shape[1]
    rt.pproto = torch.cat(
        [
            rt.pproto,
            torch.zeros(
                (len(append_ids), relation_dim),
                dtype=rt.pproto.dtype,
                device=rt.pproto.device,
            ),
        ],
        dim=0,
    )
    rt.pmask = torch.cat(
        [
            rt.pmask,
            torch.zeros(
                len(append_ids),
                dtype=rt.pmask.dtype,
                device=rt.pmask.device,
            ),
        ],
        dim=0,
    )
    rt._protein_relation_seen = np.concatenate(
        [
            rt._protein_relation_seen,
            np.zeros(len(append_ids), dtype=bool),
        ]
    )

    rt._r2e_functional_p_row = np.concatenate(
        [
            rt._r2e_functional_p_row,
            np.asarray(
                [
                    rt.functional_r2e.p_index.get(
                        alias_source.get(identifier, identifier), -1
                    )
                    for identifier in append_ids
                ],
                dtype=np.int64,
            ),
        ]
    )
    rt._clip_p_row = np.concatenate(
        [
            rt._clip_p_row,
            np.asarray(
                [
                    rt.clip.p_index.get(
                        alias_source.get(identifier, identifier), -1
                    )
                    for identifier in append_ids
                ],
                dtype=np.int64,
            ),
        ]
    )
    rt._mechanism_p_row = np.concatenate(
        [
            rt._mechanism_p_row,
            np.asarray(
                [
                    rt.mechanism.p_index.get(
                        alias_source.get(identifier, identifier), -1
                    )
                    for identifier in append_ids
                ],
                dtype=np.int64,
            ),
        ]
    )
    rt._protein_lex = rt._lex(rt.index.protein_ids)

    rt._r2e_specialists = MethodType(
        lambda self, query_id, candidate_ids: _safe_r2e_specialists(
            self,
            query_id,
            candidate_ids,
            original_protein_count=original_count,
            alias_source=alias_source,
        ),
        rt,
    )
    rt._pocket_reorder = MethodType(
        lambda self, query_id, candidate_ids, base_score: _safe_pocket_reorder(
            self,
            query_id,
            candidate_ids,
            base_score,
            alias_source=alias_source,
        ),
        rt,
    )

    return {
        "original_bridge_proteins": original_count,
        "appended_exact_sequence_aliases": len(alias_source),
        "appended_open_world_proteins": len(external_ids),
        "external_feature_failures": 0,
        "extended_bridge_proteins": len(rt.index.protein_ids),
        "appended_functional_support": int(
            sum(
                rt.functional_r2e.p_index.get(
                    alias_source.get(identifier, identifier), -1
                ) >= 0
                for identifier in append_ids
            )
        ),
        "appended_clip_support": int(
            sum(
                rt.clip.p_index.get(
                    alias_source.get(identifier, identifier), -1
                ) >= 0
                for identifier in append_ids
            )
        ),
        "appended_mechanism_support": int(
            sum(
                rt.mechanism.p_index.get(
                    alias_source.get(identifier, identifier), -1
                ) >= 0
                for identifier in append_ids
            )
        ),
    }


def _r2e_retrieval_tail(
    rt: FinalBridgeRuntime,
    reaction_id: str,
    candidate_ids: list[str],
    train_memory_weight: float,
) -> list[str]:
    qrow = rt.index.reaction_index[reaction_id]
    with torch.no_grad():
        broad_t = (
            rt.index.reaction_embeddings[qrow]
            @ rt.index.protein_embeddings.T
        ).float()
    broad = broad_t.detach().cpu().numpy().astype(
        np.float64,
        copy=False,
    )
    bstd = max(float(broad.std()), 1e-8)
    ctx, _, _, _, _ = rt._relation_components("r2e", reaction_id)
    retrieval = broad + bstd * float(train_memory_weight) * ctx
    rows = np.asarray(
        [rt.index.protein_index[x] for x in candidate_ids],
        dtype=np.int64,
    )
    values = retrieval[rows]
    lexical = np.asarray(candidate_ids, dtype=object)
    order = np.lexsort((lexical, -values))
    return [candidate_ids[int(i)] for i in order]


def evaluate_r2e(
    rt: FinalBridgeRuntime,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    usecols = [
        "route",
        "reaction_id",
        "logical_candidate_id",
        "target_protein_ids",
        "sequence",
    ]
    membership = pd.read_csv(
        R2E_MEMBERSHIP,
        dtype=str,
        usecols=usecols,
    ).fillna("")
    membership = membership[
        membership.route.eq("enzymecage")
    ].copy()
    membership = membership.drop_duplicates(
        ["reaction_id", "logical_candidate_id"],
        keep="first",
    )
    extension_audit = extend_r2e_runtime(rt, membership)
    groups = {
        q: g
        for q, g in membership.groupby("reaction_id", sort=False)
    }

    targets = pd.read_csv(
        TARGETS,
        dtype=str,
    ).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    positives = (
        targets.groupby("reaction_id")
        .protein_id.agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    budget = pd.read_csv(R2E_BUDGET, dtype=str).fillna("")
    queries = budget.reaction_id.astype(str).tolist()
    if limit is not None:
        queries = queries[:limit]

    records: list[dict[str, object]] = []
    for i, q in enumerate(queries, start=1):
        pos = positives.get(q, set())
        group = groups.get(q)
        if group is None or group.empty:
            records.append(
                {
                    "reaction_id": q,
                    "candidate_count": 0,
                    "positive_count": len(pos),
                    "gate_positive_count": 0,
                    **metric_row([], set(), len(pos)),
                }
            )
            continue

        candidates = group.logical_candidate_id.astype(str).tolist()
        invalid = [
            x
            for x in candidates
            if x not in rt.index.protein_index
        ]
        if invalid:
            raise RuntimeError(
                f"R2E extended BRIDGE index misses gate candidates: {invalid[:10]}"
            )

        positive_candidates: set[str] = set()
        for rec in group.itertuples(index=False):
            if parse_targets(rec.target_protein_ids) & pos:
                positive_candidates.add(
                    str(rec.logical_candidate_id)
                )

        top_k = min(len(candidates), fs.TOPK_R2E)
        result = rt.rank_enzymes(
            {
                "reaction_id": q,
                "candidate_ids": candidates,
                "top_k": top_k,
                "mask_clean2023": False,
            }
        )
        head = [
            str(x["candidate_id"])
            for x in result["candidates"]
        ]
        if len(head) != top_k:
            raise RuntimeError(
                f"R2E BRIDGE head size mismatch for {q}: "
                f"{len(head)} != {top_k}"
            )
        if len(candidates) > len(head):
            retrieval_order = _r2e_retrieval_tail(
                rt,
                q,
                candidates,
                float(
                    result["query"].get(
                        "train_memory_weight",
                        0.0,
                    )
                ),
            )
            head_set = set(head)
            order = head + [
                x for x in retrieval_order if x not in head_set
            ]
        else:
            order = head
        if len(order) != len(candidates) or set(order) != set(candidates):
            raise RuntimeError(
                f"R2E order coverage mismatch for {q}"
            )

        records.append(
            {
                "reaction_id": q,
                "candidate_count": len(candidates),
                "positive_count": len(pos),
                "gate_positive_count": len(positive_candidates),
                **metric_row(
                    order,
                    positive_candidates,
                    len(pos),
                ),
            }
        )
        if i % 50 == 0 or i == len(queries):
            print(
                f"cage-gate-bridge r2e {i}/{len(queries)}",
                flush=True,
            )

    qf = pd.DataFrame(records)
    qf.to_csv(OUT / "r2e_query_metrics.csv", index=False)
    return {
        "metrics": summarize(qf),
        "audit": {
            "queries": len(queries),
            "native_gate_rows": int(len(membership)),
            "gate_order_used_as_base_order": False,
            "base_order": (
                "Broad score inside the frozen EnzymeCAGE gate; "
                "CAGE-only external proteins use the repository open-world "
                "ESM-C 600M -> frozen Broad encoder path"
            ),
            "upper_layer": (
                "current BRIDGE validation-frozen corrections; "
                "missing expert evidence remains neutral"
            ),
            "tail_policy": (
                "when native gate exceeds R2E rerank depth, untouched tail "
                "retains Broad+long-term-relation retrieval order"
            ),
            "runtime_extension": extension_audit,
        },
    }


def evaluate_e2r(
    rt: FinalBridgeRuntime,
    *,
    limit: int | None = None,
) -> dict[str, object]:
    usecols = ["protein_id", "reaction_id", "native_pool"]
    membership = pd.read_csv(
        E2R_MEMBERSHIP,
        dtype=str,
        usecols=usecols,
    ).fillna("")
    membership["native_pool"] = (
        membership.native_pool.astype(str).str.lower().eq("true")
    )
    membership = membership[
        membership.native_pool
    ].drop_duplicates(
        ["protein_id", "reaction_id"],
        keep="first",
    )
    groups = {
        q: g
        for q, g in membership.groupby("protein_id", sort=False)
    }

    targets = pd.read_csv(
        TARGETS,
        dtype=str,
    ).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    positives = (
        targets.groupby("protein_id")
        .reaction_id.agg(lambda x: set(map(str, x)))
        .to_dict()
    )
    budget = pd.read_csv(E2R_BUDGET, dtype=str).fillna("")
    queries = budget.protein_id.astype(str).tolist()
    if limit is not None:
        queries = queries[:limit]

    records: list[dict[str, object]] = []
    for i, q in enumerate(queries, start=1):
        pos = positives.get(q, set())
        group = groups.get(q)
        if group is None or group.empty:
            records.append(
                {
                    "protein_id": q,
                    "candidate_count": 0,
                    "positive_count": len(pos),
                    "gate_positive_count": 0,
                    **metric_row([], set(), len(pos)),
                }
            )
            continue

        candidates = group.reaction_id.astype(str).tolist()
        invalid = [
            x
            for x in candidates
            if x not in rt.index.reaction_index
        ]
        if invalid:
            raise RuntimeError(
                f"E2R CAGE gate contains reactions outside BRIDGE universe: "
                f"{invalid[:10]}"
            )
        positive_candidates = set(candidates) & pos
        result = rt.rank_reactions(
            {
                "enzyme_id": q,
                "candidate_ids": candidates,
                "top_k": len(candidates),
                "mask_clean2023": False,
            }
        )
        order = [
            str(x["candidate_id"])
            for x in result["candidates"]
        ]
        if len(order) != len(candidates) or set(order) != set(candidates):
            raise RuntimeError(
                f"E2R order coverage mismatch for {q}: "
                f"{len(order)} != {len(candidates)}"
            )
        records.append(
            {
                "protein_id": q,
                "candidate_count": len(candidates),
                "positive_count": len(pos),
                "gate_positive_count": len(positive_candidates),
                **metric_row(
                    order,
                    positive_candidates,
                    len(pos),
                ),
            }
        )
        if i % 500 == 0 or i == len(queries):
            print(
                f"cage-gate-bridge e2r {i}/{len(queries)}",
                flush=True,
            )

    qf = pd.DataFrame(records)
    qf.to_csv(OUT / "e2r_query_metrics.csv", index=False)
    return {
        "metrics": summarize(qf),
        "audit": {
            "queries": len(queries),
            "native_gate_rows": int(len(membership)),
            "gate_order_used_as_base_order": False,
            "base_order": (
                "Broad score inside the frozen symmetric EnzymeCAGE gate "
                "candidate set"
            ),
            "upper_layer": (
                "current BRIDGE validation-frozen corrections"
            ),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--direction",
        choices=("r2e", "e2r", "both"),
        default="both",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rt = FinalBridgeRuntime(device=args.device)
    result: dict[str, object] = {
        "schema": "bridge-layered-v4-cage-gate-bridge",
        "status": (
            "completed"
            if args.limit is None
            else "smoke_test"
        ),
        "protocol": {
            "mother_benchmark": "bridge_relation_unseen_max_v3",
            "strict_relation_unseen_relative_to_clean2023": True,
            "candidate_gate": (
                "frozen EnzymeCAGE native gate; gate membership only"
            ),
            "cage_gate_order_used": False,
            "ranking_semantics": (
                "Broad establishes the base order inside the CAGE gate; "
                "current BRIDGE applies validation-frozen upper-layer corrections"
            ),
            "cage_only_external_proteins": (
                "encoded through the repository open-world ESM-C 600M -> "
                "frozen Broad protein encoder; unavailable upper expert "
                "evidence remains neutral"
            ),
            "no_retraining": True,
            "no_parameter_fitting": True,
            "no_new_gate": True,
        },
    }
    if args.direction in ("r2e", "both"):
        result["r2e"] = evaluate_r2e(
            rt,
            limit=args.limit,
        )
    if args.direction in ("e2r", "both"):
        result["e2r"] = evaluate_e2r(
            rt,
            limit=args.limit,
        )

    name = (
        "summary.json"
        if args.limit is None
        else "smoke_summary.json"
    )
    output_path = OUT / name
    if args.limit is None and output_path.exists():
        previous = json.loads(output_path.read_text())
        if previous.get("schema") == result["schema"]:
            for direction in ("r2e", "e2r"):
                if direction in previous and direction not in result:
                    result[direction] = previous[direction]
    output_path.write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
