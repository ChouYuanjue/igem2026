from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.runtime.final_system import FinalBridgeRuntime, TOPK_R2E, _z

TARGETS = ROOT / "results/bridge_relation_unseen_max_v3/targets.csv"
TRAIN = ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv"
OUT = ROOT / "results/bridge_r2e_single_validation_release_v1"
GATE = ROOT / "results/bridge_single_validation_v1/r2e_relation_gate.exploratory.pkl"
FUNCTIONAL_TEMPERATURE = 1.25


def filtered_ranks(raw: list[int]) -> list[int]:
    arr = np.asarray(raw, dtype=np.int64)
    return [int(v - np.count_nonzero(arr < v)) for v in arr]


def rank_full(score: np.ndarray, row: int, lex: np.ndarray) -> int:
    value = float(score[row])
    return int(
        np.count_nonzero(score > value)
        + np.count_nonzero((score == value) & (lex < lex[row]))
        + 1
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(TARGETS, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    positives = targets.groupby("reaction_id")["protein_id"].apply(lambda x: list(map(str, x))).to_dict()
    train = pd.read_csv(TRAIN, dtype=str).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    known = train.groupby("reaction_id")["protein_id"].apply(lambda x: set(map(str, x))).to_dict()
    seen_p = set(train.protein_id.astype(str))
    seen_r = set(train.reaction_id.astype(str))

    from projects.active.bridge.runtime.final_system import R2E_FUNCTIONAL_VALIDATION_SCALE
    assert R2E_FUNCTIONAL_VALIDATION_SCALE == FUNCTIONAL_TEMPERATURE
    rt = FinalBridgeRuntime(device=args.device)
    with GATE.open("rb") as handle:
        unified_gate = pickle.load(handle)
    assert unified_gate["direction"] == "r2e"
    rt.relation_gate["directions"]["r2e"] = unified_gate
    print("R2E_SHARED_5216_VALIDATION_RELATION_GATE_AND_FUNCTIONAL_CALIBRATION", FUNCTIONAL_TEMPERATURE, flush=True)
    lex = rt._protein_lex
    n = len(rt.index.protein_ids)
    protein_rows = torch.arange(n, device=rt.index.device)
    edge_rows = []
    query_rows = []

    def local_scores(
        q: str,
        broad: np.ndarray,
        bstd: float,
        ctx: np.ndarray,
        relation_weight: float,
        retrieval_eval: np.ndarray,
        use_relation: bool,
        use_functional: bool,
        use_structure: bool,
        use_domain: bool,
    ) -> tuple[dict[str, int], dict[str, float]]:
        finite = np.isfinite(retrieval_eval)
        k = min(TOPK_R2E, int(finite.sum()))
        score_t = torch.as_tensor(retrieval_eval, dtype=torch.float32, device=rt.index.device)
        _, inds = torch.topk(score_t, k=k, largest=True, sorted=True)
        top = inds.cpu().numpy().astype(np.int64, copy=False)
        candidates = [rt.index.protein_ids[int(row)] for row in top]

        core = broad[top]
        core_std = max(float(core.std()), 1e-6)
        core_z = (core - float(core.mean())) / core_std
        relation_local = (
            relation_weight * (bstd / core_std) * ctx[top]
            if use_relation
            else np.zeros(k, dtype=np.float64)
        )

        frows = rt._r2e_functional_p_row[top]
        fr = rt.functional_r2e.r_index.get(q, -1)
        fa = frows >= 0
        fraw = np.zeros(k, dtype=np.float64)
        if fr < 0:
            fa[:] = False
        elif fa.any():
            rows_t = torch.as_tensor(frows[fa], dtype=torch.long, device=rt.index.device)
            with torch.no_grad():
                fraw[fa] = (
                    rt.functional_r2e.p.index_select(0, rows_t)
                    @ rt.functional_r2e.r[fr]
                ).float().cpu().numpy()
        fz = _z(fraw, fa)

        crows = rt._clip_p_row[top]
        cr = rt.clip.r_index.get(q, -1)
        ca = crows >= 0
        if cr >= 0 and bool(rt.clip.r_supported[cr]):
            ca &= rt.clip.p_supported[np.maximum(crows, 0)]
        else:
            ca[:] = False
        craw = np.zeros(k, dtype=np.float64)
        if ca.any():
            rows_t = torch.as_tensor(crows[ca], dtype=torch.long, device=rt.index.device)
            with torch.no_grad():
                craw[ca] = (
                    rt.clip.p_device.index_select(0, rows_t) @ rt.clip.r_device[cr]
                ).float().cpu().numpy()
        cz = _z(craw, ca)

        mrows = rt._mechanism_p_row[top]
        mr = rt.mechanism.r_index.get(q, -1)
        ma = mrows >= 0
        mraw = np.zeros(k, dtype=np.float64)
        if mr < 0:
            ma[:] = False
        elif ma.any():
            rows_t = torch.as_tensor(mrows[ma], dtype=torch.long, device=rt.index.device)
            with torch.no_grad():
                mraw[ma] = (
                    rt.mechanism.p.index_select(0, rows_t) @ rt.mechanism.r[mr]
                ).float().cpu().numpy()
        mz = _z(mraw, ma)
        ga = ca | ma
        geometry = np.zeros(k, dtype=np.float64)
        count = np.zeros(k, dtype=np.int32)
        for part, available in ((cz, ca), (mz, ma)):
            geometry[available] += part[available]
            count[available] += 1
        geometry[ga] /= count[ga]

        fw, gw = rt._r2e_general_weights(q, core, fz, fa, geometry, ga)
        specialist, specialist_audit = rt._r2e_specialists(q, candidates)
        score = core_z + relation_local
        if use_functional:
            score = score + fw * fz
        if use_structure:
            score = score + gw * geometry
        if use_domain:
            score = score + specialist

        order = (
            rt._pocket_reorder(q, candidates, score)
            if use_structure
            else np.argsort(-score, kind="stable")
        )
        local_rank = {candidates[int(local)]: rank for rank, local in enumerate(order, 1)}
        audit = {
            "functional_weight": float(fw),
            "geometry_weight": float(gw),
            "family_weight": float(specialist_audit.get("family_weight", 0.0)),
            "tps_weight": float(specialist_audit.get("tps_weight", 0.0)),
        }
        return local_rank, audit

    variants = {
        "full": dict(use_relation=True, use_functional=True, use_structure=True, use_domain=True),
        "minus_functional": dict(use_relation=True, use_functional=False, use_structure=True, use_domain=True),
        "minus_structure_mechanism": dict(use_relation=True, use_functional=True, use_structure=False, use_domain=True),
        "minus_relational_memory": dict(use_relation=False, use_functional=True, use_structure=True, use_domain=True),
        "minus_family_domain": dict(use_relation=True, use_functional=True, use_structure=True, use_domain=False),
    }

    for qi, q in enumerate(sorted(positives), 1):
        if q not in rt.index.reaction_index:
            continue
        target_ids = [p for p in positives[q] if p in rt.index.protein_index]
        if not target_ids:
            continue

        qrow = rt.index.reaction_index[q]
        with torch.no_grad():
            broad_t = (rt.index.reaction_embeddings[qrow] @ rt.index.protein_embeddings.T).float()
        broad = broad_t.cpu().numpy().astype(np.float64, copy=False)
        bstd = max(float(broad.std()), 1e-8)
        bz = (broad - float(broad.mean())) / bstd
        ctx, ca0, cb0, available, both = rt._relation_components("r2e", q)
        rw, _, _ = rt._relation_authority("r2e", q, bz, ctx, ca0, cb0, available, both)

        broad_eval = broad.copy()
        relation_eval = broad + bstd * rw * ctx
        masked = [rt.index.protein_index[p] for p in known.get(q, set()) if p in rt.index.protein_index]
        if masked:
            m = np.asarray(masked, dtype=np.int64)
            broad_eval[m] = -np.inf
            relation_eval[m] = -np.inf

        variant_maps: dict[str, dict[str, int]] = {}
        audit = {}
        for name, flags in variants.items():
            retrieval = relation_eval if flags["use_relation"] else broad_eval
            rank_map, vaudit = local_scores(
                q=q,
                broad=broad,
                bstd=bstd,
                ctx=ctx,
                relation_weight=rw,
                retrieval_eval=retrieval,
                **flags,
            )
            variant_maps[name] = rank_map
            if name == "full":
                audit = vaudit

        raw: dict[str, list[int]] = {name: [] for name in variants}
        broad_raw = []
        for pid in target_ids:
            row = rt.index.protein_index[pid]
            broad_raw.append(rank_full(broad_eval, row, lex))
            for name, flags in variants.items():
                retrieval = relation_eval if flags["use_relation"] else broad_eval
                raw[name].append(
                    variant_maps[name].get(pid, rank_full(retrieval, row, lex))
                )

        broad_filtered = filtered_ranks(broad_raw)
        filtered = {name: filtered_ranks(values) for name, values in raw.items()}
        for i, pid in enumerate(target_ids):
            rec = {
                "protein_id": pid,
                "reaction_id": q,
                "broad_rank": broad_filtered[i],
                "protein_seen": pid in seen_p,
                "reaction_seen": q in seen_r,
            }
            for name in variants:
                rec[name + "_rank"] = filtered[name][i]
            edge_rows.append(rec)

        query_rows.append({
            "query_id": q,
            "target_relations": len(target_ids),
            "relational_memory_weight": float(rw),
            **audit,
        })
        if qi % 50 == 0 or qi == len(positives):
            print("r2e-ablation", qi, "/", len(positives), flush=True)

    ef = pd.DataFrame(edge_rows)
    qf = pd.DataFrame(query_rows)
    ef.to_csv(OUT / "edge_metrics.csv.gz", index=False)
    qf.to_csv(OUT / "query_audit.csv.gz", index=False)

    def metric(col: str) -> dict:
        r = ef[col].to_numpy(np.int64)
        return {
            "mrr": float((1.0 / r).mean()),
            "hit10": float((r <= 10).mean()),
            "hit100": float((r <= 100).mean()),
            "hit1000": float((r <= 1000).mean()),
        }

    summary = {
        "schema": "bridge-r2e-four-group-ablation-v1",
        "status": "completed",
        "edges": int(len(ef)),
        "queries": int(ef.reaction_id.nunique()),
        "groups": {
            "functional": "EnzGFM functional compatibility",
            "structure_mechanism": "CLIPZyme + reaction-center/mechanism + pocket reorder",
            "relational_memory": "query-conditioned clean2023 long-term relation memory; episodic memory absent by protocol",
            "family_domain": "family CAGE + TPS specialist authority",
        },
        "metrics": {
            "broad": metric("broad_rank"),
            **{name: metric(name + "_rank") for name in variants},
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    summary["shared_validation_positive_edges"] = 5216
    summary["graph_relation_gate_source"] = str(GATE.relative_to(ROOT))
    summary["functional_temperature_5216_selected"] = FUNCTIONAL_TEMPERATURE
    summary["test_labels_used_in_calibration"] = False
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
