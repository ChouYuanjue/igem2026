from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from projects.active.bridge.model.index import DEFAULT_INDEX
from projects.active.bridge.runtime.base_model import ModelConfig, TerpeneDualTower

BASE = ROOT / "results/fibre_vs_enzymecage_external_families_v1"
PREP = BASE / "prep"
CAGE_ROOT = ROOT / "external_repos/EnzymeCAGE/dataset/external-test-set"
FAMILIES = {
    "p450": CAGE_ROOT / "p450/test_P450.csv",
    "phosphatase": CAGE_ROOT / "phosphatase/test_Phosphatase.csv",
    "terpene": CAGE_ROOT / "terpene/test_Terpene.csv",
}
KS = (1, 3, 5, 10, 20, 50)


def rank_one(group: pd.DataFrame, score_col: str) -> dict[str, float]:
    ordered = group.sort_values([score_col, "UniprotID"], ascending=[False, True], kind="stable")
    y = pd.to_numeric(ordered["Label"], errors="raise").astype(int).to_numpy()
    ranks = np.flatnonzero(y == 1) + 1
    n = len(y)
    p = int(y.sum())
    best = int(ranks[0]) if len(ranks) else 0
    ap = float(sum(y[:r].sum() / r for r in ranks) / p) if p else 0.0
    ideal10 = sum(1.0 / math.log2(i + 2) for i in range(min(p, 10)))
    dcg10 = sum(1.0 / math.log2(int(r) + 1) for r in ranks if r <= 10)
    out: dict[str, float] = {
        "candidate_count": float(n),
        "positive_count": float(p),
        "best_positive_rank": float(best),
        "reciprocal_rank": 0.0 if best == 0 else 1.0 / best,
        "average_precision": ap,
        "ndcg_at_10": 0.0 if ideal10 == 0 else float(dcg10 / ideal10),
    }
    for k in KS:
        hits = int((ranks <= k).sum())
        out[f"hit_at_{k}"] = float(hits > 0)
        out[f"positive_hits_at_{k}"] = float(hits)
        out[f"positive_recall_at_{k}"] = float(hits / p) if p else 0.0
    for frac in (0.01, 0.03, 0.05):
        k = max(1, int(math.ceil(n * frac)))
        out[f"success_at_{int(frac*100)}pct"] = float(best > 0 and best <= k)
    return out


def evaluate(frame: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for reaction, group in frame.groupby("CANO_RXN_SMILES", sort=True):
        rows.append({"CANO_RXN_SMILES": reaction, **rank_one(group, score_col)})
    q = pd.DataFrame(rows)
    positives = float(q["positive_count"].sum())
    summary: dict[str, float] = {
        "queries": int(len(q)),
        "candidate_rows": int(len(frame)),
        "positive_rows": int(pd.to_numeric(frame["Label"]).sum()),
        "mrr": float(q["reciprocal_rank"].mean()),
        "map": float(q["average_precision"].mean()),
        "ndcg_at_10": float(q["ndcg_at_10"].mean()),
        "median_best_positive_rank": float(q["best_positive_rank"].median()),
    }
    for k in KS:
        summary[f"hit_at_{k}"] = float(q[f"hit_at_{k}"].mean())
        summary[f"macro_positive_recall_at_{k}"] = float(q[f"positive_recall_at_{k}"].mean())
        summary[f"micro_positive_recall_at_{k}"] = float(q[f"positive_hits_at_{k}"].sum() / positives)
    for pct in (1, 3, 5):
        summary[f"success_at_{pct}pct"] = float(q[f"success_at_{pct}pct"].mean())
    return q, summary


def load_current_model(device: torch.device) -> TerpeneDualTower:
    payload = torch.load(DEFAULT_INDEX, map_location="cpu", weights_only=False)
    model = TerpeneDualTower(ModelConfig(**dict(payload["model_config"])))
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval()


def score_broad(output: Path, device: torch.device) -> dict[str, object]:
    model = load_current_model(device)
    pentries = pd.read_csv(PREP / "external_protein_esmc/entries.csv", dtype=str).fillna("")
    pentries["row"] = pd.to_numeric(pentries["row"]).astype(int)
    pentries = pentries.sort_values("row")
    pmat = np.load(PREP / "external_protein_esmc/embeddings.npy").astype(np.float32)
    if len(pentries) != len(pmat):
        raise ValueError("external protein entries/features differ")
    with torch.no_grad():
        pz = model.encode_proteins(torch.as_tensor(pmat, device=device))
    pindex = {uid: i for i, uid in enumerate(pentries["Entry"].astype(str))}
    result: dict[str, object] = {}
    for family, source in FAMILIES.items():
        test = pd.read_csv(source, dtype=str).fillna("")
        registry = pd.read_csv(PREP / f"{family}_reaction_registry.csv", dtype=str).fillna("")
        rentries = pd.read_csv(PREP / f"{family}_reaction_features/entries.csv", dtype=str).fillna("")
        rentries["row"] = pd.to_numeric(rentries["row"]).astype(int)
        rentries = rentries.sort_values("row")
        rmat = np.load(PREP / f"{family}_reaction_features/reaction_feature_matrix.npy").astype(np.float32)
        with torch.no_grad():
            rz = model.encode_reactions(torch.as_tensor(rmat, device=device))
        rrow = dict(zip(rentries["reaction_id"].astype(str), range(len(rentries))))
        smi_to_rid = dict(zip(registry["reaction_smiles"].astype(str), registry["reaction_id"].astype(str)))
        missing_p = sorted(set(test["UniprotID"].astype(str)) - set(pindex))
        missing_r = sorted(set(test["CANO_RXN_SMILES"].astype(str)) - set(smi_to_rid))
        if missing_p or missing_r:
            raise ValueError(f"{family}: missing proteins={len(missing_p)}, reactions={len(missing_r)}")
        pi = torch.as_tensor([pindex[x] for x in test["UniprotID"].astype(str)], device=device)
        ri = torch.as_tensor([rrow[smi_to_rid[x]] for x in test["CANO_RXN_SMILES"].astype(str)], device=device)
        with torch.no_grad():
            score = (pz[pi] * rz[ri]).sum(dim=1).float().cpu().numpy()
        scored = test[["CANO_RXN_SMILES", "UniprotID", "Label"]].copy()
        scored["broad_score"] = score
        path = output / f"{family}_broad_pair_scores.csv.gz"
        scored.to_csv(path, index=False, compression="gzip")
        q, summary = evaluate(scored, "broad_score")
        q.to_csv(output / f"{family}_broad_query_metrics.csv", index=False)
        result[family] = summary
    return result


def attach_cage(base: pd.DataFrame, cage_path: Path, score_name: str) -> pd.DataFrame:
    cage = pd.read_csv(cage_path, dtype={"UniprotID": str}).fillna("")
    score_col = "pred_logit" if "pred_logit" in cage.columns else "pred"
    keys = ["CANO_RXN_SMILES", "UniprotID"]
    if cage.duplicated(keys).any():
        # Official pools should be unique by query/candidate; retain stable first only
        cage = cage.drop_duplicates(keys, keep="first")
    merged = base.merge(cage[keys + [score_col]], on=keys, how="left", validate="one_to_one")
    merged[score_name] = pd.to_numeric(merged[score_col], errors="coerce")
    merged = merged.drop(columns=[score_col])
    return merged


def summarize(output: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "schema": "fibre-vs-enzymecage-external-families-v1",
        "candidate_pool": "official EnzymeCAGE external-test pair tables, unchanged",
        "broad_model": str(DEFAULT_INDEX.relative_to(ROOT)),
        "families": {},
    }
    for family in FAMILIES:
        base = pd.read_csv(output / f"{family}_broad_pair_scores.csv.gz", dtype={"UniprotID": str}).fillna("")
        fam: dict[str, object] = {}
        _, fam["broad"] = evaluate(base, "broad_score")
        for mode, filename in [
            ("cage_pretrain", f"cage/{family}/pretrain/test_{'P450' if family=='p450' else 'Phosphatase' if family=='phosphatase' else 'Terpene'}_epoch_19.csv"),
            ("cage_finetune", f"cage/{family}/finetune/test_{'P450' if family=='p450' else 'Phosphatase' if family=='phosphatase' else 'Terpene'}_epoch_9.csv"),
        ]:
            path = output / filename
            if not path.exists():
                fam[mode] = {"status": "missing"}
                continue
            merged = attach_cage(base, path, mode)
            coverage = float(merged[mode].notna().mean())
            evaluable = merged[merged[mode].notna()].copy()
            _, cage_metrics = evaluate(evaluable, mode)
            _, broad_shared_metrics = evaluate(evaluable, "broad_score")
            fam[mode] = {
                "pair_coverage": coverage,
                "evaluable_queries": int(evaluable["CANO_RXN_SMILES"].nunique()),
                "evaluable_pairs": int(len(evaluable)),
                "broad_same_support": broad_shared_metrics,
                "enzymecage": cage_metrics,
            }
        summary["families"][family] = fam
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["score-broad", "summarize"])
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "score-broad":
        result = score_broad(output, torch.device(args.device))
        print(json.dumps(result, indent=2))
    else:
        summarize(output)


if __name__ == "__main__":
    main()
