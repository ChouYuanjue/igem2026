from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CELLS = [
    "broad_reaction_hash_cold_protein_seen",
    "reactzyme_reaction_projected_double_cold",
    "temporal_post2020_double_cold",
]
RUNS = {
    "nested": ROOT / "results/broad_rhea_full_candidate_nested_selected_v1/{cell}/{cell}/summary.json",
    "frozen_novelty": ROOT / "results/broad_rhea_full_candidate_novelty_expert_frozen_v1/{cell}/{cell}/summary.json",
    "rdkitplus_isolate": ROOT / "results/broad_rhea_rdkitplus_outer_eval_v1/isolate/{cell}/{cell}/summary.json",
    "rdkitplus_frozen_novelty": ROOT / "results/broad_rhea_rdkitplus_outer_eval_v1/novelty/{cell}/{cell}/summary.json",
}
METRICS = [
    "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "hit_at_20", "hit_at_50",
    "mrr", "map", "macro_roc_auc", "ndcg_at_10", "top1_percent_ef",
    "median_best_positive_rank", "success_at_0.01_fraction",
]


def path_for(template: Path, cell: str) -> Path:
    return Path(str(template).format(cell=cell))


def markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        vals=[]
        for v in row:
            if isinstance(v, float): vals.append(f"{v:.6g}")
            else: vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    out = ROOT / "results/broad_rhea_rdkitplus_outer_matrix_v1"
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for cell in CELLS:
        manifest = json.loads((ROOT / f"results/broad_rhea_fair_benchmarks_v1/{cell}/manifest.json").read_text())
        for run, template in RUNS.items():
            p=path_for(template,cell)
            payload=json.loads(p.read_text())
            m=payload["metrics"]["reaction_to_enzyme"]
            rows.append({
                "cell":cell,
                "claim_tier":manifest.get("claim_tier",""),
                "run":run,
                "queries":m["query_count"],
                "candidate_proteins":payload["candidate_proteins"],
                **{k:m[k] for k in METRICS},
            })
    frame=pd.DataFrame(rows)
    frame.to_csv(out/"matrix.csv",index=False)
    (out/"matrix.md").write_text(markdown(frame),encoding="utf-8")
    # frozen-combination delta against each prior clean run
    deltas=[]
    for cell in CELLS:
        sub=frame[frame.cell.eq(cell)].set_index("run")
        cand=sub.loc["rdkitplus_frozen_novelty"]
        for baseline in ["nested","frozen_novelty"]:
            base=sub.loc[baseline]
            deltas.append({
                "cell":cell,"baseline":baseline,
                **{f"delta_{k}":float(cand[k]-base[k]) for k in METRICS if k != "median_best_positive_rank"},
                "median_rank_improvement":float(base["median_best_positive_rank"]-cand["median_best_positive_rank"]),
            })
    delta=pd.DataFrame(deltas); delta.to_csv(out/"deltas.csv",index=False)
    (out/"deltas.md").write_text(markdown(delta),encoding="utf-8")
    # collect paired bootstrap vs frozen novelty where available
    boots=[]
    for cell in CELLS:
        p=ROOT/f"results/broad_rhea_rdkitplus_outer_bootstrap_v1/{cell}/vs_novelty/paired_bootstrap.csv"
        if p.exists():
            b=pd.read_csv(p); b.insert(0,"cell",cell); boots.append(b)
    if boots:
        boot=pd.concat(boots,ignore_index=True); boot.to_csv(out/"bootstrap_vs_frozen_novelty.csv",index=False)
        (out/"bootstrap_vs_frozen_novelty.md").write_text(markdown(boot),encoding="utf-8")
    print(frame.to_string(index=False))
    print("\nDELTA VS FROZEN NOVELTY\n",delta[delta.baseline.eq('frozen_novelty')].to_string(index=False))


if __name__ == "__main__":
    main()
