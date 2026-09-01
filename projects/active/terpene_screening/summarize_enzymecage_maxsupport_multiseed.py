from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from projects.active.terpene_screening.evaluate_enzymecage_official_aligned import evaluate_scores

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / "results/enzymecage_local_maxsupport_v1"
SEEDS = (40, 41, 42, 43, 44)
SPECS = {
    "complete226": "complete_candidate_226_epoch_19.csv",
    "max295": "max_pair_intersection_295_epoch_19.csv",
}


def _metric_frame(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    raw = pd.read_csv(path, dtype=str).fillna("")
    frame = pd.DataFrame(
        {
            "reaction_id": raw["CANO_RXN_SMILES"].astype(str),
            "protein_id": raw["UniprotID"].astype(str),
            "label": pd.to_numeric(raw["Label"], errors="raise").astype(int),
            "score": pd.to_numeric(raw["pred"], errors="raise"),
        }
    )
    metrics, _ = evaluate_scores(frame, "score")
    counts = {
        "raw_rows": int(len(raw)),
        "query_count": int(frame["reaction_id"].nunique()),
        "candidate_uid_count": int(frame["protein_id"].nunique()),
        "positive_rows": int(frame["label"].sum()),
    }
    return metrics, counts


def _flatten(prefix: str, value: object, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if np.isfinite(float(value)):
            out[prefix] = float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate EnzymeCAGE official seeds 40-44 on frozen Enzyme-405 supports."
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    output = (args.output or (results_dir / "multiseed_summary.json")).resolve()

    summary: dict[str, object] = {
        "protocol": "EnzymeCAGE official pretrained seeds 40-44 on frozen maximum-support Enzyme-405 reservoirs",
        "seed_selection_allowed": False,
        "seeds": list(SEEDS),
        "query_grouping": "CANO_RXN_SMILES",
        "score_column": "pred",
        "std_definition": "population standard deviation across all five fixed official seeds (ddof=0)",
        "supports": {},
    }

    for spec, filename in SPECS.items():
        per_seed: dict[str, object] = {}
        flat_by_seed: dict[int, dict[str, float]] = {}
        reference_counts: dict[str, int] | None = None
        for seed in SEEDS:
            path = results_dir / f"{spec}_seed{seed}" / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            metrics, counts = _metric_frame(path)
            if reference_counts is None:
                reference_counts = counts
            elif counts != reference_counts:
                raise ValueError(f"support drift for {spec} seed {seed}: {counts} != {reference_counts}")
            flat: dict[str, float] = {}
            _flatten("", metrics, flat)
            flat_by_seed[seed] = flat
            per_seed[str(seed)] = {"path": str(path), "counts": counts, "metrics": metrics}

        metric_keys = sorted(set.intersection(*(set(v) for v in flat_by_seed.values())))
        aggregate = {}
        for key in metric_keys:
            values = np.asarray([flat_by_seed[s][key] for s in SEEDS], dtype=np.float64)
            aggregate[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "max": float(values.max()),
                "values_by_seed": {str(seed): float(flat_by_seed[seed][key]) for seed in SEEDS},
            }
        summary["supports"][spec] = {
            "counts": reference_counts,
            "per_seed": per_seed,
            "aggregate": aggregate,
        }

    # Fail closed if the newly reconstructed seed42 no longer equals the historical recorded summary.
    checks: dict[str, object] = {}
    for spec in SPECS:
        old_path = results_dir / f"{spec}_seed42" / "summary.json"
        if not old_path.is_file():
            raise FileNotFoundError(old_path)
        old = json.loads(old_path.read_text(encoding="utf-8"))["metrics"]
        current = summary["supports"][spec]["per_seed"]["42"]["metrics"]
        check_fields = [
            ("enzymecage_native_r2e", "top10_sr"),
            ("enzymecage_native_r2e", "top10_dcg"),
            ("reaction_to_enzyme", "mrr"),
            ("reaction_to_enzyme", "map"),
            ("reaction_to_enzyme", "macro_roc_auc"),
            ("reaction_to_enzyme", "ndcg_at_10"),
            ("reaction_to_enzyme", "hit_at_10"),
        ]
        deltas = {
            f"{section}.{field}": float(current[section][field]) - float(old[section][field])
            for section, field in check_fields
        }
        max_abs = max(abs(value) for value in deltas.values())
        if max_abs > 1e-12:
            raise ValueError(f"seed42 reproduction drift for {spec}: max_abs_delta={max_abs}")
        checks[spec] = {"max_abs_delta": max_abs, "fields": deltas}
    summary["seed42_reproduction_checks"] = checks

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
