from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RELEASE_ROOT = Path("/tmp/bridge_rhea_hist")
DEFAULT_OUT = ROOT / "results/bridge_rhea_longitudinal_growth_v4"
RELEASES = tuple(range(116, 143))
EVAL_BASELINE = 128
FINAL_RELEASE = 142

KNOWN_EXPECTED = {
    128: "06a88c1fb29a3170bc533e9557e8da7b278c6c0e2583159492c08aaea958abe0",
    134: "b5e3927284ae0f6fed10a68c7709d75f8445043bf12d0dfa11cc2b71739c78ef",
    138: "25a3ed70b0c555835f10680b47a6538a6a4fa5123eb7a320b4f9107794ea7693",
    142: "ff2e1038df2352d4ce4f3e9ae840c350211d8193e79b6a0a2d29fc1c2f392def",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_alias_map(meta: pd.DataFrame) -> tuple[dict[str, str], int]:
    alias_to_ids: dict[str, set[str]] = {}
    for row in meta[["protein_id", "canonical_accession", "aliases"]].fillna("").itertuples(index=False):
        values = {str(row.protein_id), str(row.canonical_accession)}
        values.update(x for x in re.split(r";+", str(row.aliases)) if x)
        for alias in values:
            if alias:
                alias_to_ids.setdefault(alias, set()).add(str(row.protein_id))
    ambiguous = sum(len(ids) > 1 for ids in alias_to_ids.values())
    return {a: next(iter(ids)) for a, ids in alias_to_ids.items() if len(ids) == 1}, ambiguous


def load_release(n: int, root: Path, alias: dict[str, str], reaction_ids: set[str]):
    path = root / str(n) / "tsv/rhea2uniprot_sprot.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256(path)
    expected = KNOWN_EXPECTED.get(n)
    if expected and digest != expected:
        raise ValueError(f"Rhea release {n} mapping hash drift: {digest}")
    source = pd.read_csv(path, sep="\t", dtype=str, usecols=["RHEA_ID", "ID"]).fillna("")
    frame = source.copy()
    frame["protein_id"] = frame.ID.map(alias).fillna("")
    frame["reaction_id"] = "RHEA:" + frame.RHEA_ID.astype(str)
    frame = frame[
        frame.protein_id.ne("") & frame.reaction_id.isin(reaction_ids)
    ][["protein_id", "reaction_id"]].drop_duplicates()
    return frame.reset_index(drop=True), int(len(source)), digest


def classify(protein_id: str, reaction_id: str, proteins: set[str], reactions: set[str]) -> tuple[str, str]:
    p_seen = protein_id in proteins
    r_seen = reaction_id in reactions
    n_old = int(p_seen) + int(r_seen)
    growth = {
        2: "edge_completion",
        1: "single_endpoint_expansion",
        0: "double_endpoint_expansion",
    }[n_old]
    orientation = (
        "both_old"
        if n_old == 2
        else (
            "new_protein"
            if (not p_seen and r_seen)
            else ("new_reaction" if (p_seen and not r_seen) else "both_new")
        )
    )
    return growth, orientation


def dist(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"events": 0, "mean_edges": 0.0, "median_edges": 0.0, "p90_edges": 0.0, "max_edges": 0}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "events": int(len(arr)),
        "mean_edges": float(arr.mean()),
        "median_edges": float(np.median(arr)),
        "p90_edges": float(np.quantile(arr, 0.9, method="nearest")),
        "max_edges": int(arr.max()),
    }


def aggregate_windows(frame: pd.DataFrame, start_previous_release: int) -> dict[str, object]:
    z = frame[frame.previous_release >= start_previous_release].copy()
    orient = {
        k: int(v)
        for k, v in z[["both_old", "new_protein", "new_reaction", "both_new"]].sum().items()
    }
    added = int(z.added_edges.sum())
    expansion = orient["new_protein"] + orient["new_reaction"] + orient["both_new"]
    return {
        "windows": int(len(z)),
        "added_edges": added,
        "removed_edges": int(z.removed_edges.sum()),
        "orientation_counts": orient,
        "edge_completion_fraction": float(orient["both_old"] / added) if added else 0.0,
        "graph_expansion_fraction": float(expansion / added) if added else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    release_root = args.release_root.resolve()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/protein_metadata.csv",
        dtype=str,
    ).fillna("")
    alias, ambiguous = build_alias_map(meta)
    reactions = pd.read_csv(
        ROOT / "data/catalyst_candidate_universes/general_merged/reactions.csv",
        dtype=str,
    ).fillna("")
    reaction_ids = set(reactions.reaction_id.astype(str))
    clean = pd.read_csv(
        ROOT / "data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv",
        dtype=str,
    ).fillna("").drop_duplicates(["protein_id", "reaction_id"])
    clean_set = set(map(tuple, clean[["protein_id", "reaction_id"]].to_numpy()))

    frames: dict[int, pd.DataFrame] = {}
    graphs: dict[int, set[tuple[str, str]]] = {}
    proteins: dict[int, set[str]] = {}
    reactions_by_release: dict[int, set[str]] = {}
    audits: dict[int, dict[str, object]] = {}
    for n in RELEASES:
        frame, source_rows, digest = load_release(n, release_root, alias, reaction_ids)
        frames[n] = frame
        graphs[n] = set(map(tuple, frame[["protein_id", "reaction_id"]].to_numpy()))
        proteins[n] = set(frame.protein_id.astype(str))
        reactions_by_release[n] = set(frame.reaction_id.astype(str))
        audits[n] = {
            "source_rows": source_rows,
            "mapped_pairs": int(len(frame)),
            "proteins": int(frame.protein_id.nunique()),
            "reactions": int(frame.reaction_id.nunique()),
            "sha256": digest,
        }

    window_rows = []
    all_protein_bundles: list[int] = []
    all_reaction_bundles: list[int] = []
    post_protein_bundles: list[int] = []
    post_reaction_bundles: list[int] = []
    reaction_attachment_ratios: list[float] = []
    protein_attachment_ratios: list[float] = []
    post_reaction_attachment_ratios: list[float] = []
    post_protein_attachment_ratios: list[float] = []
    reaction_attachment_higher = 0
    protein_attachment_higher = 0

    for n in RELEASES[1:]:
        prev = n - 1
        added = graphs[n] - graphs[prev]
        removed = graphs[prev] - graphs[n]
        counts = Counter()
        protein_bundles = Counter()
        reaction_bundles = Counter()
        p_degree = Counter(p for p, _ in graphs[prev])
        r_degree = Counter(r for _, r in graphs[prev])
        selected_old_reactions: set[str] = set()
        selected_old_proteins: set[str] = set()

        for protein_id, reaction_id in added:
            _, orientation = classify(
                protein_id,
                reaction_id,
                proteins[prev],
                reactions_by_release[prev],
            )
            counts[orientation] += 1
            if protein_id not in proteins[prev]:
                protein_bundles[protein_id] += 1
            if reaction_id not in reactions_by_release[prev]:
                reaction_bundles[reaction_id] += 1
            if protein_id not in proteins[prev] and reaction_id in reactions_by_release[prev]:
                selected_old_reactions.add(reaction_id)
            if protein_id in proteins[prev] and reaction_id not in reactions_by_release[prev]:
                selected_old_proteins.add(protein_id)

        old_reaction_ratio = np.nan
        if selected_old_reactions:
            selected_median = float(np.median([r_degree[x] for x in selected_old_reactions]))
            all_median = float(np.median(list(r_degree.values())))
            old_reaction_ratio = selected_median / all_median if all_median else np.nan
            reaction_attachment_ratios.append(float(old_reaction_ratio))
            reaction_attachment_higher += int(selected_median > all_median)
            if prev >= EVAL_BASELINE:
                post_reaction_attachment_ratios.append(float(old_reaction_ratio))

        old_protein_ratio = np.nan
        if selected_old_proteins:
            selected_median = float(np.median([p_degree[x] for x in selected_old_proteins]))
            all_median = float(np.median(list(p_degree.values())))
            old_protein_ratio = selected_median / all_median if all_median else np.nan
            protein_attachment_ratios.append(float(old_protein_ratio))
            protein_attachment_higher += int(selected_median > all_median)
            if prev >= EVAL_BASELINE:
                post_protein_attachment_ratios.append(float(old_protein_ratio))

        p_bundle_values = list(protein_bundles.values())
        r_bundle_values = list(reaction_bundles.values())
        all_protein_bundles.extend(p_bundle_values)
        all_reaction_bundles.extend(r_bundle_values)
        if prev >= EVAL_BASELINE:
            post_protein_bundles.extend(p_bundle_values)
            post_reaction_bundles.extend(r_bundle_values)

        window_rows.append(
            {
                "previous_release": prev,
                "release": n,
                "added_edges": len(added),
                "removed_edges": len(removed),
                "both_old": counts["both_old"],
                "new_protein": counts["new_protein"],
                "new_reaction": counts["new_reaction"],
                "both_new": counts["both_new"],
                "new_protein_events": len(protein_bundles),
                "new_reaction_events": len(reaction_bundles),
                "new_protein_bundle_mean": float(np.mean(p_bundle_values)) if p_bundle_values else 0.0,
                "new_reaction_bundle_mean": float(np.mean(r_bundle_values)) if r_bundle_values else 0.0,
                "new_protein_to_old_reaction_degree_median_ratio": old_reaction_ratio,
                "new_reaction_to_old_protein_degree_median_ratio": old_protein_ratio,
            }
        )

    window_frame = pd.DataFrame(window_rows)
    window_frame.to_csv(out / "window_stats.csv", index=False)

    baseline_graph = graphs[EVAL_BASELINE]
    final_graph = graphs[FINAL_RELEASE]
    target_pairs = sorted((final_graph - baseline_graph) - clean_set)
    target_rows = []
    baseline_proteins = proteins[EVAL_BASELINE]
    baseline_reactions = reactions_by_release[EVAL_BASELINE]

    for protein_id, reaction_id in target_pairs:
        first = next(
            n for n in range(EVAL_BASELINE + 1, FINAL_RELEASE + 1)
            if (protein_id, reaction_id) in graphs[n]
        )
        prev = first - 1
        insertion_growth, insertion_orientation = classify(
            protein_id,
            reaction_id,
            proteins[prev],
            reactions_by_release[prev],
        )
        baseline_growth, baseline_orientation = classify(
            protein_id,
            reaction_id,
            baseline_proteins,
            baseline_reactions,
        )
        target_rows.append(
            {
                "protein_id": protein_id,
                "reaction_id": reaction_id,
                "first_observed_release": first,
                "previous_release": prev,
                "insertion_growth_class": insertion_growth,
                "insertion_orientation": insertion_orientation,
                "baseline_growth_class": baseline_growth,
                "baseline_orientation": baseline_orientation,
                "protein_present_previous": protein_id in proteins[prev],
                "reaction_present_previous": reaction_id in reactions_by_release[prev],
                "protein_present_r128": protein_id in baseline_proteins,
                "reaction_present_r128": reaction_id in baseline_reactions,
            }
        )

    target = pd.DataFrame(target_rows)
    target.to_csv(out / "targets.csv", index=False)

    classification_changed = int(
        (target.insertion_orientation != target.baseline_orientation).sum()
    )
    insertion_counts = {
        k: int(v) for k, v in target.insertion_orientation.value_counts().items()
    }
    baseline_counts = {
        k: int(v) for k, v in target.baseline_orientation.value_counts().items()
    }

    summary = {
        "schema": "bridge-rhea-longitudinal-growth-v4",
        "source": "official Rhea historical rhea2uniprot_sprot.tsv mappings",
        "release_range": [RELEASES[0], RELEASES[-1]],
        "evaluation_cutoff": EVAL_BASELINE,
        "final_release": FINAL_RELEASE,
        "mapping": {
            "unambiguous_protein_alias_keys": int(len(alias)),
            "ambiguous_alias_keys": int(ambiguous),
            "reaction_id_semantics": "direction-specific RHEA_ID",
            "clean2023_pair_overlap_excluded": True,
        },
        "releases": {str(n): audits[n] for n in RELEASES},
        "longitudinal_graph_growth": {
            "all_windows": aggregate_windows(window_frame, RELEASES[0]),
            "post_cutoff_windows": aggregate_windows(window_frame, EVAL_BASELINE),
            "new_protein_arrival_bundles_all": dist(all_protein_bundles),
            "new_reaction_arrival_bundles_all": dist(all_reaction_bundles),
            "new_protein_arrival_bundles_post_cutoff": dist(post_protein_bundles),
            "new_reaction_arrival_bundles_post_cutoff": dist(post_reaction_bundles),
            "preferential_attachment": {
                "new_protein_to_old_reaction": {
                    "windows_with_events": int(len(reaction_attachment_ratios)),
                    "windows_above_background_median_degree": int(reaction_attachment_higher),
                    "median_selected_to_background_degree_ratio": float(np.median(reaction_attachment_ratios)),
                    "post_cutoff_median_ratio": float(np.median(post_reaction_attachment_ratios)),
                },
                "new_reaction_to_old_protein": {
                    "windows_with_events": int(len(protein_attachment_ratios)),
                    "windows_above_background_median_degree": int(protein_attachment_higher),
                    "median_selected_to_background_degree_ratio": float(np.median(protein_attachment_ratios)),
                    "post_cutoff_median_ratio": float(np.median(post_protein_attachment_ratios)),
                },
            },
        },
        "persistent_post_cutoff_targets": {
            "edges": int(len(target)),
            "r2e_queries": int(target.reaction_id.nunique()),
            "e2r_queries": int(target.protein_id.nunique()),
            "by_insertion_orientation": insertion_counts,
            "by_baseline_orientation": baseline_counts,
            "classification_changed_edges": classification_changed,
            "classification_changed_fraction": float(classification_changed / len(target)),
            "by_first_release": {
                str(k): int(v)
                for k, v in target.first_observed_release.value_counts().sort_index().items()
            },
        },
        "protocol_note": (
            "Long-run releases 116-142 are used to characterize the database growth process. "
            "Ranking claims use only relations added after release128 because BRIDGE expert assets "
            "are frozen to clean2023; pre-cutoff releases are never used as future-label ranking tests."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
