from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projects.active.bridge.model.assets import ROOT
from projects.active.bridge.model.index import DEFAULT_INDEX, FibreCandidateIndex


DEFAULT_PREPARED = ROOT / "results/unified_safe_system_v1/e2r_clipzyme_anchored_lambdamart_v4_dev/prepared"
DEFAULT_OUTPUT = ROOT / "results/fibre_score_evidence_main_v1/e2r"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebind established E2R OOF evidence to the frozen BRIDGE broad core.")
    ap.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    broad = FibreCandidateIndex(device=args.device)
    parts_core: list[pd.DataFrame] = []
    parts_clip: list[pd.DataFrame] = []
    parts_enz: list[pd.DataFrame] = []

    for fold in (0, 1, 2):
        root = args.prepared / f"fold{fold}"
        queries = pd.read_csv(root / "queries.csv", dtype=str).query_id.astype(str).tolist()
        candidates = [x.strip() for x in (root / "candidate_reactions.txt").read_text().splitlines() if x.strip()]
        feature_names = json.loads((root / "feature_names.json").read_text())
        raw_enzgfm_i = feature_names.index("raw_enzgfm")
        clip_i = feature_names.index("clip_raw")
        clip_c_i = feature_names.index("clip_candidate_supported")
        clip_q_i = feature_names.index("clip_query_supported")
        x = np.load(root / "X.npy", mmap_mode="r")
        rows = np.load(root / "rows.npy", mmap_mode="r")
        labels = np.load(root / "labels.npy", mmap_mode="r")
        offsets = np.load(root / "offsets.npy")
        if len(offsets) != len(queries) + 1:
            raise ValueError(f"fold {fold}: query offsets do not align")

        core_rows: list[pd.DataFrame] = []
        clip_rows: list[pd.DataFrame] = []
        enz_rows: list[pd.DataFrame] = []
        for qi, query_id in enumerate(queries):
            a, b = map(int, offsets[qi : qi + 2])
            local_rows = np.asarray(rows[a:b], dtype=np.int64)
            local_ids = [candidates[int(i)] for i in local_rows]
            local_labels = np.asarray(labels[a:b], dtype=np.int8)

            prow = broad.protein_index.get(str(query_id))
            if prow is None:
                raise KeyError(f"broad core missing protein {query_id}")
            rrows = torch.as_tensor(
                [broad.reaction_index[x] for x in local_ids],
                dtype=torch.long,
                device=broad.device,
            )
            with torch.no_grad():
                core = broad.reaction_embeddings.index_select(0, rrows) @ broad.protein_embeddings[prow]
            core_np = core.float().cpu().numpy()

            enzgfm_score = np.asarray(x[a:b, raw_enzgfm_i], dtype=np.float32)
            enzgfm_available = np.isfinite(enzgfm_score)
            clip_available = (
                (np.asarray(x[a:b, clip_c_i]) > 0.5)
                & (np.asarray(x[a:b, clip_q_i]) > 0.5)
            )

            core_rows.append(pd.DataFrame({
                "query_id": str(query_id),
                "candidate_id": local_ids,
                "core_score": core_np,
                "label": local_labels,
                "fold": fold,
            }))
            clip_rows.append(pd.DataFrame({
                "direction": "e2r",
                "query_id": str(query_id),
                "candidate_id": local_ids,
                "score": np.asarray(x[a:b, clip_i], dtype=np.float32),
                "available": clip_available,
            }))
            enz_rows.append(pd.DataFrame({
                "direction": "e2r",
                "query_id": str(query_id),
                "candidate_id": local_ids,
                "score": enzgfm_score,
                "available": enzgfm_available,
            }))
            if (qi + 1) % 1000 == 0:
                print(f"fold={fold} queries={qi + 1}/{len(queries)}", flush=True)

        parts_core.append(pd.concat(core_rows, ignore_index=True))
        parts_clip.append(pd.concat(clip_rows, ignore_index=True))
        parts_enz.append(pd.concat(enz_rows, ignore_index=True))

    core = pd.concat(parts_core, ignore_index=True)
    clip = pd.concat(parts_clip, ignore_index=True)
    enz = pd.concat(parts_enz, ignore_index=True)
    if core.duplicated(["query_id", "candidate_id"]).any():
        raise ValueError("E2R OOF table contains duplicate query/candidate rows")

    args.output.mkdir(parents=True, exist_ok=True)
    core.to_csv(args.output / "core.csv", index=False)
    center = float(core.core_score.mean())
    scale = float(core.core_score.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("broad E2R core scores have no usable fixed scale")
    calibrated = core.copy()
    calibrated["core_score"] = (calibrated.core_score - center) / scale
    calibrated.to_csv(args.output / "core_calibrated.csv", index=False)
    clip.to_csv(args.output / "clipzyme.csv", index=False)
    enz.to_csv(args.output / "enzgfm.csv", index=False)

    checkpoint_hash = sha256(DEFAULT_INDEX)
    baseline_id = f"fibre-broad-rankstrong-r2e98-e2r-{checkpoint_hash[:8]}"
    (args.output / "core_calibration.json").write_text(json.dumps({
        "method": "fixed_global_affine_v1",
        "baseline_id": baseline_id,
        "center": center,
        "scale": scale,
        "ranking_invariant": True,
        "checkpoint_sha256": checkpoint_hash,
    }, indent=2) + "\n")

    descriptors = {
        "clipzyme_descriptor.json": {
            "name": "clipzyme_structure", "kind": "structural", "role": "rerank",
            "directions": ["e2r"],
            "score_semantics": "higher cosine means stronger CLIPZyme structural compatibility",
            "availability_semantics": "both enzyme query and reaction candidate have valid CLIPZyme embeddings",
            "quality_semantics": None,
            "provenance": "frozen CLIPZyme query/reaction assets already present in igem2026",
            "score_direction": "higher_is_better",
        },
        "enzgfm_descriptor.json": {
            "name": "enzgfm_e2r", "kind": "molecular_view", "role": "rerank",
            "directions": ["e2r"],
            "score_semantics": "higher frozen EnzGFM E2R dual-tower cosine means stronger enzyme-reaction compatibility",
            "availability_semantics": "query protein and candidate reaction exist in the frozen EnzGFM E2R universe",
            "quality_semantics": None,
            "provenance": "fold-specific OOF raw_enzgfm from the established E2R prepared cache; clean2023 full production checkpoint at runtime",
            "score_direction": "higher_is_better",
        },
    }
    for name, payload in descriptors.items():
        (args.output / name).write_text(json.dumps(payload, indent=2) + "\n")

    manifest = {
        "schema": "fibre-score-evidence-e2r-v1",
        "rows": int(len(core)),
        "queries": int(core.query_id.nunique()),
        "positive_rows": int(core.label.sum()),
        "baseline_id": baseline_id,
        "baseline_checkpoint": str(DEFAULT_INDEX.relative_to(ROOT)),
        "baseline_checkpoint_sha256": checkpoint_hash,
        "core_calibration": {"center": center, "scale": scale},
        "experts": ["clipzyme_structure", "enzgfm_e2r"],
        "tps_policy": "application-only; excluded from frozen broad admission",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
