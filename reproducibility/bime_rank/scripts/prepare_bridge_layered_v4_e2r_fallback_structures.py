from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from projects.active.bridge.model.assets import ROOT

SUPPORT = ROOT / "results/bridge_layered_v4_cage_support"
REGISTRY = SUPPORT / "protein_support_registry.csv.gz"
RESOLUTION = SUPPORT / "e2r_fallback_afdb_alias_resolution.csv"
LOCAL_AF = ROOT / "results/clipzyme_native_extension_v1/structures/af_v6"
OUT = ROOT / "results/bridge_layered_v4_cage_features/e2r_fallback"
STRUCTURES = OUT / "structures"


def fetch(uid: str, destination: Path) -> tuple[str, str]:
    local = LOCAL_AF / f"AF-{uid}-F1-model_v6.cif"
    if local.exists():
        if not destination.exists():
            try:
                destination.symlink_to(local)
            except OSError:
                shutil.copy2(local, destination)
        return uid, "local_reuse"
    if destination.exists() and destination.stat().st_size > 0:
        return uid, "download_reuse"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.cif"
    tmp = destination.with_suffix(".cif.tmp")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        tmp.replace(destination)
        return uid, "downloaded"
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return uid, f"failed:{type(exc).__name__}:{exc}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STRUCTURES.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(REGISTRY, dtype=str).fillna("")
    fallback = registry[
        registry.entity_kind.eq("general_merged")
        & registry.scopes.str.contains("e2r_query", regex=False)
        & registry.support_source.eq("fallback_structure")
    ][["entity_id", "chosen_uid", "sequence"]].drop_duplicates("entity_id")
    resolution = pd.read_csv(RESOLUTION, dtype=str).fillna("")
    merged = fallback.merge(
        resolution[["protein_id", "afdb_uid"]],
        left_on="entity_id",
        right_on="protein_id",
        how="left",
        validate="one_to_one",
    ).fillna("")
    available = merged[merged.afdb_uid.ne("")].copy()
    available["UniprotID"] = available.afdb_uid.astype(str)
    available[["UniprotID", "sequence", "entity_id", "chosen_uid"]].to_csv(
        OUT / "input.csv", index=False
    )

    uids = sorted(set(available.UniprotID.astype(str)))
    records: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(fetch, uid, STRUCTURES / f"{uid}.cif"): uid
            for uid in uids
        }
        for future in as_completed(futures):
            records.append(future.result())

    audit = pd.DataFrame(records, columns=["UniprotID", "structure_source"]).sort_values(
        "UniprotID", kind="stable"
    )
    audit.to_csv(OUT / "structure_audit.csv", index=False)
    failed = audit.structure_source.str.startswith("failed:")
    summary = {
        "schema": "bridge-layered-v4-e2r-fallback-structures",
        "status": "completed",
        "fallback_queries": int(len(fallback)),
        "afdb_resolved_queries": int(len(available)),
        "unresolved_queries": int(len(fallback) - len(available)),
        "structure_files": int(sum((STRUCTURES / f"{uid}.cif").exists() for uid in uids)),
        "download_failures": int(failed.sum()),
        "structure_sources": {
            str(k): int(v)
            for k, v in audit.structure_source.value_counts().sort_index().items()
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
