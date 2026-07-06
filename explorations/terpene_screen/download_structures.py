from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.error
import urllib.request
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from explorations.terpene_screen.common import (
    SOURCE_FILES,
    TERPENE_DATA_DIR,
    TERPENE_RESULTS_DIR,
    coerce_text,
    dedupe_preserve_order,
    identify_terpene_columns,
    parse_uniprot_id,
    read_table,
    safe_json_dump,
    write_table,
)


ALPHAFOLD_URL_TEMPLATE = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.cif"
MIN_VALID_CIF_BYTES = 1000
DOWNLOAD_TIMEOUT_SECONDS = 60


def _download_url(url: str, output_path: Path) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
        if len(payload) < MIN_VALID_CIF_BYTES:
            return False, "downloaded file too small"
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_bytes(payload)
        tmp_path.replace(output_path)
        return True, ""
    except urllib.error.HTTPError as exc:
        return False, f"http_error={exc.code}"
    except Exception as exc:  # noqa: BLE001 - we want a robust best-effort downloader
        return False, str(exc)


def _resolve_api_url(uniprot_id: str) -> str | None:
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        with urllib.request.urlopen(api_url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, list) and payload:
            cif_url = payload[0].get("cifUrl")
            if isinstance(cif_url, str) and cif_url:
                return cif_url
    except Exception:
        return None
    return None


def _normalize_candidate_pool(candidate_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_table(candidate_csv)
    cols = identify_terpene_columns(df)
    id_col = cols["uniprot_id"]["column"] or cols["enzyme_id"]["column"]
    seq_col = cols["sequence"]["column"]

    records: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    seen: set[str] = set()

    for idx, row in df.iterrows():
        raw_id = coerce_text(row.get(id_col)) if id_col else ""
        uniprot_id = parse_uniprot_id(raw_id)
        sequence = coerce_text(row.get(seq_col)) if seq_col else ""
        if not uniprot_id:
            failed.append(
                {
                    "source_file": str(candidate_csv),
                    "row_index": int(idx),
                    "raw_id": raw_id,
                    "resolved_uniprot_id": "",
                    "reason": "unresolved_uniprot_id",
                    "sequence_present": bool(sequence),
                }
            )
            continue
        if not sequence:
            failed.append(
                {
                    "source_file": str(candidate_csv),
                    "row_index": int(idx),
                    "raw_id": raw_id,
                    "resolved_uniprot_id": uniprot_id,
                    "reason": "missing_sequence",
                    "sequence_present": False,
                }
            )
            continue
        if uniprot_id in seen:
            continue
        seen.add(uniprot_id)
        records.append(
            {
                "enzyme_id": raw_id or uniprot_id,
                "uniprot_id": uniprot_id,
                "sequence": sequence,
            }
        )

    pool_df = pd.DataFrame(records)
    if pool_df.empty:
        pool_df = pd.DataFrame(columns=["enzyme_id", "uniprot_id", "sequence"])

    failed_df = pd.DataFrame(failed)
    if failed_df.empty:
        failed_df = pd.DataFrame(
            columns=["source_file", "row_index", "raw_id", "resolved_uniprot_id", "reason", "sequence_present"]
        )
    return pool_df, failed_df


def _download_single(record: dict[str, Any], structure_dir: Path) -> dict[str, Any]:
    uniprot_id = record["uniprot_id"]
    enzyme_id = record["enzyme_id"]
    output_path = structure_dir / f"{uniprot_id}.cif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size >= MIN_VALID_CIF_BYTES:
        return {
            "enzyme_id": enzyme_id,
            "uniprot_id": uniprot_id,
            "status": "existing",
            "attempted_versions": "",
            "downloaded_version": "",
            "download_url": "",
            "local_path": str(output_path),
            "error": "",
        }

    if output_path.exists():
        output_path.unlink()

    attempts: list[str] = []
    for version in (4, 3, 2):
        url = ALPHAFOLD_URL_TEMPLATE.format(uniprot=uniprot_id, version=version)
        ok, error = _download_url(url, output_path)
        attempts.append(f"v{version}:{'ok' if ok else error}")
        if ok:
            return {
                "enzyme_id": enzyme_id,
                "uniprot_id": uniprot_id,
                "status": "downloaded",
                "attempted_versions": "|".join(attempts),
                "downloaded_version": str(version),
                "download_url": url,
                "local_path": str(output_path),
                "error": "",
            }

    api_url = _resolve_api_url(uniprot_id)
    if api_url:
        ok, error = _download_url(api_url, output_path)
        attempts.append(f"api:{'ok' if ok else error}")
        if ok:
            return {
                "enzyme_id": enzyme_id,
                "uniprot_id": uniprot_id,
                "status": "downloaded",
                "attempted_versions": "|".join(attempts),
                "downloaded_version": "api",
                "download_url": api_url,
                "local_path": str(output_path),
                "error": "",
            }

    if output_path.exists():
        output_path.unlink()
    return {
        "enzyme_id": enzyme_id,
        "uniprot_id": uniprot_id,
        "status": "failed",
        "attempted_versions": "|".join(attempts),
        "downloaded_version": "",
        "download_url": "",
        "local_path": "",
        "error": "; ".join(attempts),
    }


def download_structures(candidate_csv: Path, structure_dir: Path, report_csv: Path, failed_mapping_tsv: Path, workers: int) -> dict[str, Any]:
    pool_df, failed_df = _normalize_candidate_pool(candidate_csv)
    if failed_df.empty:
        failed_df = pd.DataFrame(
            columns=["source_file", "row_index", "raw_id", "resolved_uniprot_id", "reason", "sequence_present"]
        )
    write_table(failed_df, failed_mapping_tsv, sep="\t")

    structure_dir.mkdir(parents=True, exist_ok=True)
    pool_records = pool_df.to_dict("records")

    report_rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(_download_single, record, structure_dir): record for record in pool_records}
        for future in cf.as_completed(future_map):
            report_rows.append(future.result())

    report_rows = sorted(report_rows, key=lambda row: row["uniprot_id"])
    report_df = pd.DataFrame(report_rows)
    write_table(report_df, report_csv, sep=",")

    summary = {
        "candidate_csv": str(candidate_csv),
        "structure_dir": str(structure_dir),
        "report_csv": str(report_csv),
        "failed_mapping_tsv": str(failed_mapping_tsv),
        "n_input_rows": int(len(read_table(candidate_csv))),
        "n_unique_candidates": int(len(pool_df)),
        "n_failed_id_mapping_rows": int(len(failed_df)),
        "n_existing_structures": int((report_df["status"] == "existing").sum()) if not report_df.empty else 0,
        "n_downloaded_structures": int((report_df["status"] == "downloaded").sum()) if not report_df.empty else 0,
        "n_failed_structures": int((report_df["status"] == "failed").sum()) if not report_df.empty else 0,
        "n_total_report_rows": int(len(report_df)),
    }
    safe_json_dump(summary, report_csv.with_suffix(".json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AlphaFold CIF structures for terpene synthase candidates.")
    parser.add_argument("--candidate_pairs", default=str(TERPENE_DATA_DIR / "terpene_candidate_pairs.csv"))
    parser.add_argument("--structure_dir", default=str(TERPENE_DATA_DIR / "structures"))
    parser.add_argument("--report_csv", default=str(TERPENE_RESULTS_DIR / "structure_download_report.csv"))
    parser.add_argument("--failed_id_mapping", default=str(TERPENE_RESULTS_DIR / "failed_id_mapping.tsv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    summary = download_structures(
        candidate_csv=Path(args.candidate_pairs),
        structure_dir=Path(args.structure_dir),
        report_csv=Path(args.report_csv),
        failed_mapping_tsv=Path(args.failed_id_mapping),
        workers=args.workers,
    )
    print(summary)


if __name__ == "__main__":
    main()
