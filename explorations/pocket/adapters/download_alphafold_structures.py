from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ENZYME_COLUMNS = ["UniprotID", "uniprot_id", "uid", "protein_id", "enzyme_id"]
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v{version}.cif"


def _normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize(column): column for column in columns}
    for candidate in candidates:
        key = _normalize(candidate)
        if key in normalized:
            return normalized[key]
    return None


def _download(url: str, output_path: Path, timeout: int = 60) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if getattr(response, "status", 200) != 200:
                return False, f"http_status={getattr(response, 'status', 'unknown')}"
            data = response.read()
        if len(data) < 1000:
            return False, "downloaded file is too small to be a valid AlphaFold CIF"
        output_path.write_bytes(data)
        return True, ""
    except urllib.error.HTTPError as exc:
        return False, f"http_error={exc.code}"
    except Exception as exc:  # noqa: BLE001 - downloader should keep going
        return False, str(exc)


def download_structures(input_csv: Path, output_root: Path, output_pairs_csv: Path | None = None) -> dict[str, Any]:
    cif_dir = output_root / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "alphafold_download_report.csv"
    summary_path = output_root / "alphafold_download_summary.json"

    df = pd.read_csv(input_csv)
    enzyme_col = _find_column(list(df.columns), ENZYME_COLUMNS)
    if enzyme_col is None:
        raise ValueError(f"Could not find enzyme id column in {input_csv}")

    enzyme_ids = list(dict.fromkeys(df[enzyme_col].dropna().astype(str).str.strip()))
    report_rows: list[dict[str, Any]] = []
    downloaded_paths: dict[str, Path] = {}

    for enzyme_id in enzyme_ids:
        final_path = cif_dir / f"{enzyme_id}.cif"
        if final_path.exists() and final_path.stat().st_size > 1000:
            downloaded_paths[enzyme_id] = final_path
            report_rows.append(
                {
                    "enzyme_id": enzyme_id,
                    "status": "already_exists",
                    "version": "",
                    "url": "",
                    "local_path": str(final_path),
                    "error": "",
                }
            )
            continue

        success = False
        errors = []
        for version in [4, 3, 2, 6, 5]:
            url = ALPHAFOLD_URL.format(uniprot=enzyme_id, version=version)
            ok, error = _download(url, final_path)
            report_rows.append(
                {
                    "enzyme_id": enzyme_id,
                    "status": "downloaded" if ok else "failed_attempt",
                    "version": version,
                    "url": url,
                    "local_path": str(final_path) if ok else "",
                    "error": error,
                }
            )
            if ok:
                downloaded_paths[enzyme_id] = final_path
                success = True
                break
            errors.append(f"v{version}:{error}")
        if not success and final_path.exists():
            final_path.unlink()

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)

    filtered = df[df[enzyme_col].astype(str).isin(downloaded_paths)].copy()
    if enzyme_col != "UniprotID":
        filtered["UniprotID"] = filtered[enzyme_col].astype(str)
    if "enzyme_id" not in filtered.columns:
        filtered["enzyme_id"] = filtered["UniprotID"].astype(str)
    filtered["structure_path"] = filtered["UniprotID"].astype(str).map(lambda uid: str(downloaded_paths[uid]))

    if output_pairs_csv is None:
        output_pairs_csv = input_csv.parent / "smallset_pairs_with_structures.csv"
    output_pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_pairs_csv, index=False)

    summary = {
        "status": "completed" if downloaded_paths else "blocked_no_alphafold_structures_downloaded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "output_root": str(output_root),
        "cif_dir": str(cif_dir),
        "output_pairs_csv": str(output_pairs_csv),
        "n_requested_enzymes": len(enzyme_ids),
        "n_downloaded_enzymes": len(downloaded_paths),
        "n_input_pairs": int(len(df)),
        "n_pairs_with_structures": int(len(filtered)),
        "report_csv": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download full AlphaFold CIF structures for a smallset.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_root", default="data/assets/alphafold_structures")
    parser.add_argument("--output_pairs_csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = download_structures(
        input_csv=Path(args.input_csv),
        output_root=Path(args.output_root),
        output_pairs_csv=Path(args.output_pairs_csv) if args.output_pairs_csv else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
