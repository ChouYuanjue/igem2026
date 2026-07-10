from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


MANIFEST_FIELDS = [
    "run_id",
    "enzyme_id",
    "structure_path",
    "pocket_method",
    "pocket_source",
    "pocket_rank",
    "pocket_global_id",
    "pocket_score",
    "pocket_center_x",
    "pocket_center_y",
    "pocket_center_z",
    "pocket_residues",
    "pocket_pdb_path",
    "source_raw_dir",
    "pocket_pdb_mode",
]

POCKET_PDB_MODES = {
    "cropped_pocket",
    "full_structure_placeholder",
    "external_original",
}


@dataclass
class PocketRecord:
    run_id: str
    enzyme_id: str
    structure_path: str
    pocket_method: str
    pocket_source: str
    pocket_rank: int
    pocket_global_id: str
    pocket_score: float | None
    pocket_center_x: float | None
    pocket_center_y: float | None
    pocket_center_z: float | None
    pocket_residues: str
    pocket_pdb_path: str
    source_raw_dir: str
    pocket_pdb_mode: str


def make_pocket_global_id(enzyme_id: str, pocket_source: str, pocket_rank: int) -> str:
    return f"{enzyme_id}__{pocket_source}__rank{int(pocket_rank)}"


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _to_optional_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    return float(value)


def _to_int(value: object) -> int:
    if _is_missing(value):
        raise ValueError("pocket_rank is required and cannot be empty")
    return int(float(value))


def _to_str(value: object) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _infer_source(row: dict[str, object]) -> str:
    source = _to_str(row.get("pocket_source"))
    if source:
        return source
    method = _to_str(row.get("pocket_method"))
    if method.startswith("p2rank"):
        return "p2rank"
    if method.startswith("fpocket"):
        return "fpocket"
    return method or "unknown"


def _validate_mode(mode: str) -> str:
    if mode in POCKET_PDB_MODES:
        return mode
    if mode:
        return mode
    return "external_original"


def _record_to_csv_row(record: PocketRecord) -> dict[str, object]:
    row = asdict(record)
    if not row.get("pocket_global_id"):
        row["pocket_global_id"] = make_pocket_global_id(
            row["enzyme_id"],
            row["pocket_source"],
            row["pocket_rank"],
        )
    row["pocket_pdb_mode"] = _validate_mode(str(row.get("pocket_pdb_mode", "")))
    for key in [
        "pocket_score",
        "pocket_center_x",
        "pocket_center_y",
        "pocket_center_z",
    ]:
        if row[key] is None:
            row[key] = ""
    return {field: row.get(field, "") for field in MANIFEST_FIELDS}


def _row_to_record(row: dict[str, object]) -> PocketRecord:
    enzyme_id = _to_str(row.get("enzyme_id"))
    pocket_source = _infer_source(row)
    pocket_rank = _to_int(row.get("pocket_rank"))
    pocket_global_id = _to_str(row.get("pocket_global_id")) or make_pocket_global_id(
        enzyme_id,
        pocket_source,
        pocket_rank,
    )
    return PocketRecord(
        run_id=_to_str(row.get("run_id")),
        enzyme_id=enzyme_id,
        structure_path=_to_str(row.get("structure_path")),
        pocket_method=_to_str(row.get("pocket_method")),
        pocket_source=pocket_source,
        pocket_rank=pocket_rank,
        pocket_global_id=pocket_global_id,
        pocket_score=_to_optional_float(row.get("pocket_score")),
        pocket_center_x=_to_optional_float(row.get("pocket_center_x")),
        pocket_center_y=_to_optional_float(row.get("pocket_center_y")),
        pocket_center_z=_to_optional_float(row.get("pocket_center_z")),
        pocket_residues=_to_str(row.get("pocket_residues")),
        pocket_pdb_path=_to_str(row.get("pocket_pdb_path")),
        source_raw_dir=_to_str(row.get("source_raw_dir")),
        pocket_pdb_mode=_validate_mode(_to_str(row.get("pocket_pdb_mode"))),
    )


def read_manifest(path: str | Path) -> list[PocketRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        return [_row_to_record(row) for row in reader]


def write_manifest(records: list[PocketRecord], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_csv_row(record))


def records_to_dataframe(records: Iterable[PocketRecord]) -> pd.DataFrame:
    rows = [_record_to_csv_row(record) for record in records]
    return pd.DataFrame(rows, columns=MANIFEST_FIELDS)


def dataframe_to_records(df: pd.DataFrame) -> list[PocketRecord]:
    records: list[PocketRecord] = []
    for _, row in df.iterrows():
        values = {field: row.get(field, "") for field in MANIFEST_FIELDS}
        for legacy_field in ["run_id", "enzyme_id", "structure_path", "pocket_method", "pocket_rank"]:
            if legacy_field in df.columns and legacy_field not in values:
                values[legacy_field] = row.get(legacy_field, "")
        records.append(_row_to_record(values))
    return records


if __name__ == "__main__":
    raise SystemExit("pocket_manifest.py is a library module, not a CLI.")
