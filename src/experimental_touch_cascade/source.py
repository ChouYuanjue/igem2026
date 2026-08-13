from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).split()).upper().rstrip("*")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    # Fingerprint exact source bytes, including gzip container bytes when compressed.
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_fingerprint(profile) -> str:
    h = hashlib.sha256()
    for p in [profile.source.candidates_path, profile.source.metadata_path]:
        if p:
            h.update(str(p.resolve()).encode())
            h.update(file_sha256(p).encode())
    return h.hexdigest()


def read_candidates(profile) -> pd.DataFrame:
    s = profile.source
    if s.kind != "csv":
        raise NotImplementedError("Current isolated adapter supports CSV/TSV candidate sources; SQLite RO adapter is intentionally separate and opt-in.")
    sep = "\t" if ".tsv" in s.candidates_path.name else ","
    d = pd.read_csv(s.candidates_path, sep=sep, dtype=str).fillna("")
    required = {s.id_column, s.sequence_column}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"candidate source missing columns: {sorted(missing)}")
    out = pd.DataFrame({
        "candidate_id": d[s.id_column].astype(str),
        "sequence": d[s.sequence_column].map(clean_sequence),
    })
    if out.candidate_id.duplicated().any():
        raise ValueError("candidate IDs must be unique")
    if (out.sequence == "").any():
        raise ValueError("empty candidate sequence")
    out["sequence_md5"] = out.sequence.map(lambda x: hashlib.md5(x.encode()).hexdigest())
    out["sequence_sha256"] = out.sequence.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return out


def read_metadata(profile, columns: list[str] | None = None) -> pd.DataFrame | None:
    p = profile.source.metadata_path
    if not p:
        return None
    sep = "\t" if ".tsv" in p.name else ","
    if columns:
        header = pd.read_csv(p, sep=sep, nrows=0).columns
        use = [c for c in columns if c in header]
    else:
        use = None
    return pd.read_csv(p, sep=sep, usecols=use, dtype=str).fillna("")
