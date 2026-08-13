from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import PROTOCOL_ID

EVIDENCE_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uniparc_exact(
  sequence_md5 TEXT PRIMARY KEY,
  found INTEGER NOT NULL CHECK(found IN (0,1)),
  uniparc_id TEXT,
  accessions_json TEXT NOT NULL DEFAULT '[]',
  first_seen TEXT,
  last_seen TEXT,
  release TEXT,
  release_date TEXT,
  fetched_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uniprot_summary(
  accession TEXT PRIMARY KEY,
  reviewed INTEGER NOT NULL DEFAULT 0 CHECK(reviewed IN (0,1)),
  pe_level INTEGER,
  pe_text TEXT,
  pdb_ids_json TEXT NOT NULL DEFAULT '[]',
  functional_experimental INTEGER NOT NULL DEFAULT 0 CHECK(functional_experimental IN (0,1)),
  catalytic_experimental INTEGER NOT NULL DEFAULT 0 CHECK(catalytic_experimental IN (0,1)),
  kinetics_present INTEGER NOT NULL DEFAULT 0 CHECK(kinetics_present IN (0,1)),
  mass_spec_present INTEGER NOT NULL DEFAULT 0 CHECK(mass_spec_present IN (0,1)),
  pubmed_ids_json TEXT NOT NULL DEFAULT '[]',
  evidence_depth INTEGER NOT NULL DEFAULT 1 CHECK(evidence_depth BETWEEN 1 AND 3),
  release TEXT,
  release_date TEXT,
  fetched_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS structure_evidence(
  accession TEXT NOT NULL,
  pdb_id TEXT NOT NULL,
  experimental INTEGER NOT NULL CHECK(experimental IN (0,1)),
  method TEXT,
  resolution REAL,
  fetched_utc TEXT NOT NULL,
  PRIMARY KEY(accession,pdb_id)
);
CREATE TABLE IF NOT EXISTS pubmed_article(
  pmid TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  abstract TEXT NOT NULL DEFAULT '',
  evidence_terms_json TEXT NOT NULL DEFAULT '[]',
  suggested_touch TEXT,
  fetched_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS literature_search(
  search_key TEXT PRIMARY KEY,
  aliases_json TEXT NOT NULL,
  pmids_json TEXT NOT NULL DEFAULT '[]',
  fetched_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS literature_evidence(
  evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
  public_key TEXT NOT NULL,
  pmid TEXT,
  title TEXT,
  abstract TEXT,
  evidence_terms_json TEXT NOT NULL DEFAULT '[]',
  exact_identity_confirmed INTEGER NOT NULL DEFAULT 0 CHECK(exact_identity_confirmed IN (0,1)),
  suggested_touch TEXT,
  source TEXT NOT NULL,
  fetched_utc TEXT NOT NULL,
  UNIQUE(public_key,pmid,source)
);
"""

RUN_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates(
  candidate_id TEXT PRIMARY KEY,
  sequence_sha256 TEXT NOT NULL,
  sequence_md5 TEXT NOT NULL,
  source_namespace TEXT NOT NULL,
  source_locator TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS focus(
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  focus_group TEXT NOT NULL,
  model_rank INTEGER NOT NULL,
  force_stage INTEGER NOT NULL CHECK(force_stage IN (2,3)),
  PRIMARY KEY(candidate_id,focus_group,force_stage)
);
CREATE TABLE IF NOT EXISTS candidate_accession(
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  accession TEXT NOT NULL,
  PRIMARY KEY(candidate_id,accession)
);
CREATE TABLE IF NOT EXISTS tag_state(
  candidate_id TEXT PRIMARY KEY REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  touch_level TEXT NOT NULL CHECK(touch_level IN ('T0','T1','T2','T3','T4','T5')),
  evidence_stage INTEGER NOT NULL CHECK(evidence_stage BETWEEN 1 AND 3),
  finalized INTEGER NOT NULL DEFAULT 0 CHECK(finalized IN (0,1)),
  reason TEXT NOT NULL,
  updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_decision(
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  stage INTEGER NOT NULL CHECK(stage BETWEEN 1 AND 3),
  decision TEXT NOT NULL CHECK(decision IN ('FINALIZE','PROMOTE','FORCE','REVIEW')),
  reason TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  PRIMARY KEY(candidate_id,stage)
);
CREATE TABLE IF NOT EXISTS manual_override(
  candidate_id TEXT PRIMARY KEY REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  touch_level TEXT NOT NULL CHECK(touch_level IN ('T0','T1','T2','T3','T4','T5')),
  evidence_ref TEXT NOT NULL,
  note TEXT,
  reviewer TEXT,
  created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidate_evidence_ref(
  candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  stage INTEGER NOT NULL CHECK(stage BETWEEN 1 AND 3),
  evidence_kind TEXT NOT NULL,
  public_key TEXT NOT NULL,
  external_ref TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(candidate_id,stage,evidence_kind,public_key,external_ref)
);
CREATE INDEX IF NOT EXISTS idx_tag_touch ON tag_state(touch_level);
CREATE INDEX IF NOT EXISTS idx_stage_decision ON stage_decision(stage,decision);
CREATE INDEX IF NOT EXISTS idx_focus_rank ON focus(force_stage,model_rank);
"""


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    path = path.resolve()
    if readonly:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_evidence_db(path: Path) -> None:
    con = connect(path)
    con.executescript(EVIDENCE_SCHEMA)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('role','public_evidence')")
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('protocol_id',?)", (PROTOCOL_ID,))
    con.commit(); con.close()


def init_run_db(path: Path, profile_id: str, source_fingerprint: str) -> None:
    con = connect(path)
    con.executescript(RUN_SCHEMA)
    for k, v in {
        "role": "candidate_run_state",
        "protocol_id": PROTOCOL_ID,
        "profile_id": profile_id,
        "source_fingerprint": source_fingerprint,
    }.items():
        con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (k, str(v)))
    con.commit(); con.close()


def set_meta(con: sqlite3.Connection, key: str, value) -> None:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, value))


def require_role(con: sqlite3.Connection, expected: str) -> None:
    row = con.execute("SELECT value FROM meta WHERE key='role'").fetchone()
    actual = None if row is None else row[0]
    if actual != expected:
        raise RuntimeError(f"database role mismatch: expected={expected!r}, actual={actual!r}")
