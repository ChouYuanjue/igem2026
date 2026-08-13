from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .classifier import SummaryEvidence, classify
from .db import connect, init_evidence_db, require_role, set_meta
from .pipeline import initialize_run, run_db_path, utcnow
from .planner import Stage1Signals, stage1_decision, stage2_decision
from .safety import ensure_runtime_dirs


def _bool(v) -> bool:
    if isinstance(v, bool): return v
    if pd.isna(v): return False
    return str(v).strip().lower() in {"1","true","yes","y"}


def _int_or_none(v):
    if v is None or pd.isna(v) or str(v).strip()=="": return None
    return int(float(v))


def _text(v) -> str:
    if v is None or pd.isna(v):
        return ""
    return str(v).strip()


def _split(s: str) -> list[str]:
    text = _text(s)
    if not text: return []
    return sorted(set(x.strip() for x in text.split(";") if x.strip()))


def bootstrap_from_v2(profile, run_id: str, v2_tags: Path, overwrite: bool = False) -> dict:
    """Migrate the existing full v2 snapshot into isolated run/evidence stores.

    Only exact sequence->UniParc/accession mappings are copied into the reusable evidence DB.
    Candidate-level aggregated UniProt annotations are NOT fanned out to individual accessions,
    avoiding false attribution. Stage-2 can later refresh only the promoted shortlist.
    """
    ensure_runtime_dirs(profile)
    init_evidence_db(profile.runtime.evidence_db)
    path=initialize_run(profile,run_id,overwrite=overwrite)
    d=pd.read_csv(v2_tags,sep="\t",low_memory=False)
    required={"enzyme_id","sequence_md5","uniparc_exact_found","experimental_touch_level"}
    if not required.issubset(d.columns): raise ValueError(f"v2 snapshot missing {sorted(required-set(d.columns))}")
    econ=connect(profile.runtime.evidence_db); rcon=connect(path)
    require_role(econ, "public_evidence"); require_role(rcon, "candidate_run_state")
    ids={x[0] for x in rcon.execute("SELECT candidate_id FROM candidates")}
    if set(d.enzyme_id.astype(str))!=ids:
        raise ValueError("v2 snapshot candidate IDs do not exactly match profile candidate source")
    # Safe reusable exact-sequence mapping only.
    for r in d.to_dict("records"):
        found=_bool(r.get("uniparc_exact_found"))
        accs=_split(r.get("uniprot_accessions",""))
        econ.execute(
            """INSERT OR REPLACE INTO uniparc_exact
               (sequence_md5,found,uniparc_id,accessions_json,first_seen,last_seen,release,release_date,fetched_utc)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (str(r["sequence_md5"]),1 if found else 0,str(r.get("uniparc_id","") or ""),json.dumps(accs),
             "","","2026_02","10-June-2026",utcnow()),
        )
        for a in accs:
            rcon.execute("INSERT OR IGNORE INTO candidate_accession(candidate_id,accession) VALUES(?,?)",(str(r["enzyme_id"]),a))
    econ.commit()
    force2={x[0] for x in rcon.execute("SELECT DISTINCT candidate_id FROM focus WHERE force_stage=2")}
    force3={x[0] for x in rcon.execute("SELECT DISTINCT candidate_id FROM focus WHERE force_stage=3")}
    stage1={"FINALIZE":0,"PROMOTE":0}; stage2={"FINALIZE":0,"PROMOTE":0}
    rcon.execute("DELETE FROM stage_decision"); rcon.execute("DELETE FROM tag_state")
    for r in d.to_dict("records"):
        cid=str(r["enzyme_id"]); public=_bool(r.get("uniparc_exact_found")); pe=_int_or_none(r.get("best_pe_level"))
        reviewed=_bool(r.get("reviewed_exact")); has_pdb=bool(_text(r.get("pdb_exact_ids","")))
        structured=_bool(r.get("experimental_eco")) or _bool(r.get("functional_experimental_eco")) or _bool(r.get("experimental_functional_eco"))
        if not profile.policy.promote_reviewed: reviewed=False
        if not profile.policy.promote_pdb: has_pdb=False
        if not profile.policy.promote_structured_experiment: structured=False
        dec1,why1=stage1_decision(Stage1Signals(
            public_exact=public,best_pe_level=pe,reviewed=reviewed,has_pdb=has_pdb,
            structured_experiment=structured,force_stage2=cid in force2 or cid in force3,force_stage3=cid in force3,
        ),profile.policy.promote_pe_at_most)
        stage1[dec1]+=1
        touch=str(r["experimental_touch_level"])
        if touch not in {f"T{i}" for i in range(6)}: raise ValueError(f"bad v2 tag {touch}")
        # v2 itself is already a structured evidence snapshot; preserve its tag for bootstrap.
        if dec1=="PROMOTE":
            dec2,why2=stage2_decision(touch,False,cid in force3)
            stage2[dec2]+=1
            evstage=2; final=1 if dec2=="FINALIZE" else 0
            rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,2,dec2,"bootstrap-v2: "+why2,utcnow()))
        else:
            evstage=1; final=1
        rcon.execute("INSERT OR REPLACE INTO tag_state VALUES(?,?,?,?,?,?)",
                     (cid,touch,evstage,final,"bootstrap from validated ExperimentalTouch v2 snapshot",utcnow()))
        rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,1,dec1,"bootstrap-v2: "+why1,utcnow()))
    if stage1["PROMOTE"]>profile.policy.stage2_max_candidates:
        raise RuntimeError(f"stage2 queue {stage1['PROMOTE']} exceeds configured cap")
    if stage2["PROMOTE"]>profile.policy.stage3_max_candidates:
        raise RuntimeError(f"stage3 queue {stage2['PROMOTE']} exceeds configured cap")
    set_meta(rcon,"bootstrap_source",str(v2_tags.resolve())); set_meta(rcon,"bootstrap_mode","validated-v2-snapshot")
    set_meta(rcon,"stage1_counts",stage1); set_meta(rcon,"stage2_counts",stage2); set_meta(rcon,"bootstrapped_utc",utcnow())
    rcon.commit(); econ.close(); rcon.close()
    return {"run_db":str(path),"stage1":stage1,"stage2":stage2}
