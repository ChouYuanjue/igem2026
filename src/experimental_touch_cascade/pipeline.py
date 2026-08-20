from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .classifier import SummaryEvidence, classify
from .config import Profile
from .db import connect, init_evidence_db, init_run_db, require_role, set_meta
from .focus import load_focus
from .planner import Stage1Signals, stage1_decision, stage2_decision
from .providers import pubmed, rcsb, uniprot
from .safety import ensure_runtime_dirs
from .source import read_candidates, read_metadata, source_fingerprint


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def run_db_path(profile: Profile, run_id: str) -> Path:
    return profile.runtime.run_root / run_id / "run.sqlite"


def _upsert_uniparc(con, rows: list[dict], requested: list[str], release="", release_date=""):
    found = {r["sequence_md5"] for r in rows}
    now = utcnow()
    for r in rows:
        con.execute(
            """INSERT OR REPLACE INTO uniparc_exact
               (sequence_md5,found,uniparc_id,accessions_json,first_seen,last_seen,release,release_date,fetched_utc)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (r["sequence_md5"],1,r.get("uniparc_id",""),json.dumps(r.get("accessions",[])),
             r.get("first_seen",""),r.get("last_seen",""),r.get("release",release),
             r.get("release_date",release_date),r.get("fetched_utc",now)),
        )
    for md5 in requested:
        if md5 not in found:
            con.execute(
                """INSERT OR REPLACE INTO uniparc_exact
                   (sequence_md5,found,uniparc_id,accessions_json,release,release_date,fetched_utc)
                   VALUES(?,0,'','[]',?,?,?)""",
                (md5, release, release_date, now),
            )


def _upsert_uniprot(con, rows: list[dict]):
    for r in rows:
        con.execute(
            """INSERT OR REPLACE INTO uniprot_summary
               (accession,reviewed,pe_level,pe_text,pdb_ids_json,functional_experimental,
                catalytic_experimental,kinetics_present,mass_spec_present,pubmed_ids_json,
                evidence_depth,release,release_date,fetched_utc)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["accession"],int(r.get("reviewed",0)),r.get("pe_level"),r.get("pe_text",""),
             json.dumps(r.get("pdb_ids",[])),int(r.get("functional_experimental",0)),
             int(r.get("catalytic_experimental",0)),int(r.get("kinetics_present",0)),
             int(r.get("mass_spec_present",0)),json.dumps(r.get("pubmed_ids",[])),
             int(r.get("evidence_depth",1)),r.get("release",""),r.get("release_date",""),
             r.get("fetched_utc",utcnow())),
        )


def _candidate_aggregate(econ, rcon, candidate_id: str) -> dict:
    c = rcon.execute("SELECT sequence_md5 FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
    if c is None:
        raise KeyError(candidate_id)
    u = econ.execute("SELECT * FROM uniparc_exact WHERE sequence_md5=?", (c["sequence_md5"],)).fetchone()
    if not u or not int(u["found"]):
        return {
            "public_exact": False, "best_pe_level": None, "reviewed": False,
            "has_pdb": False, "functional_experimental": False, "catalytic_experimental": False,
            "kinetics_present": False, "mass_spec_present": False, "pubmed_ids": [], "accessions": [],
        }
    accs = json.loads(u["accessions_json"] or "[]")
    rows = []
    for a in accs:
        x = econ.execute("SELECT * FROM uniprot_summary WHERE accession=?", (a,)).fetchone()
        if x: rows.append(x)
    pe = [int(x["pe_level"]) for x in rows if x["pe_level"] is not None]
    pdbs = sorted({p for x in rows for p in json.loads(x["pdb_ids_json"] or "[]")})
    pmids = sorted({p for x in rows for p in json.loads(x["pubmed_ids_json"] or "[]")})
    return {
        "public_exact": True,
        "best_pe_level": min(pe) if pe else None,
        "reviewed": any(int(x["reviewed"]) for x in rows),
        "has_pdb": bool(pdbs),
        "pdb_ids": pdbs,
        "functional_experimental": any(int(x["functional_experimental"]) for x in rows),
        "catalytic_experimental": any(int(x["catalytic_experimental"]) for x in rows),
        "kinetics_present": any(int(x["kinetics_present"]) for x in rows),
        "mass_spec_present": any(int(x["mass_spec_present"]) for x in rows),
        "pubmed_ids": pmids,
        "accessions": accs,
    }


def initialize_run(profile: Profile, run_id: str, overwrite: bool = False) -> Path:
    ensure_runtime_dirs(profile)
    init_evidence_db(profile.runtime.evidence_db)
    path = run_db_path(profile, run_id)
    if path.exists() and not overwrite:
        raise FileExistsError(f"run already exists: {path}")
    if path.exists(): path.unlink()
    fp = source_fingerprint(profile)
    init_run_db(path, profile.profile_id, fp)
    cand = read_candidates(profile)
    meta = read_metadata(profile, [
        "enzyme_id","first_source_file","first_locus_tag","protein_ids","locus_tags","organisms","screening_sources"
    ])
    meta_map = {}
    if meta is not None and "enzyme_id" in meta.columns:
        for r in meta.to_dict("records"):
            meta_map[str(r["enzyme_id"])] = {k:v for k,v in r.items() if k != "enzyme_id" and str(v).strip()}
    con = connect(path)
    rows = []
    for r in cand.itertuples(index=False):
        md = meta_map.get(str(r.candidate_id), {})
        locator = ":".join(x for x in [str(md.get("first_source_file","")),str(md.get("first_locus_tag",""))] if x)
        rows.append((str(r.candidate_id),str(r.sequence_sha256),str(r.sequence_md5),profile.profile_id,locator,json.dumps(md,ensure_ascii=False)))
    con.executemany(
        "INSERT INTO candidates(candidate_id,sequence_sha256,sequence_md5,source_namespace,source_locator,metadata_json) VALUES(?,?,?,?,?,?)",
        rows,
    )
    focus = load_focus(profile)
    known = set(cand.candidate_id.astype(str))
    frows = [tuple(x) for x in focus[["candidate_id","focus_group","model_rank","force_stage"]].itertuples(index=False, name=None) if str(x[0]) in known]
    con.executemany("INSERT OR IGNORE INTO focus(candidate_id,focus_group,model_rank,force_stage) VALUES(?,?,?,?)", frows)
    set_meta(con,"candidate_count",len(cand)); set_meta(con,"focus_rows",len(frows)); set_meta(con,"created_utc",utcnow())
    con.commit(); con.close()
    return path


def run_stage1(profile: Profile, run_id: str) -> dict:
    path = run_db_path(profile,run_id)
    if not path.exists(): raise FileNotFoundError(path)
    econ = connect(profile.runtime.evidence_db); rcon = connect(path)
    require_role(econ, "public_evidence"); require_role(rcon, "candidate_run_state")
    md5s = [x[0] for x in rcon.execute("SELECT sequence_md5 FROM candidates ORDER BY candidate_id")]
    existing = {x[0] for x in econ.execute("SELECT sequence_md5 FROM uniparc_exact")}
    missing = [x for x in md5s if x not in existing]
    batches = list(chunks(missing, profile.policy.stage1_uniparc_batch_size))
    with ThreadPoolExecutor(max_workers=profile.policy.stage1_workers) as ex:
        fut = {ex.submit(uniprot.fetch_uniparc_exact,b):b for b in batches}
        for f in as_completed(fut):
            b=fut[f]; rows,headers=f.result()
            _upsert_uniparc(econ,rows,b,headers.get("X-UniProt-Release",""),headers.get("X-UniProt-Release-Date","")); econ.commit()
    # map accessions to run candidates
    rcon.execute("DELETE FROM candidate_accession")
    for c in rcon.execute("SELECT candidate_id,sequence_md5 FROM candidates"):
        u=econ.execute("SELECT accessions_json FROM uniparc_exact WHERE sequence_md5=? AND found=1",(c["sequence_md5"],)).fetchone()
        if u:
            for a in json.loads(u["accessions_json"] or "[]"):
                rcon.execute("INSERT OR IGNORE INTO candidate_accession(candidate_id,accession) VALUES(?,?)",(c["candidate_id"],a))
    rcon.commit()
    accessions=sorted({x[0] for x in rcon.execute("SELECT accession FROM candidate_accession")})
    have={x[0] for x in econ.execute("SELECT accession FROM uniprot_summary WHERE evidence_depth>=1")}
    missing_acc=[a for a in accessions if a not in have]
    batches=list(chunks(missing_acc,profile.policy.stage1_uniprot_batch_size))
    with ThreadPoolExecutor(max_workers=profile.policy.stage1_workers) as ex:
        fut={ex.submit(uniprot.fetch_uniprot_summary,b,False):b for b in batches}
        for f in as_completed(fut):
            rows,_=f.result(); _upsert_uniprot(econ,rows); econ.commit()
    force2={x[0] for x in rcon.execute("SELECT DISTINCT candidate_id FROM focus WHERE force_stage=2")}
    force3={x[0] for x in rcon.execute("SELECT DISTINCT candidate_id FROM focus WHERE force_stage=3")}
    rcon.execute("DELETE FROM stage_decision WHERE stage=1")
    rcon.execute("DELETE FROM tag_state")
    counts={"FINALIZE":0,"PROMOTE":0}
    for (cid,) in rcon.execute("SELECT candidate_id FROM candidates ORDER BY candidate_id"):
        a=_candidate_aggregate(econ,rcon,cid)
        e=SummaryEvidence(
            public_exact=a["public_exact"],best_pe_level=a["best_pe_level"],reviewed=a["reviewed"],
            has_pdb=a.get("has_pdb",False),functional_experimental=False,catalytic_experimental=False,
            kinetics_present=False,mass_spec_present=False,
        )
        level,reason=classify(e,1)
        dec,dreason=stage1_decision(Stage1Signals(
            public_exact=a["public_exact"],best_pe_level=a["best_pe_level"],reviewed=a["reviewed"],
            has_pdb=a.get("has_pdb",False),structured_experiment=False,
            force_stage2=cid in force2 or cid in force3,force_stage3=cid in force3,
        ),profile.policy.promote_pe_at_most)
        counts[dec]+=1
        rcon.execute("INSERT OR REPLACE INTO tag_state VALUES(?,?,?,?,?,?)",(cid,level,1,1 if dec=="FINALIZE" else 0,reason,utcnow()))
        rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,1,dec,dreason,utcnow()))
    if counts["PROMOTE"]>profile.policy.stage2_max_candidates:
        rcon.rollback(); raise RuntimeError(f"stage2 promotion {counts['PROMOTE']} exceeds cap {profile.policy.stage2_max_candidates}")
    set_meta(rcon,"stage1_completed_utc",utcnow()); set_meta(rcon,"stage1_counts",counts)
    rcon.commit(); econ.close(); rcon.close(); return counts


def _validate_pdbs(econ, accessions_to_pdbs: dict[str,list[str]], workers: int):
    existing={(x[0],x[1]) for x in econ.execute("SELECT accession,pdb_id FROM structure_evidence")}
    todo=[(a,p) for a,ps in accessions_to_pdbs.items() for p in ps if (a,p) not in existing]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut={ex.submit(rcsb.fetch_entry,p):(a,p) for a,p in todo}
        for f in as_completed(fut):
            a,p=fut[f]; z=f.result()
            econ.execute("INSERT OR REPLACE INTO structure_evidence(accession,pdb_id,experimental,method,resolution,fetched_utc) VALUES(?,?,?,?,?,?)",
                         (a,p,int(z["experimental"]),z["method"],z["resolution"],z["fetched_utc"]))
            econ.commit()


def run_stage2(profile: Profile, run_id: str) -> dict:
    path=run_db_path(profile,run_id); econ=connect(profile.runtime.evidence_db); rcon=connect(path)
    require_role(econ, "public_evidence"); require_role(rcon, "candidate_run_state")
    cids=[x[0] for x in rcon.execute("SELECT candidate_id FROM stage_decision WHERE stage=1 AND decision='PROMOTE'")]
    if len(cids)>profile.policy.stage2_max_candidates: raise RuntimeError("stage2 cap exceeded")
    accs=sorted({x[0] for cid in cids for x in rcon.execute("SELECT accession FROM candidate_accession WHERE candidate_id=?",(cid,))})
    need=[]
    for a in accs:
        row=econ.execute("SELECT evidence_depth FROM uniprot_summary WHERE accession=?",(a,)).fetchone()
        if row is None or int(row["evidence_depth"])<2: need.append(a)
    for b in chunks(need,profile.policy.stage1_uniprot_batch_size):
        rows,_=uniprot.fetch_uniprot_summary(b,True); _upsert_uniprot(econ,rows); econ.commit()
    acc_pdb={}
    for a in accs:
        row=econ.execute("SELECT pdb_ids_json FROM uniprot_summary WHERE accession=?",(a,)).fetchone()
        if row:
            ps=json.loads(row["pdb_ids_json"] or "[]")
            if ps: acc_pdb[a]=ps
    _validate_pdbs(econ,acc_pdb,profile.policy.stage2_workers)
    force3={x[0] for x in rcon.execute("SELECT DISTINCT candidate_id FROM focus WHERE force_stage=3")}
    rcon.execute("DELETE FROM stage_decision WHERE stage=2")
    counts={"FINALIZE":0,"PROMOTE":0}
    for cid in cids:
        a=_candidate_aggregate(econ,rcon,cid)
        # only experimentally verified PDB records count at stage2+
        verified_pdb=False
        for acc in a["accessions"]:
            if econ.execute("SELECT 1 FROM structure_evidence WHERE accession=? AND experimental=1 LIMIT 1",(acc,)).fetchone():
                verified_pdb=True; break
        e=SummaryEvidence(
            public_exact=a["public_exact"],best_pe_level=a["best_pe_level"],reviewed=a["reviewed"],
            has_pdb=verified_pdb,functional_experimental=a["functional_experimental"],
            catalytic_experimental=a["catalytic_experimental"],kinetics_present=a["kinetics_present"],
            mass_spec_present=a["mass_spec_present"],
        )
        level,reason=classify(e,2)
        dec,dreason=stage2_decision(level,bool(a["pubmed_ids"]),cid in force3)
        counts[dec]+=1
        rcon.execute("INSERT OR REPLACE INTO tag_state VALUES(?,?,?,?,?,?)",(cid,level,2,1 if dec=="FINALIZE" else 0,reason,utcnow()))
        rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,2,dec,dreason,utcnow()))
    if counts["PROMOTE"]>profile.policy.stage3_max_candidates:
        rcon.rollback(); raise RuntimeError(f"stage3 promotion {counts['PROMOTE']} exceeds cap {profile.policy.stage3_max_candidates}")
    set_meta(rcon,"stage2_completed_utc",utcnow()); set_meta(rcon,"stage2_counts",counts)
    rcon.commit(); econ.close(); rcon.close(); return counts


def _metadata_aliases(metadata_json: str) -> list[str]:
    md=json.loads(metadata_json or "{}")
    out=[]
    for k in ("protein_ids","locus_tags","first_locus_tag"):
        raw=str(md.get(k,"") or "")
        out += [x.strip() for x in re.split(r"[;,\s]+",raw) if x.strip()]
    return out


def _stage3_search_key(aliases: list[str]) -> str:
    normalized=sorted(set(str(x).strip() for x in aliases if str(x).strip()))
    return hashlib.sha256(json.dumps(normalized,ensure_ascii=False,separators=(",", ":")).encode()).hexdigest()


def _article_exact_identity(article: dict, aliases: list[str], linked: bool) -> tuple[bool,list[str]]:
    """Linked exact-sequence UniProt references are identity-confirmed by linkage.

    Alias-search results need an explicit accession/locus/protein-id occurrence in title/abstract.
    """
    if linked:
        return True, ["linked_exact_sequence_uniprot_reference"]
    blob=(str(article.get("title", ""))+" "+str(article.get("abstract", ""))).lower()
    matched=sorted(set(x for x in aliases if len(x)>=4 and x.lower() in blob))
    return bool(matched), matched


def _load_or_search_pmids(econ, aliases: list[str], max_results: int) -> list[str]:
    key=_stage3_search_key(aliases)
    row=econ.execute("SELECT pmids_json FROM literature_search WHERE search_key=?",(key,)).fetchone()
    if row is not None:
        return list(json.loads(row["pmids_json"] or "[]"))[:max_results]
    pmids=pubmed.search_pmids(aliases,max_results)
    econ.execute(
        "INSERT OR REPLACE INTO literature_search(search_key,aliases_json,pmids_json,fetched_utc) VALUES(?,?,?,?)",
        (key,json.dumps(sorted(set(aliases)),ensure_ascii=False),json.dumps(pmids),utcnow()),
    )
    econ.commit()
    return pmids


def _load_pubmed_articles(econ, pmids: list[str]) -> dict[str,dict]:
    if not pmids:
        return {}
    out={}
    missing=[]
    for pmid in sorted(set(str(x) for x in pmids if str(x))):
        row=econ.execute("SELECT * FROM pubmed_article WHERE pmid=?",(pmid,)).fetchone()
        if row is None:
            missing.append(pmid)
        else:
            out[pmid]={
                "pmid":pmid,"title":row["title"],"abstract":row["abstract"],
                "evidence_terms":json.loads(row["evidence_terms_json"] or "[]"),
                "suggested_touch":row["suggested_touch"],"fetched_utc":row["fetched_utc"],
            }
    # NCBI EFetch accepts many PMIDs at once; batching avoids one fetch per candidate.
    for b in chunks(missing,100):
        articles=pubmed.fetch_articles(b)
        returned={str(a["pmid"]):a for a in articles}
        for pmid in b:
            a=returned.get(str(pmid),{
                "pmid":str(pmid),"title":"","abstract":"","evidence_terms":[],
                "suggested_touch":None,"fetched_utc":utcnow(),
            })
            econ.execute(
                """INSERT OR REPLACE INTO pubmed_article
                   (pmid,title,abstract,evidence_terms_json,suggested_touch,fetched_utc)
                   VALUES(?,?,?,?,?,?)""",
                (a["pmid"],a["title"],a["abstract"],json.dumps(a["evidence_terms"]),a["suggested_touch"],a["fetched_utc"]),
            )
            out[str(pmid)]=a
        econ.commit()
    return out


def run_stage3(profile: Profile, run_id: str) -> dict:
    path=run_db_path(profile,run_id); econ=connect(profile.runtime.evidence_db); rcon=connect(path)
    require_role(econ, "public_evidence"); require_role(rcon, "candidate_run_state")
    cids=[x[0] for x in rcon.execute("SELECT candidate_id FROM stage_decision WHERE stage=2 AND decision='PROMOTE' ORDER BY candidate_id")]
    if len(cids)>profile.policy.stage3_max_candidates: raise RuntimeError("stage3 cap exceeded")

    # Phase A: resolve a bounded PMID set per candidate. Exact UniProt-linked PMIDs are preferred;
    # only candidates without linked PMIDs pay for alias search.
    plans={}
    all_pmids=set()
    search_calls=0
    for cid in cids:
        a=_candidate_aggregate(econ,rcon,cid)
        c=rcon.execute("SELECT sequence_md5,metadata_json FROM candidates WHERE candidate_id=?",(cid,)).fetchone()
        aliases=sorted(set(list(a["accessions"])+_metadata_aliases(c["metadata_json"] if c else "{}")))
        seqmd5=c["sequence_md5"]
        public_key=("uniprot:"+a["accessions"][0]) if a["accessions"] else ("seqmd5:"+seqmd5)
        linked_pmids=sorted(set(a["pubmed_ids"]))[:profile.policy.stage3_max_papers_per_candidate]
        if linked_pmids:
            pmids=linked_pmids
            origin="linked_uniprot"
        else:
            before=econ.execute("SELECT 1 FROM literature_search WHERE search_key=?",(_stage3_search_key(aliases),)).fetchone() if aliases else None
            pmids=_load_or_search_pmids(econ,aliases,profile.policy.stage3_max_papers_per_candidate) if aliases else []
            if before is None and aliases: search_calls+=1
            origin="alias_search"
        plans[cid]={
            "aggregate":a,"aliases":aliases,"public_key":public_key,
            "pmids":pmids,"origin":origin,"linked_pmids":set(linked_pmids),
        }
        all_pmids.update(pmids)

    # Phase B: fetch articles globally, not per candidate.
    articles=_load_pubmed_articles(econ,sorted(all_pmids))

    # Phase C: deterministic candidate-level evidence adjudication.
    rcon.execute("DELETE FROM stage_decision WHERE stage=3")
    rcon.execute("DELETE FROM candidate_evidence_ref WHERE stage=3")
    counts={"FINALIZE":0,"REVIEW":0,"AUTO_UPGRADE":0,"SEARCH_CALLS":search_calls,"UNIQUE_PMIDS":len(all_pmids)}
    for cid in cids:
        plan=plans[cid]
        current=rcon.execute("SELECT touch_level FROM tag_state WHERE candidate_id=?",(cid,)).fetchone()[0]
        best=current; unconfirmed_hint=False
        for pmid in plan["pmids"]:
            art=articles.get(str(pmid))
            if not art:
                continue
            linked=str(pmid) in plan["linked_pmids"]
            exact,matched=_article_exact_identity(art,plan["aliases"],linked)
            source="PubMed:linked_uniprot" if linked else "PubMed:alias_search"
            econ.execute(
                """INSERT OR REPLACE INTO literature_evidence
                   (public_key,pmid,title,abstract,evidence_terms_json,exact_identity_confirmed,suggested_touch,source,fetched_utc)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (plan["public_key"],art["pmid"],art["title"],art["abstract"],json.dumps(art["evidence_terms"]),1 if exact else 0,
                 art["suggested_touch"],source,art["fetched_utc"]),
            )
            rcon.execute(
                "INSERT OR IGNORE INTO candidate_evidence_ref(candidate_id,stage,evidence_kind,public_key,external_ref) VALUES(?,3,?,?,?)",
                (cid,source,plan["public_key"],art["pmid"]),
            )
            if art["suggested_touch"]:
                if exact and int(art["suggested_touch"][1:])>int(best[1:]):
                    best=art["suggested_touch"]
                elif not exact:
                    unconfirmed_hint=True
        econ.commit()
        if best!=current:
            reason="deep literature pass found identity-confirmed experimental evidence"
            rcon.execute("INSERT OR REPLACE INTO tag_state VALUES(?,?,?,?,?,?)",(cid,best,3,1,reason,utcnow()))
            rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,3,"FINALIZE",reason,utcnow()))
            counts["AUTO_UPGRADE"]+=1; counts["FINALIZE"]+=1
        elif unconfirmed_hint:
            reason="literature contains experimental terms but candidate identity requires review"
            rcon.execute("UPDATE tag_state SET evidence_stage=3,finalized=0,reason=?,updated_utc=? WHERE candidate_id=?",(reason,utcnow(),cid))
            rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,3,"REVIEW",reason,utcnow()))
            counts["REVIEW"]+=1
        else:
            reason="deep targeted search found no identity-confirmed evidence sufficient to change current T level"
            rcon.execute("UPDATE tag_state SET evidence_stage=3,finalized=1,reason=?,updated_utc=? WHERE candidate_id=?",(reason,utcnow(),cid))
            rcon.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",(cid,3,"FINALIZE",reason,utcnow()))
            counts["FINALIZE"]+=1
        rcon.commit()
    set_meta(rcon,"stage3_completed_utc",utcnow()); set_meta(rcon,"stage3_counts",counts)
    rcon.commit(); econ.close(); rcon.close(); return counts


def apply_manual_overrides(profile: Profile, run_id: str, csv_path: Path) -> int:
    d=pd.read_csv(csv_path,dtype=str).fillna("")
    req={"candidate_id","touch_level","evidence_ref"}
    if not req.issubset(d.columns): raise ValueError(f"override CSV requires {sorted(req)}")
    if not d.touch_level.isin([f"T{i}" for i in range(6)]).all(): raise ValueError("invalid T level")
    con=connect(run_db_path(profile,run_id)); require_role(con, "candidate_run_state"); n=0
    for r in d.to_dict("records"):
        cid=r["candidate_id"]
        if not con.execute("SELECT 1 FROM candidates WHERE candidate_id=?",(cid,)).fetchone(): raise KeyError(cid)
        con.execute("INSERT OR REPLACE INTO manual_override VALUES(?,?,?,?,?,?)",
                    (cid,r["touch_level"],r["evidence_ref"],r.get("note",""),r.get("reviewer",""),utcnow()))
        con.execute("INSERT OR REPLACE INTO tag_state VALUES(?,?,?,?,?,?)",
                    (cid,r["touch_level"],3,1,"manual evidence review override",utcnow()))
        con.execute("INSERT OR REPLACE INTO stage_decision VALUES(?,?,?,?,?)",
                    (cid,3,"FINALIZE","manual evidence review override",utcnow()))
        n+=1
    con.commit(); con.close(); return n


def status(profile: Profile, run_id: str) -> dict:
    con=connect(run_db_path(profile,run_id),readonly=True); require_role(con, "candidate_run_state")
    out={
        "candidate_count":con.execute("SELECT count(*) FROM candidates").fetchone()[0],
        "focus_stage2_unique":con.execute("SELECT count(DISTINCT candidate_id) FROM focus WHERE force_stage=2").fetchone()[0],
        "focus_stage3_unique":con.execute("SELECT count(DISTINCT candidate_id) FROM focus WHERE force_stage=3").fetchone()[0],
        "touch_counts":{r[0]:r[1] for r in con.execute("SELECT touch_level,count(*) FROM tag_state GROUP BY touch_level ORDER BY touch_level")},
        "stage1":{r[0]:r[1] for r in con.execute("SELECT decision,count(*) FROM stage_decision WHERE stage=1 GROUP BY decision")},
        "stage2":{r[0]:r[1] for r in con.execute("SELECT decision,count(*) FROM stage_decision WHERE stage=2 GROUP BY decision")},
        "stage3":{r[0]:r[1] for r in con.execute("SELECT decision,count(*) FROM stage_decision WHERE stage=3 GROUP BY decision")},
        "finalized":con.execute("SELECT count(*) FROM tag_state WHERE finalized=1").fetchone()[0],
        "needs_review":con.execute("SELECT count(*) FROM stage_decision WHERE stage=3 AND decision='REVIEW'").fetchone()[0],
    }
    con.close(); return out


def export_tags(profile: Profile, run_id: str, out_path: Path) -> Path:
    con=connect(run_db_path(profile,run_id),readonly=True); require_role(con, "candidate_run_state")
    d=pd.read_sql_query(
        """SELECT c.candidate_id,c.sequence_sha256,c.sequence_md5,t.touch_level,t.evidence_stage,t.finalized,t.reason,
                  c.source_locator,c.metadata_json
           FROM candidates c LEFT JOIN tag_state t USING(candidate_id) ORDER BY c.candidate_id""",con)
    con.close(); out_path.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(out_path,sep="\t" if ".tsv" in out_path.name else ",",index=False,compression="gzip" if out_path.suffix==".gz" else None)
    return out_path
