from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_profile
from .migrate_v2 import bootstrap_from_v2
from .pipeline import (
    apply_manual_overrides, export_tags, initialize_run, run_stage1, run_stage2, run_stage3,
    status,
)
from .safety import assert_runtime_separation


def main(argv=None):
    ap=argparse.ArgumentParser(prog="experimental-touch-cascade")
    ap.add_argument("--profile",required=True,type=Path,help="YAML profile; switching profile switches source/runtime DBs")
    sub=ap.add_subparsers(dest="cmd",required=True)
    sub.add_parser("doctor")
    p=sub.add_parser("init"); p.add_argument("--run-id",required=True); p.add_argument("--overwrite",action="store_true")
    p=sub.add_parser("bootstrap-v2"); p.add_argument("--run-id",required=True); p.add_argument("--v2-tags",required=True,type=Path); p.add_argument("--overwrite",action="store_true")
    for name in ("stage1","stage2","stage3","status"):
        p=sub.add_parser(name); p.add_argument("--run-id",required=True)
    p=sub.add_parser("export"); p.add_argument("--run-id",required=True); p.add_argument("--out",required=True,type=Path)
    p=sub.add_parser("apply-overrides"); p.add_argument("--run-id",required=True); p.add_argument("--csv",required=True,type=Path)
    a=ap.parse_args(argv); profile=load_profile(a.profile)
    assert_runtime_separation(profile)
    if a.cmd=="doctor":
        out={
            "profile_id":profile.profile_id,
            "candidate_source":str(profile.source.candidates_path),
            "metadata_source":str(profile.source.metadata_path) if profile.source.metadata_path else None,
            "evidence_db":str(profile.runtime.evidence_db),
            "run_root":str(profile.runtime.run_root),
            "cache_root":str(profile.runtime.cache_root),
            "ranking_focus":str(profile.focus.ranking_path) if profile.focus.ranking_path else None,
            "stage2_top_k_per_group":profile.focus.stage2_top_k_per_group,
            "stage3_top_k_per_group":profile.focus.stage3_top_k_per_group,
            "separation_check":"PASS",
        }
    elif a.cmd=="init": out={"run_db":str(initialize_run(profile,a.run_id,a.overwrite))}
    elif a.cmd=="bootstrap-v2": out=bootstrap_from_v2(profile,a.run_id,a.v2_tags,a.overwrite)
    elif a.cmd=="stage1": out=run_stage1(profile,a.run_id)
    elif a.cmd=="stage2": out=run_stage2(profile,a.run_id)
    elif a.cmd=="stage3": out=run_stage3(profile,a.run_id)
    elif a.cmd=="status": out=status(profile,a.run_id)
    elif a.cmd=="export": out={"output":str(export_tags(profile,a.run_id,a.out))}
    elif a.cmd=="apply-overrides": out={"applied":apply_manual_overrides(profile,a.run_id,a.csv)}
    else: raise AssertionError(a.cmd)
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
