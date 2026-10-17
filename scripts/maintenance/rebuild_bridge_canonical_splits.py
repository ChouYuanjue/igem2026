#!/usr/bin/env python3
"""Rebuild frozen BRIDGE train/validation/test membership without model weights or GPU."""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "benchmarks/bridge/canonical-v1/manifest.json"
KEYS = ("protein_id", "reaction_id")
DEV_COLUMNS = (*KEYS, "partition", "novelty", "difficulty_stratum", "source")

def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def load_csv(path: Path):
    data = path.read_bytes()
    plain = gzip.decompress(data) if path.suffix == ".gz" else data
    rows = list(csv.DictReader(io.StringIO(plain.decode("utf-8"))))
    if not rows or not set(KEYS).issubset(rows[0]):
        raise ValueError("invalid protein/reaction CSV: " + str(path))
    return data, plain, rows

def pair(row):
    return row["protein_id"], row["reaction_id"]

def unique(rows, label):
    found = {}
    for row in rows:
        p = pair(row)
        if not all(p) or p in found:
            raise ValueError(label + ": empty/duplicate relationship " + repr(p))
        found[p] = row
    return found

def fraction(protein, salt):
    return int.from_bytes(hashlib.blake2b((salt + protein).encode(), digest_size=8).digest(), "big") / 2**64

def parent_split(protein):
    if fraction(protein, "IGEM26:E2R:DEV:20261010") >= .10:
        return "evaluation"
    if fraction(protein, "IGEM26:E2R:INTERNALVALID:20261010") < .25:
        return "gate_validation"
    return "gate_learn"

def external_split(protein, parent_membership):
    if protein in parent_membership:
        return parent_membership[protein]
    v = hashlib.blake2b(("IGEM26:E2R:EXT_VALID:20261010:" + protein).encode(), digest_size=8).digest()
    return "gate_validation" if int.from_bytes(v, "big") % 4 == 0 else "gate_learn"

def bucket(degree):
    if degree == 0: return "0"
    if degree == 1: return "1"
    if degree <= 3: return "2-3"
    if degree <= 7: return "4-7"
    if degree <= 15: return "8-15"
    return "16+"

def exposure(p, r, pd, rd):
    a,b = pd[p],rd[r]
    if a and b:
        return "both_seen", ("both_seen|min_degree=1" if min(a,b) == 1 else "both_seen|min_degree=2+")
    if b: return "protein_cold","protein_cold|seen_degree=" + bucket(b)
    if a: return "reaction_cold","reaction_cold|seen_degree=" + bucket(a)
    return "double_cold","double_cold|degree=0"

def csv_bytes(rows, columns):
    s=io.StringIO(newline="")
    w=csv.DictWriter(s,fieldnames=columns,lineterminator="\n",extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return s.getvalue().encode("utf-8")

def split(train_rows, parent_rows, rhea_rows):
    train=unique(train_rows,"train")
    parent=unique(parent_rows,"parent")
    rhea=unique(rhea_rows,"Rhea")
    if train.keys() & parent.keys():
        raise ValueError("train/parent positive association overlap")
    pd=collections.Counter(p for p,r in train)
    rd=collections.Counter(r for p,r in train)
    membership={p:parent_split(p) for p,r in parent}
    test=[dict(zip(KEYS,k)) for k in sorted(k for k in parent if membership[k[0]]=="evaluation")]
    dev={k:(membership[k[0]],"parent_dev") for k in parent if membership[k[0]]!="evaluation"}
    clean_rhea=[k for k in rhea if k not in parent and k not in train]
    skipped=0
    for k in clean_rhea:
        group=external_split(k[0],membership)
        if group=="evaluation":
            skipped+=1
        else:
            if k in dev: raise ValueError("duplicate development association")
            dev[k]=(group,"rhea_independent")
    val=[]
    for (p,r),(group,source) in sorted(dev.items()):
        novelty,stratum=exposure(p,r,pd,rd)
        val.append(dict(protein_id=p,reaction_id=r,partition="learn" if group=="gate_learn" else "validation",
                        novelty=novelty,difficulty_stratum=stratum,source=source))
    if train.keys() & dev.keys() or train.keys() & {pair(x) for x in test}:
        raise ValueError("train/development/test positive association overlap")
    val_p={x["protein_id"] for x in val}
    test_p={x["protein_id"] for x in test}
    if val_p & test_p:raise ValueError("shared E2R protein query between validation and test")
    report=dict(
      training_relations=len(train),validation_relations=len(val),test_relations=len(test),
      parent_relations=len(parent),independent_rhea_relations=len(clean_rhea),
      independent_rhea_rejected_by_query=skipped,
      validation_source_counts=dict(collections.Counter(x["source"] for x in val)),
      development_partition_counts=dict(collections.Counter(x["partition"] for x in val)),
      validation_protein_queries=len(val_p),validation_reaction_queries=len({x["reaction_id"] for x in val}),
      test_protein_queries=len(test_p),test_reaction_queries=len({x["reaction_id"] for x in test}),
      validation_test_shared_reaction_queries=len({x["reaction_id"] for x in val}&{x["reaction_id"] for x in test})
    )
    return val,test,report

def run(train_path,parent_path,rhea_path,lock_path,output=None):
    lock=json.loads(lock_path.read_text(encoding="utf-8"))
    sources={}
    for label,path in (("train",train_path),("parent",parent_path),("rhea",rhea_path)):
        raw,plain,rows=load_csv(path)
        if digest(raw)!=lock["source_inputs"][label]["sha256"]:
            raise ValueError(label + " source bytes differ from locked SHA256")
        sources[label]=(plain,rows)
    val,test,report=split(sources["train"][1],sources["parent"][1],sources["rhea"][1])
    data={"train.csv.gz":sources["train"][0],
          "validation.csv.gz":csv_bytes(val,DEV_COLUMNS),
          "test.csv.gz":csv_bytes(test,KEYS)}
    for name,content in data.items():
        expected=lock["splits"][name]
        if len(content.splitlines())-1 != expected["rows"] or digest(content)!=expected["csv_sha256"]:
            raise ValueError(name + " differs from frozen content SHA256 or row count")
    for k,v in lock["checks"].items():
        if report[k]!=v:raise ValueError(k+" failed: "+repr(report[k]))
    if output:
        output.mkdir(parents=True,exist_ok=True)
        for name,content in data.items():
            (output/name).write_bytes(gzip.compress(content,compresslevel=9,mtime=0))
        (output/"rebuild-report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path,default=ROOT/"data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv")
    parser.add_argument("--parent",type=Path,default=ROOT/"results/bridge_relation_unseen_max_v3/targets.csv")
    parser.add_argument("--rhea",type=Path,default=ROOT/"results/bridge_rhea_longitudinal_growth_v4/targets.csv")
    parser.add_argument("--lock",type=Path,default=LOCK)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.train,args.parent,args.rhea,args.lock,args.output),indent=2))

if __name__=="__main__":
    main()
