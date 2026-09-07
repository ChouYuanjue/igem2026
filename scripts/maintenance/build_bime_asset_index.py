"""Index existing BiME assets without inference, training, moving or deleting data.

Explicit selection.json is reviewed policy, never inferred from latest filenames.
The inventory is conservative: unclassified assets are not deletion candidates.
"""
from __future__ import annotations
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reproducibility/bime_rank'
PATH_RE = re.compile(r'(?:/home/s241850073/igem2026/)?(?:results|data|configs|projects|external_models|external_repos|external_runtime|local_candidate_libraries|scripts|reproducibility)/[^\s\"\'`<>;,|(){}\[\]]+')
TEXT_SUFFIX = {'.json', '.yaml', '.yml', '.py', '.md', '.sh', '.toml'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def refpaths(text):
    return sorted({m.group().removeprefix(str(ROOT) + '/').split('::',1)[0].rstrip('.:') for m in PATH_RE.finditer(text)})

def main():
    selection = json.loads((OUT / 'selection.json').read_text())
    selected = set()
    for c in selection['claims'].values():
        selected.update([c['primary'], *c['assets']])
    queue = collections.deque(sorted(selected))
    visited, edges, missing = set(), [], []
    # Follow concrete manifest/config references. Directory references retain all
    # members but only immediate manifest/config files are traversed as graphs.
    while queue:
        rel = queue.popleft()
        if rel in visited or rel.startswith('reproducibility/bime_rank/'):
            continue
        visited.add(rel)
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        if p.is_dir():
            for name in ('manifest.json','config.json','summary.json','result.json','deployment.json'):
                if (p/name).is_file():queue.append(str((p/name).relative_to(ROOT)))
            continue
        if p.suffix not in {'.json','.yaml','.yml'} or p.stat().st_size > 8*1024*1024:
            continue
        for target in refpaths(p.read_text(errors='replace')):
            if target == rel:continue
            edges.append({'source':rel,'target':target,'exists':(ROOT/target).exists()})
            if (ROOT/target).exists():queue.append(target)
            else:missing.append(target)
    # Inventory all asset-bearing roots (metadata only for large files).
    roots = ['results','data','external_models','external','external_repos','external_runtime','downloads','local_candidate_libraries','reports']
    groups = collections.defaultdict(lambda:{'files':0,'bytes':0,'latest_mtime_ns':0,'symlinks':0})
    critical_files = set(p for p in visited if (ROOT/p).is_file())
    selected_dirs = [p for p in visited if (ROOT/p).is_dir()]
    big, symlinks, protected = [], [], []
    for root in roots:
        for base, dirs, files in os.walk(ROOT/root,followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'.git','.venv','node_modules'}]
            for name in files:
                p=Path(base)/name; rel=str(p.relative_to(ROOT))
                try:s=p.lstat()
                except OSError:continue
                key='/'.join(p.relative_to(ROOT).parts[:2]);g=groups[key]
                g['files']+=1;g['bytes']+=s.st_size;g['latest_mtime_ns']=max(g['latest_mtime_ns'],s.st_mtime_ns)
                if p.is_symlink():
                    g['symlinks']+=1;symlinks.append({'path':rel,'target':os.readlink(p),'exists':p.exists()});continue
                is_protected=rel in critical_files or any(rel.startswith(x+'/') for x in selected_dirs)
                if is_protected:
                    entry={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
                    # Hash model/config/summary and candidate identity tables; large
                    # embeddings retained by manifest identities plus size, not re-read.
                    if s.st_size<=16*1024*1024 and (p.suffix in TEXT_SUFFIX|{'.csv','.tsv','.fasta','.pkl','.joblib'}):
                        entry['sha256']=sha(p)
                    else:entry['integrity']='metadata_only; existing manifest hashes retained; full blob hash not recomputed'
                    protected.append(entry)
                if s.st_size>=128*1024*1024:big.append({'path':rel,'bytes':s.st_size,'protected':is_protected})
    # Concrete code/document reference audit, without treating every old script
    # as a current production caller.
    references=[]
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for rel in tracked:
        p=ROOT/rel
        if not p.is_file() or p.suffix not in TEXT_SUFFIX or p.stat().st_size>2*1024*1024:continue
        text=p.read_text(errors='replace')
        for target in refpaths(text):
            replacement=next((v for k,v in selection['superseded'].items() if target==k or target.startswith(k+'/')),None)
            if replacement:references.append({'source':rel,'target':target,'canonical_claim':replacement,'disposition':'historical reproducibility reference; not current claim authority'})
    claims={}
    for key,c in selection['claims'].items():
        p=ROOT/c['primary']
        if not p.is_file():raise RuntimeError(f'Missing canonical primary: {p}')
        claims[key]={**c,'sha256':sha(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns}
    for rel in visited:
        p=ROOT/rel
        if p.is_file() and not any(x['path']==rel for x in protected):
            protected.append({'path':rel,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':sha(p)})
    inventory=[]
    for key,g in sorted(groups.items()):
        contains=any(p==key or p.startswith(key+'/') for p in visited)
        superseded=key in selection['superseded']
        category='current_or_reproducibility_dependency' if contains else ('historical_superseded' if superseded else 'unclassified_preserve')
        inventory.append({'path':key,'category':category,**g})
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    lock={'schema_version':1,'audited_at':now,'source_commit':selection['source_commit'],'claims':claims,'superseded':selection['superseded'],'policy':selection['policy']}
    for name,d in [('canonical.json',lock),('dependency_graph.json',{'paths':sorted(visited),'edges':edges,'unresolved_references':sorted(set(missing))}),('protected_files.json',sorted(protected,key=lambda x:x['path'])),('inventory.json',inventory),('large_files.json',sorted(big,key=lambda x:-x['bytes'])),('symlinks.json',symlinks),('historical_references.json',references)]:
        (OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'claims':len(claims),'dependency_paths':len(visited),'protected_files':len(protected),'inventory_groups':len(inventory),'files':sum(g['files'] for g in groups.values()),'bytes':sum(g['bytes'] for g in groups.values()),'unresolved_references':len(set(missing)),'historical_references':len(references)}))

if __name__=='__main__':main()
