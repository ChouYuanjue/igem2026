from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
SUPPORT=ROOT/'results/enzymarc_open_world_v1/support'; OUT=ROOT/'results/enzymarc_open_world_v1/sequence_form_gate'

def write_fasta(path:Path, rows):
    with path.open('w') as f:
        for ident,seq in rows:
            f.write(f'>{ident}\n{seq}\n')

def prepare(out:Path):
    out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(SUPPORT/'parents.csv',dtype=str).fillna('').sort_values('parent_accession')
    d=pd.read_csv(SUPPORT/'decoys.csv',dtype=str).fillna(''); d=d[d.category.eq('catalytic_residue')].sort_values('parent_accession')
    common=sorted(set(p.parent_accession)&set(d.parent_accession)); pm=p.set_index('parent_accession'); dm=d.set_index('parent_accession')
    write_fasta(out/'parents.fasta',[(x,str(pm.loc[x,'parent_sequence'])) for x in common]); write_fasta(out/'catalytic_residue_decoys.fasta',[(x,str(dm.loc[x,'decoy_sequence'])) for x in common])
    result={'status':'fastas_ready_no_alignment_statistics_read','parent_query_count':len(common),'thresholds':{'fident':0.8,'qcov':0.8,'tcov':0.8},'labels_used':False,'model_scores_read':False}
    (out/'prepare_manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def finalize(out:Path):
    hits=pd.read_csv(out/'matched_search.tsv',sep='\t',names=['query','target','fident','qcov','tcov','alnlen'],dtype={'query':str,'target':str})
    for c in ['fident','qcov','tcov','alnlen']: hits[c]=pd.to_numeric(hits[c],errors='coerce')
    # MMseqs fident is normally fractional. Tolerate percentage-format builds deterministically.
    if hits.fident.dropna().max()>1.0: hits['fident']=hits.fident/100.0
    matched=hits[hits['query'].eq(hits['target'])].sort_values(['query','fident','qcov','tcov'],ascending=[True,False,False,False]).drop_duplicates('query')
    prepared=json.loads((out/'prepare_manifest.json').read_text()); all_ids=[]
    with (out/'parents.fasta').open() as f:
        for line in f:
            if line.startswith('>'): all_ids.append(line[1:].strip())
    m=matched.set_index('query'); rows=[]
    for x in all_ids:
        if x in m.index:
            r=m.loc[x]; fid=float(r.fident); qc=float(r.qcov); tc=float(r.tcov); ok=fid>=.8 and qc>=.8 and tc>=.8
            rows.append({'parent_accession':x,'matched_hit':True,'fident':fid,'qcov':qc,'tcov':tc,'alnlen':int(r.alnlen),'eligible':ok})
        else: rows.append({'parent_accession':x,'matched_hit':False,'fident':None,'qcov':None,'tcov':None,'alnlen':None,'eligible':False})
    a=pd.DataFrame(rows); a.to_csv(out/'audit.csv',index=False); elig=a[a.eligible].parent_accession.astype(str); pd.DataFrame({'parent_accession':elig}).to_csv(out/'eligible_parents.csv',index=False)
    support_d=pd.read_csv(SUPPORT/'decoys.csv',dtype=str).fillna(''); support_r=pd.read_csv(SUPPORT/'parent_reaction_relations.csv',dtype=str).fillna(''); eset=set(elig)
    result={'status':'sequence_form_gate_frozen','mapped_parents_before':len(a),'eligible_parents':int(a.eligible.sum()),'excluded_parents':int((~a.eligible).sum()),'missing_matched_hits':int((~a.matched_hit).sum()),'eligible_decoys':int(support_d.parent_accession.isin(eset).sum()),'eligible_relations':int(support_r.parent_accession.isin(eset).sum()),'thresholds':prepared['thresholds'],'model_scores_read':False,'selection_allowed':False}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['prepare','finalize']); ap.add_argument('--output-dir',type=Path,default=OUT); a=ap.parse_args(); out=a.output_dir.resolve(); print(json.dumps(prepare(out) if a.action=='prepare' else finalize(out),indent=2))
if __name__=='__main__': main()
