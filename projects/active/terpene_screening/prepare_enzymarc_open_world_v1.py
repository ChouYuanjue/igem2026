from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'data/external/enzymarc_v1'; OUT=ROOT/'results/enzymarc_open_world_v1/support'
CLEAN=ROOT/'data/external/enzymecage_current/catalyst_features/clean2023/training_pairs.csv'
SEQ=ROOT/'data/catalyst_candidate_universes/general_merged/protein_sequences.tsv'
RXN=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_center_v1/entries.csv'
FILES={'catalytic':'decoys_cataliticals.fasta','radius_5A':'decoys_5A.fasta','radius_10A':'decoys_10A.fasta','radius_15A':'decoys_15A.fasta'}

def sha256_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def parse_fasta(path:Path,category:str):
 header=None; seq=[]
 def emit(h,s):
  parts=h.split('|');
  if len(parts)<3: raise ValueError(f'bad EnzymARC header {h}')
  acc=parts[0].strip(); ec=next((x[3:] for x in parts[1:] if x.startswith('EC:')),'')
  return {'parent_accession':acc,'category':category,'original_ec':ec,'decoy_sequence':''.join(s).strip().upper()}
 with path.open() as f:
  for line in f:
   line=line.strip()
   if not line: continue
   if line.startswith('>'):
    if header is not None: yield emit(header,seq)
    header=line[1:]; seq=[]
   else: seq.append(line)
 if header is not None: yield emit(header,seq)

def build(raw:Path,out:Path):
 out.mkdir(parents=True,exist_ok=True); clean=pd.read_csv(CLEAN,dtype=str).fillna(''); seq=pd.read_csv(SEQ,sep='\t',dtype=str).fillna(''); registry=set(pd.read_csv(RXN,dtype=str).reaction_id.astype(str)); seqmap=dict(zip(seq.protein_id.astype(str),seq.sequence.astype(str)))
 clean=clean[clean.reaction_id.isin(registry)&clean.protein_id.isin(seqmap)].drop_duplicates(); relevant=set(clean.protein_id.astype(str)); decoys=[]; raw_counts={}; hashes={}
 for cat,name in FILES.items():
  p=raw/name
  if not p.exists(): raise FileNotFoundError(p)
  hashes[name]=sha256_file(p); total=0; kept=0
  for row in parse_fasta(p,cat):
   total+=1
   if row['parent_accession'] in relevant: decoys.append(row); kept+=1
  raw_counts[cat]={'total':total,'mapped_clean2023_parents':kept}
 d=pd.DataFrame(decoys); d=d.drop_duplicates(['parent_accession','category']).sort_values(['parent_accession','category']); parents=sorted(set(d.parent_accession)); rel=clean[clean.protein_id.isin(parents)].rename(columns={'protein_id':'parent_accession'})[['parent_accession','reaction_id']].drop_duplicates().sort_values(['parent_accession','reaction_id']); parent=pd.DataFrame({'parent_accession':parents,'parent_sequence':[seqmap[x] for x in parents]})
 parent.to_csv(out/'parents.csv',index=False); d.to_csv(out/'decoys.csv',index=False); rel.to_csv(out/'parent_reaction_relations.csv',index=False)
 per=d.category.value_counts().to_dict(); formal=len(parents)>=50 and len(d)>=200
 result={'status':'support_ready_formal_claim_allowed' if formal else 'support_underpowered','raw_files_sha256':hashes,'raw_counts':raw_counts,'mapped_parent_count':len(parents),'mapped_decoy_count':len(d),'mapped_relation_count':len(rel),'mapped_decoys_by_category':{k:int(per.get(k,0)) for k in FILES},'minimums':{'parents':50,'decoys':200},'formal_claim_allowed':formal,'model_scores_read':False,'selection_allowed':False}
 (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n'); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--raw-dir',type=Path,default=RAW); ap.add_argument('--output-dir',type=Path,default=OUT); a=ap.parse_args(); print(json.dumps(build(a.raw_dir.resolve(),a.output_dir.resolve()),indent=2))
if __name__=='__main__': main()
