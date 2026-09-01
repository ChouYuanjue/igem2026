from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.train_dual_tower_cold import ModelConfig, TerpeneDualTower

RHEA=ROOT/'data/external/reactzyme/rhea_molecules.tsv'
FEATURE=ROOT/'data/catalyst_candidate_universes/general_merged/reaction_features/drfp_categorical_rdkitplus_v1'
BASE=ROOT/'results/enzymecage_cleanroom_rdkitplus_v1'
OUT=ROOT/'results/reactzyme_native_bag_adapter_v1'
GEN=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def seed_all(s:int): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None

def bag_feature(substrate:str, product:str)->tuple[np.ndarray,int,int]:
    rows=[]; invalid=0
    for token in (str(substrate)+'.'+str(product)).split('.'):
        token=token.strip()
        if not token: continue
        mol=Chem.MolFromSmiles(token)
        if mol is None:
            invalid+=1; continue
        arr=np.asarray(GEN.GetFingerprintAsNumPy(mol),dtype=np.float32)
        rows.append(arr)
    if not rows: return np.zeros(4096,dtype=np.float32),0,invalid
    x=np.stack(rows)
    return np.concatenate([x.mean(0),x.max(0)]).astype(np.float32),len(rows),invalid

class BagAdapter(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(4096),nn.Linear(4096,512),nn.GELU(),nn.Linear(512,320))
    def forward(self,x): return F.normalize(self.net(x),p=2,dim=-1)

def load_teacher(fold:int,device:torch.device):
    p=BASE/f'fold{fold}/models/production_seed20260723.pt'; payload=torch.load(p,map_location=device,weights_only=False)
    cfg=ModelConfig(**payload['model_config']); model=TerpeneDualTower(cfg).to(device); model.load_state_dict(payload['model_state_dict']); model.eval()
    for x in model.parameters(): x.requires_grad_(False)
    return model,p,payload

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True,choices=[0,1,2]); ap.add_argument('--output',type=Path,default=OUT); a=ap.parse_args()
    fold=a.fold; seed=20260901; seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trp=BASE/f'fold{fold}/training_pairs.csv'; dvp=BASE/f'fold{fold}/dev_pairs.csv'
    tr_ids=sorted(set(pd.read_csv(trp,dtype=str).reaction_id.astype(str))); dv_ids=sorted(set(pd.read_csv(dvp,dtype=str).reaction_id.astype(str)))
    assert not (set(tr_ids)&set(dv_ids))
    bags=pd.read_csv(RHEA,sep='\t',dtype=str).fillna('').set_index('Rhea ID'); entries=pd.read_csv(FEATURE/'entries.csv',dtype=str).sort_values('row'); mat=np.load(FEATURE/'reaction_feature_matrix.npy').astype(np.float32); row={r:i for i,r in enumerate(entries.reaction_id.astype(str))}
    missing=[r for r in tr_ids+dv_ids if r not in bags.index or r not in row]; assert not missing,missing[:5]
    all_ids=tr_ids+dv_ids; feats=[]; audit=[]
    for rid in all_ids:
        v,n,bad=bag_feature(bags.loc[rid,'substrate'],bags.loc[rid,'product']); feats.append(v); audit.append({'reaction_id':rid,'split':'train' if rid in set(tr_ids) else 'dev','valid_molecules':n,'invalid_molecules':bad,'zero_feature':bool(n==0)})
    feats=np.stack(feats).astype(np.float32); ntr=len(tr_ids)
    teacher,ckpt,payload=load_teacher(fold,device)
    with torch.no_grad():
        teacher_lat=[]
        for s in range(0,len(all_ids),512):
            x=np.stack([mat[row[r]] for r in all_ids[s:s+512]])
            teacher_lat.append(teacher.encode_reactions(torch.from_numpy(x).to(device)).cpu().numpy())
    teacher_lat=np.concatenate(teacher_lat).astype(np.float32)
    model=BagAdapter().to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); rng=np.random.default_rng(seed); hist=[]
    tx=torch.from_numpy(feats[:ntr]); ty=torch.from_numpy(teacher_lat[:ntr])
    for epoch in range(1,81):
        model.train(); order=rng.permutation(ntr); losses=[]
        for s in range(0,ntr,256):
            idx=torch.from_numpy(order[s:s+256]); x=tx[idx].to(device); y=ty[idx].to(device); pred=model(x); loss=(1-(pred*y).sum(-1)).mean(); opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        hist.append({'epoch':epoch,'train_cosine_loss':float(np.mean(losses))})
    out=a.output/f'fold{fold}'; out.mkdir(parents=True,exist_ok=True); model.eval()
    torch.save({'model_type':'reactzyme_native_bag_adapter_v1','model_state_dict':model.state_dict(),'input_dim':4096,'hidden_dim':512,'output_dim':320,'fold':fold,'seed':seed},out/'adapter.pt')
    np.save(out/'bag_features.npy',feats); np.save(out/'teacher_reaction_latents.npy',teacher_lat); pd.DataFrame(audit).to_csv(out/'bag_audit.csv',index=False); pd.DataFrame(hist).to_csv(out/'training_history.csv',index=False); pd.DataFrame({'reaction_id':all_ids,'split':['train']*ntr+['dev']*len(dv_ids),'row':range(len(all_ids))}).to_csv(out/'entries.csv',index=False)
    summary={'status':'trained','fold':fold,'train_reactions':len(tr_ids),'dev_reactions':len(dv_ids),'reaction_overlap':0,'teacher_checkpoint':str(ckpt),'teacher_checkpoint_sha256':sha(ckpt),'training_pairs_sha256':sha(trp),'dev_pairs_sha256':sha(dvp),'rhea_source_sha256':sha(RHEA),'feature_manifest_sha256':sha(FEATURE/'manifest.json'),'target_external_labels_read':False,'dev_reaction_ids_used_for_training':False,'architecture':'LayerNorm4096-Linear512-GELU-Linear320-L2','epochs':80,'batch_size':256,'lr':1e-3,'weight_decay':1e-4,'zero_bags':int(sum(x['zero_feature'] for x in audit)),'invalid_molecules':int(sum(x['invalid_molecules'] for x in audit))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
