from __future__ import annotations
import argparse,json,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch_geometric.loader import DataLoader
from projects.active.bridge.model.assets import ROOT
from reproducibility.bime_rank.scripts.extract_enzymecage_family_response_v1 import resolve_config,build_dataset,make_model,GENERIC_CKPT

BASE=ROOT/'results/bridge_pocket_interaction_train_v2';OUT=BASE

class Capture:
 def __init__(self,model):self.blocks=[];self.h=model.interaction_model.register_forward_hook(self.hook)
 def hook(self,module,args,output):
  fused=output[0] if isinstance(output,tuple) else output;self.blocks.append(fused.detach().float().cpu().numpy().astype(np.float32))
 def close(self):self.h.remove()
 def array(self):return np.concatenate(self.blocks,axis=0)

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--batch-size',type=int,default=256);a=ap.parse_args();device=torch.device(a.device);conf=resolve_config(BASE/'infer.yaml');frame,ds,audit=build_dataset(conf);model=make_model(conf,GENERIC_CKPT,device);cap=Capture(model);loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,follow_batch=['protein','reaction_feature','esm_feature','substrates','products'],num_workers=0);logits=[]
 for bi,b in enumerate(loader):
  logits.append(model(b.to(device)).detach().float().cpu().numpy())
  if (bi+1)%100==0:print('train-interaction',bi+1,'/',len(loader),flush=True)
 fused=cap.array();cap.close();logits=np.concatenate(logits);np.save(OUT/'interaction_fused_f32.npy',fused)
 cols=[c for c in ['reaction_id','protein_id','Label','broad_score','broad_rank_top1000','heldout_folds'] if c in frame.columns];meta=frame[cols].copy();meta['generic_logit']=logits;meta.to_csv(OUT/'pairs_scored.csv.gz',index=False)
 summary={'schema':'bridge-pocket-interaction-train-v2-repr','rows':len(frame),'queries':int(frame.reaction_id.nunique()),'dim':int(fused.shape[1]),'checkpoint':str(GENERIC_CKPT.relative_to(ROOT)),'audit':audit};(OUT/'repr_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
