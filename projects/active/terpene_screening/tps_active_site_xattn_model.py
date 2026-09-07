from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class XAttnConfig:
    latent_dim:int
    attention_heads:int
    cross_attention_layers:int
    dropout:float
    learning_rate:float
    weight_decay:float
    margin:float
    hard_negatives_per_positive:int

class CrossBlock(nn.Module):
    def __init__(self,d:int,heads:int,dropout:float):
        super().__init__(); self.pn=nn.LayerNorm(d); self.rn=nn.LayerNorm(d)
        self.p_to_r=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.r_to_p=nn.MultiheadAttention(d,heads,dropout=dropout,batch_first=True)
        self.pff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
        self.rff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,4*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(4*d,d))
        self.drop=nn.Dropout(dropout)
    def forward(self,p,r,pm,rm):
        pn=self.pn(p); rn=self.rn(r)
        pd,_=self.p_to_r(pn,rn,rn,key_padding_mask=~rm,need_weights=False)
        rd,_=self.r_to_p(rn,pn,pn,key_padding_mask=~pm,need_weights=False)
        p=(p+self.drop(pd))*pm.unsqueeze(-1); r=(r+self.drop(rd))*rm.unsqueeze(-1)
        p=(p+self.drop(self.pff(p)))*pm.unsqueeze(-1); r=(r+self.drop(self.rff(r)))*rm.unsqueeze(-1)
        return p,r

def masked_mean(x,mask):
    den=mask.sum(1,keepdim=True).clamp_min(1).to(x.dtype); return (x*mask.unsqueeze(-1)).sum(1)/den

class TPSActiveSiteXAttn(nn.Module):
    def __init__(self,config:XAttnConfig,protein_dim:int=1152,reaction_dim:int=23):
        super().__init__(); d=config.latent_dim
        self.p_proj=nn.Linear(protein_dim,d); self.r_proj=nn.Linear(reaction_dim,d)
        self.type_emb=nn.Embedding(6,d); self.rel_emb=nn.Embedding(25,d)
        self.blocks=nn.ModuleList([CrossBlock(d,config.attention_heads,config.dropout) for _ in range(config.cross_attention_layers)])
        self.head=nn.Sequential(nn.LayerNorm(6*d),nn.Linear(6*d,2*d),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(2*d,d),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(d,1))
    def forward(self,protein,pmask,ptype,prel,reaction,rmask,rchanged):
        p=self.p_proj(protein)+self.type_emb(ptype)+self.rel_emb((prel+12).clamp(0,24)); r=self.r_proj(reaction)
        p=p*pmask.unsqueeze(-1); r=r*rmask.unsqueeze(-1)
        for block in self.blocks: p,r=block(p,r,pmask,rmask)
        pg=p[:,0]; pmean=masked_mean(p,pmask); rmean=masked_mean(r,rmask); cmask=rmask&rchanged
        rc=masked_mean(r,cmask); no=~cmask.any(1)
        if no.any(): rc=rc.clone(); rc[no]=rmean[no]
        z=torch.cat([pg,pmean,rmean,rc,pmean*rc,torch.abs(pmean-rc)],dim=-1)
        return self.head(z).squeeze(-1)
