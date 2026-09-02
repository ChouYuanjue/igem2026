from __future__ import annotations
import argparse,gc,hashlib,json,math,random,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from projects.active.terpene_screening.broad_rhea_metrics import evaluate_full_candidate_ranks
from projects.active.terpene_screening.tps_active_site_xattn_model import TPSActiveSiteXAttn,XAttnConfig
PROTOCOL=ROOT/'projects/active/terpene_screening/CATALYST_TPS_ACTIVE_SITE_XATTN_V1.json'
TOK=ROOT/'data/terpene_tps_active_site_xattn_v1'; SPLIT=ROOT/'results/tps_active_site_xattn_v1/fresh_split'; OUT=ROOT/'results/tps_active_site_xattn_v1'
SEED=20260902

def stable_offset(pid,q): return int.from_bytes(hashlib.blake2b(f'{pid}::{q}'.encode(),digest_size=8).digest(),'big')

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def compact(values,mask,*extras):
    n,L=mask.shape; counts=mask.sum(1).astype(int); out=np.zeros_like(values); om=np.zeros_like(mask); oe=[np.zeros_like(x) for x in extras]
    for i in range(n):
        idx=np.flatnonzero(mask[i]); k=len(idx); out[i,:k]=values[i,idx]; om[i,:k]=True
        for a,b in zip(oe,extras): a[i,:k]=b[i,idx]
    return (out,om,counts,*oe)

class Assets:
    def __init__(self):
        pe=pd.read_csv(TOK/'protein_entries.csv',dtype=str).sort_values('row'); re=pd.read_csv(TOK/'reaction_entries.csv',dtype=str).sort_values('row'); self.pids=pe.protein_id.astype(str).tolist(); self.rids=re.reaction_id.astype(str).tolist(); self.pidx={x:i for i,x in enumerate(self.pids)}; self.ridx={x:i for i,x in enumerate(self.rids)}
        p=np.load(TOK/'protein_tokens.npy').astype(np.float16); pm=np.load(TOK/'protein_token_mask.npy'); pt=np.load(TOK/'protein_token_type.npy'); pr=np.load(TOK/'protein_token_relpos.npy'); self.p,self.pm,self.pc,self.pt,self.pr=compact(p,pm,pt,pr)
        r=np.load(TOK/'reaction_atom_features.npy').astype(np.float16); rm=np.load(TOK/'reaction_atom_mask.npy'); rc=np.load(TOK/'reaction_atom_changed.npy'); self.r,self.rm,self.rcount,self.rc=compact(r,rm,rc)
    def batch(self,prows,rrows,device):
        prows=np.asarray(prows,int); rrows=np.asarray(rrows,int); pl=int(self.pc[prows].max()); rl=int(self.rcount[rrows].max())
        def t(x,dtype=None): return torch.as_tensor(x,device=device,dtype=dtype)
        return (t(self.p[prows,:pl],torch.float32),t(self.pm[prows,:pl],torch.bool),t(self.pt[prows,:pl],torch.long),t(self.pr[prows,:pl],torch.long),t(self.r[rrows,:rl],torch.float32),t(self.rm[rrows,:rl],torch.bool),t(self.rc[rrows,:rl],torch.bool))

def load_config_from_trial(trial)->XAttnConfig:
    return XAttnConfig(latent_dim=trial.suggest_categorical('latent_dim',[96,128,160]),attention_heads=trial.suggest_categorical('attention_heads',[4,8]),cross_attention_layers=trial.suggest_categorical('cross_attention_layers',[1,2]),dropout=trial.suggest_categorical('dropout',[0.0,0.1,0.2]),learning_rate=trial.suggest_float('learning_rate',1e-4,8e-4,log=True),weight_decay=trial.suggest_float('weight_decay',1e-6,1e-3,log=True),margin=trial.suggest_categorical('margin',[0.05,0.1,0.2]),hard_negatives_per_positive=trial.suggest_categorical('hard_negatives_per_positive',[16,24,32]))

def hard_pools(fold,k):
    x=pd.read_csv(SPLIT/f'fold{fold}_hard_pool_max32.csv',dtype=str).fillna(''); x['pool_rank']=pd.to_numeric(x.pool_rank).astype(int); x=x[x.pool_rank<=k]
    return {q:g.sort_values('pool_rank').candidate_id.astype(str).tolist() for q,g in x.groupby('query_id')}

def train_one(fold,config,assets,device):
    seed=SEED+fold; seed_all(seed); model=TPSActiveSiteXAttn(config).to(device); opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    pairs=pd.read_csv(SPLIT/f'fold{fold}_train_pairs.csv',dtype=str).fillna('')[['Entry','rhea_id']].drop_duplicates().sort_values(['rhea_id','Entry']).reset_index(drop=True); pools=hard_pools(fold,config.hard_negatives_per_positive); hist=[]
    base_idx=np.arange(len(pairs))
    for epoch in range(24):
        rng=np.random.default_rng(seed+epoch); order=rng.permutation(base_idx); losses=[]; acc=[]
        neg=[]
        for row in pairs.itertuples(index=False):
            pool=pools[str(row.rhea_id)]; neg.append(pool[(epoch+stable_offset(str(row.Entry),str(row.rhea_id)))%len(pool)])
        for start in range(0,len(order),64):
            ii=order[start:start+64]; sub=pairs.iloc[ii]; pp=[assets.pidx[x] for x in sub.Entry]; npid=[assets.pidx[neg[i]] for i in ii]; rr=[assets.ridx[x] for x in sub.rhea_id]
            opt.zero_grad(set_to_none=True)
            bp=assets.batch(pp,rr,device); bn=assets.batch(npid,rr,device)
            ac=torch.autocast(device_type=device.type,dtype=torch.bfloat16) if device.type=='cuda' else torch.autocast(device_type='cpu',enabled=False)
            with ac: ps=model(*bp); ns=model(*bn); loss=F.softplus(ns-ps+config.margin).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); losses.append(float(loss.detach().cpu())); acc.append(float((ps>ns).float().mean().detach().cpu()))
        hist.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'pairwise_accuracy':float(np.mean(acc))})
    return model,hist

def score_pairs(model,assets,q,candidates,device,batch=128):
    rr=assets.ridx[q]; rows=[]
    model.eval()
    with torch.no_grad():
        for s in range(0,len(candidates),batch):
            ids=candidates[s:s+batch]; prows=[assets.pidx[x] for x in ids]; rrows=[rr]*len(ids); b=assets.batch(prows,rrows,device)
            ac=torch.autocast(device_type=device.type,dtype=torch.bfloat16) if device.type=='cuda' else torch.autocast(device_type='cpu',enabled=False)
            with ac: z=model(*b)
            rows.extend(z.float().cpu().numpy().tolist())
    return np.asarray(rows,np.float32)

def order_and_metrics(fold,model,assets,device):
    train=pd.read_csv(SPLIT/f'fold{fold}_train_pairs.csv',dtype=str).fillna(''); dev=pd.read_csv(SPLIT/f'fold{fold}_dev_pairs.csv',dtype=str).fillna(''); sl=pd.read_csv(SPLIT/f'fold{fold}_shortlist.csv',dtype=str).fillna(''); sl['shortlist_rank']=pd.to_numeric(sl.shortlist_rank).astype(int); pack=np.load(SPLIT/f'fold{fold}_transfer_scores.npz'); qids=pack['query_ids'].astype(str).tolist(); pids=pack['protein_ids'].astype(str).tolist(); base=pack['baseline']; qrow={q:i for i,q in enumerate(qids)}
    rows=[]
    for q,g in dev.groupby('rhea_id',sort=True):
        q=str(q); positives=set(g.Entry.astype(str)); known=set(train.loc[train.rhea_id.eq(q),'Entry'].astype(str)); score=base[qrow[q]].astype(np.float64); cand=sl[sl.query_id.eq(q)].sort_values('shortlist_rank').candidate_id.astype(str).tolist(); cset=set(cand)
        transfer_idx={pid:i for i,pid in enumerate(pids)}
        # Transfer-score rows have their own explicit protein-id order. Token rows are
        # intentionally independent and must never index the transfer vector.
        bs=np.asarray([score[transfer_idx[x]] for x in cand],float)
        if not np.isfinite(bs).all():
            raise RuntimeError(f'non-finite aligned shortlist baseline for fold={fold} query={q}')
        # same-support baseline prefix
        bprefix=sorted(cand,key=lambda x:(-float(score[transfer_idx[x]]),x)); tail=sorted([x for x in pids if x not in cset and x not in known],key=lambda x:(-float(score[transfer_idx[x]]),x)); border=bprefix+tail
        raw=score_pairs(model,assets,q,cand,device); mu=float(bs.mean()); sd=max(float(bs.std()),1e-6); final=(bs-mu)/sd+2.0*np.tanh(raw)
        if not np.isfinite(final).all():
            raise RuntimeError(f'non-finite aligned XAttn score for fold={fold} query={q}')
        xprefix=[cand[i] for i in np.lexsort((np.asarray(cand,dtype=object),-final))]; xorder=xprefix+tail
        for method,order in [('baseline',border),('xattn',xorder)]:
            posmap={x:i+1 for i,x in enumerate(order)}; ranks=np.asarray(sorted(posmap[x] for x in positives if x in posmap),np.int64); m=evaluate_full_candidate_ranks(ranks,len(order),budgets=(3,10,20,50)); rows.append({'fold':fold,'query_id':q,'method':method,'shortlist_has_positive':int(bool(cset&positives)),**m})
    return pd.DataFrame(rows)

def aggregate(frame,method):
    x=frame[frame.method.eq(method)]; return {'queries':int(len(x)),'mrr':float(x.reciprocal_rank.mean()),'map':float(x.average_precision.mean()),'ndcg_at_10':float(x.ndcg_at_10.mean()),'hit_at_10':float(x.hit_at_10.mean()),'hit_at_20':float(x.hit_at_20.mean()),'hit_at_50':float(x.hit_at_50.mean()),'shortlist_oracle_hit':float(x.shortlist_has_positive.mean()),'median_best_positive_rank':float(x.best_positive_rank.median())}

def run_config(config,folds,assets,device,out_dir,save_models=False):
    frames=[]; training=[]
    for f in folds:
        model,hist=train_one(f,config,assets,device); q=order_and_metrics(f,model,assets,device); frames.append(q); training.append({'fold':f,'final_loss':hist[-1]['loss'],'final_pairwise_accuracy':hist[-1]['pairwise_accuracy']})
        if save_models:
            (out_dir/'models').mkdir(parents=True,exist_ok=True); torch.save({'config':asdict(config),'state_dict':model.state_dict(),'fold':f,'seed':SEED+f},out_dir/'models'/f'fold{f}.pt')
        del model; gc.collect();
        if device.type=='cuda': torch.cuda.empty_cache()
    frame=pd.concat(frames,ignore_index=True); return frame,{'baseline':aggregate(frame,'baseline'),'xattn':aggregate(frame,'xattn'),'training':training}

def parameter_count(config): return sum(p.numel() for p in TPSActiveSiteXAttn(config).parameters())

def hpo(device):
    import optuna
    out=OUT/'hpo'; out.mkdir(parents=True,exist_ok=True)
    if (out/'hpo_selection.json').exists(): raise RuntimeError('HPO selection already exists; refusing to rerun/reselect')
    assets=Assets(); sampler=optuna.samplers.TPESampler(seed=SEED); study=optuna.create_study(direction='maximize',sampler=sampler); results=[]
    def objective(trial):
        config=load_config_from_trial(trial); frame,summary=run_config(config,[0,1,2],assets,device,out); rec={'trial':trial.number,'config':asdict(config),'parameter_count':parameter_count(config),'summary':summary}; results.append(rec); (out/f'trial_{trial.number:02d}.json').write_text(json.dumps(rec,indent=2)+'\n'); frame.to_csv(out/f'trial_{trial.number:02d}_query_metrics.csv',index=False); return summary['xattn']['hit_at_10']
    study.optimize(objective,n_trials=18,show_progress_bar=False)
    ranked=sorted(results,key=lambda r:(-r['summary']['xattn']['hit_at_10'],-r['summary']['xattn']['mrr'],-r['summary']['xattn']['ndcg_at_10'],r['parameter_count'],r['trial']))
    best=ranked[0]; selection={'status':'hpo_selected_confirmation_unread','trials_completed':len(results),'selection_folds':[0,1,2],'confirmation_folds_unread':[3,4],'selected_trial':best['trial'],'selected_config':best['config'],'parameter_count':best['parameter_count'],'development_summary':best['summary'],'selection_rule':'Hit@10, then MRR, NDCG@10, lower params, lower trial','confirmation_metrics_read':False}
    (out/'all_trials.json').write_text(json.dumps(results,indent=2)+'\n'); (out/'hpo_selection.json').write_text(json.dumps(selection,indent=2)+'\n'); print(json.dumps(selection,indent=2)); return selection

def confirm(device):
    hpo_dir=OUT/'hpo'; conf=OUT/'confirmation'; sel_path=hpo_dir/'hpo_selection.json'
    if not sel_path.exists(): raise RuntimeError('selected HPO config missing; confirmation access forbidden')
    if (conf/'confirmation_result.json').exists(): raise RuntimeError('confirmation already revealed; refusing second run')
    sel=json.loads(sel_path.read_text()); config=XAttnConfig(**sel['selected_config']); conf.mkdir(parents=True,exist_ok=True); assets=Assets(); frame,summary=run_config(config,[3,4],assets,device,conf,save_models=True); frame.to_csv(conf/'query_metrics.csv',index=False); b=summary['baseline']; x=summary['xattn']; gain=100*(x['hit_at_10']-b['hit_at_10']); passed=gain>=3.0-1e-12 and x['mrr']>=b['mrr']-1e-12 and x['ndcg_at_10']>=b['ndcg_at_10']-1e-12
    result={'status':'passed_internal_confirmation' if passed else 'failed_internal_confirmation','pass':passed,'folds':[3,4],'selected_trial':sel['selected_trial'],'selected_config':sel['selected_config'],'baseline':b,'xattn':x,'hit10_gain_pp':gain,'gate':{'required_hit10_gain_pp':3.0,'mrr_nonregression':True,'ndcg10_nonregression':True},'no_post_confirmation_retuning':True}; (conf/'confirmation_result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); return result

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['hpo','confirm']); ap.add_argument('--device',default='cuda'); a=ap.parse_args(); device=torch.device(a.device)
 if a.action=='hpo': hpo(device)
 else: confirm(device)
if __name__=='__main__': main()
