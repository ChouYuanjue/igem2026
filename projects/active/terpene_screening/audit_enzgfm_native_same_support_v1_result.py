from __future__ import annotations
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'results/enzgfm_native_same_support_catalyst_v1'
PREP = RUN / 'prepared'
ARCHIVE = ROOT / 'data/external/reactzyme/enzyme_smi_split.zip'
TRAIN_RAW = ROOT / 'data/external/reactzyme/enzyme_smi_split/positive_train_val_seq_smi.pt'
TEST_RAW = ROOT / 'data/external/reactzyme/enzyme_smi_split/positive_test_seq_smi.pt'
EMB = ROOT / 'data/external/enzgfm_current/general_merged_650m_mean_v1/embeddings.npy'
OUT = RUN / 'posthoc_audit/summary.json'
EXPECTED_ARCHIVE_MD5 = 'e351fdb85830968fc9abe933c39f9eda'
PAPER = {
    'citation': 'Wang et al., Nature Communications 17, 8760 (2026)',
    'doi': '10.1038/s41467-026-75283-3',
    'table': 'Supplementary Table S6',
    'replicates': 5,
    'e2r': {'map': .5156, 'ndcg@1': .4233, 'ndcg@5': .5152, 'top1': .4233, 'top5': .6636},
    'r2e': {'map': .8211, 'ndcg@1': .7210, 'ndcg@5': .8484, 'top1': .7210, 'top5': .9425},
    'sd': {
        'e2r': {'map': .0112, 'ndcg@1': .0085, 'ndcg@5': .0126, 'top1': .0085, 'top5': .0154},
        'r2e': {'map': .0094, 'ndcg@1': .0133, 'ndcg@5': .0088, 'top1': .0133, 'top5': .0072},
    },
}

def digest(path: Path, algo='sha256') -> str:
    h = hashlib.new(algo)
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def build_labels(pairs, proteins, reactions, shape):
    pm = {int(r.protein_row): i for i, r in proteins.reset_index(drop=True).iterrows()}
    rm = {int(r.reaction_idx): i for i, r in reactions.reset_index(drop=True).iterrows()}
    y = np.zeros(shape, dtype=bool)
    for r in pairs.itertuples(index=False):
        y[pm[int(r.protein_row)], rm[int(r.reaction_idx)]] = True
    return y

def metrics(scores, labels):
    rows=[]
    for sc, lab in zip(scores, labels):
        if not lab.any():
            continue
        order=np.argsort(-sc, kind='stable'); rel=lab[order].astype(float); n=int(lab.sum()); ranks=np.flatnonzero(rel)+1
        ap=float(((np.cumsum(rel)/(np.arange(len(rel))+1))*rel).sum()/n)
        d={'map':ap,'top1':float((ranks<=1).any()),'top5':float((ranks<=5).any())}
        for k in (1,5):
            dcg=sum(1/np.log2(r+1) for r in ranks if r<=k)
            idcg=sum(1/np.log2(i+2) for i in range(min(n,k)))
            d[f'ndcg@{k}']=float(dcg/idcg)
        rows.append(d)
    return {'n_queries':len(rows)} | {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}

def raw_sequence_overlap():
    a=torch.load(TRAIN_RAW,map_location='cpu',weights_only=False); b=torch.load(TEST_RAW,map_location='cpu',weights_only=False)
    ta=defaultdict(list); tb=defaultdict(list)
    for k,v in a.items(): ta[str(v[1])].append(str(k))
    for k,v in b.items(): tb[str(v[1])].append(str(k))
    ov=sorted(set(ta)&set(tb))
    return {'unique_exact_sequences':len(ov),'train_source_keys':[ta[s] for s in ov],'test_source_keys':[tb[s] for s in ov]}

def main():
    assert digest(ARCHIVE,'md5') == EXPECTED_ARCHIVE_MD5
    selection=json.loads((RUN/'selection/summary.json').read_text())
    prep=json.loads((PREP/'summary.json').read_text())
    scores=np.load(RUN/'final/dual_tower/native_test_scores.npy')
    tr=pd.read_csv(PREP/'all_train_pairs.csv'); te=pd.read_csv(PREP/'test_pairs.csv'); proteins=pd.read_csv(PREP/'test_proteins.csv'); reactions=pd.read_csv(PREP/'test_reactions.csv')
    labels=build_labels(te,proteins,reactions,scores.shape)
    paper_aligned={'e2r':metrics(scores,labels),'r2e':metrics(scores.T,labels.T)}
    # Official archive has three exact train/test sequence overlaps. Quantify a stricter posthoc view without them.
    train_proteins=set(tr.protein_row.astype(int)); keep=~proteins.protein_row.astype(int).isin(train_proteins).to_numpy()
    strict_scores=scores[keep]; strict_labels=labels[keep]
    strict={'removed_exact_overlap_test_proteins':int((~keep).sum()),'e2r':metrics(strict_scores,strict_labels),'r2e':metrics(strict_scores.T,strict_labels.T)}
    # All native test reactions are train-seen. A zero-tuned train-label prototype diagnostic measures split easiness.
    emb=np.load(EMB,mmap_mode='r'); rids=reactions.reaction_idx.astype(int).to_numpy(); rmap={r:i for i,r in enumerate(rids)}
    proto=np.zeros((len(rids),emb.shape[1]),np.float32); counts=np.zeros(len(rids),np.int64)
    rel=tr[tr.reaction_idx.astype(int).isin(set(rids))]
    for rid,g in rel.groupby('reaction_idx'):
        rows=np.unique(g.protein_row.astype(int).to_numpy()); v=np.asarray(emb[rows],dtype=np.float32)
        proto[rmap[int(rid)]]=v.mean(0); counts[rmap[int(rid)]]=len(rows)
    assert (counts>0).all()
    proto/=np.maximum(np.linalg.norm(proto,axis=1,keepdims=True),1e-12)
    q=np.asarray(emb[proteins.protein_row.astype(int).to_numpy()],dtype=np.float32); q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12)
    proto_scores=q@proto.T
    prototype={'status':'post_reveal_descriptive_only_no_model_selection','score_construction_reads_test_labels':False,'e2r':metrics(proto_scores,labels),'r2e':metrics(proto_scores.T,labels.T),'train_support_per_test_reaction':{'min':int(counts.min()),'median':float(np.median(counts)),'mean':float(counts.mean())}}
    deltas={d:{k:paper_aligned[d][k]-PAPER[d][k] for k in PAPER[d]} for d in ('e2r','r2e')}
    test_reactions=set(te.reaction_idx.astype(int)); train_reactions=set(tr.reaction_idx.astype(int))
    audit={
      'name':'enzgfm_native_same_support_catalyst_v1_posthoc_audit',
      'status':'revealed_external_frozen_posthoc_descriptive_only',
      'baseline_policy':'EnzGFM-1.5B remains the unique authoritative external baseline for this native bidirectional contract; the 650M prototype is diagnostic only and never substitutes for it.',
      'no_post_reveal_model_selection':True,
      'selected_candidate':selection['selected_candidate'],
      'alternative_candidate_native_test_scored':(RUN/'final/author_pairwise/test_evaluation.json').exists(),
      'official_split_provenance':{'reactzyme_zenodo_record':'11494913','archive_md5':digest(ARCHIVE,'md5'),'matches_official_zenodo_md5':True,'train_rows_source':prep['train_source_rows'],'test_rows':prep['native_test_rows'],'all_reaction_bags':prep['unique_reaction_bags_all_inputs']},
      'split_structure':{'test_unique_proteins':int(proteins.protein_row.nunique()),'test_unique_reactions':int(reactions.reaction_idx.nunique()),'test_reaction_seen_in_train_count':len(test_reactions & train_reactions),'test_reaction_seen_in_train_fraction':len(test_reactions & train_reactions)/len(test_reactions),'raw_exact_train_test_sequence_overlap':raw_sequence_overlap(),'claim_boundary':'Official enzyme-similarity split primarily tests generalization to sequence-divergent enzymes for train-seen reactions; it is not a reaction-novel benchmark. The public archive itself contains three exact train/test protein sequences despite the paper-level >=60% difference description.'},
      'paper_metric_alignment':{'map_definition_matches_standard_average_precision':True,'common_metrics':['map','ndcg@1','ndcg@5','top1','top5'],'paper':PAPER,'single_catalyst_run_vs_paper_five_run_mean':True,'paired_statistical_superiority_claim_allowed':False},
      'catalyst_frozen_paper_aligned_metrics':paper_aligned,
      'absolute_deltas_vs_paper_mean':deltas,
      'strict_remove_three_exact_overlap_posthoc':strict,
      'train_only_650m_prototype_diagnostic':prototype,
      'interpretation':{'large_e2r_gain_is_not_explained_by_metric_mismatch_wrong_archive_or_three_exact_sequence_overlaps':True,'simple_train_only_650m_prototype_is_already_far_above_paper_e2r_baseline':True,'native_split_should_not_support_reaction_novel_discovery_claim':True,'future_tuning_on_this_native_test_forbidden':True},
      'frozen_artifact_sha256':{'selection':digest(RUN/'selection/summary.json'),'test_evaluation':digest(RUN/'final/dual_tower/test_evaluation.json'),'native_test_scores':digest(RUN/'final/dual_tower/native_test_scores.npy'),'final_model':digest(RUN/'final/dual_tower/model.pt')},
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(audit,indent=2)+'\n'); print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
