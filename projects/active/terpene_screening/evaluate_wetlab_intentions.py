from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[3]
INTENTIONS = ROOT / 'data/wetlab_intentions/diterpene_sesquiterpene_intentions.csv'
KNOWN = ROOT / 'data/terpene/enzyme_terpene_synthase.tsv'
ALLSEQ = ROOT / 'data/terpene/all_seq_terpene_synthase.tsv'
META = ROOT / 'results/terpene_cage_fair/meta_ranker/reaction_only_oof_meta_scores.csv'
OUT = ROOT / 'results/wetlab_intentions'
OUT.mkdir(parents=True, exist_ok=True)

FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
AA_RE = re.compile(r'^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$')


def clean_id(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    s = str(x).strip()
    if not s or s.lower() == 'nan':
        return ''
    if s.startswith('GenBank:'):
        return s.split(':', 1)[1].strip()
    return s


def is_uniprot_like(s: str) -> bool:
    return bool(re.match(r'^[A-Z0-9]{6,10}$', s or '')) and not s.startswith('WP_')


def mol_from_smiles(s: str):
    if not s or not isinstance(s, str):
        return None
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def heavy_carbon_count(mol) -> int:
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)


def largest_carbon_mol(side: str):
    parts = [p for p in str(side).split('.') if p]
    mols = []
    for p in parts:
        m = mol_from_smiles(p)
        if m is not None:
            mols.append((heavy_carbon_count(m), Chem.MolToSmiles(m, isomericSmiles=True), m))
    if not mols:
        return None, '', 0
    mols.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return mols[0][2], mols[0][1], mols[0][0]


def rxn_parts(rxn: str) -> Tuple[str, str]:
    if not isinstance(rxn, str) or '>>' not in rxn:
        return '', ''
    left, right = rxn.split('>>', 1)
    return left, right


def rxn_product_mol(rxn: str):
    _, right = rxn_parts(rxn)
    return largest_carbon_mol(right)


def rxn_substrate_class(rxn: str) -> str:
    left, _ = rxn_parts(rxn)
    mol, _, c = largest_carbon_mol(left)
    if c >= 19:
        return 'GGPP-like'
    if 14 <= c <= 16:
        return 'FPP-like'
    if 9 <= c <= 11:
        return 'GPP-like'
    return f'C{c}' if c else ''


def intent_substrate_class(x: str) -> str:
    s = str(x).upper()
    if 'GGPP' in s:
        return 'GGPP-like'
    if 'FPP' in s:
        return 'FPP-like'
    if 'GPP' in s:
        return 'GPP-like'
    return ''


def fp(mol):
    if mol is None:
        return None
    try:
        return FPGEN.GetFingerprint(mol)
    except Exception:
        return None


def tanimoto(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(a, b))


def kmer_set(seq: str, k: int = 3) -> set:
    if not isinstance(seq, str):
        return set()
    s = ''.join(str(seq).upper().split())
    if not s or s.lower() == 'nan' or not AA_RE.match(s):
        return set()
    return {s[i:i+k] for i in range(max(0, len(s)-k+1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def same_ec(a: object, b: object) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    aa = set(re.findall(r'\d+\.\d+\.\d+\.\d+', str(a)))
    bb = set(re.findall(r'\d+\.\d+\.\d+\.\d+', str(b)))
    return bool(aa & bb)


def build_zero_shot_panel(g: pd.DataFrame, B: int, k: int) -> set:
    main = g.sort_values(['reaction_similarity', 'uniprot_id'], ascending=[False, True]).head(max(0, B-k))
    used = set(main['uniprot_id'])
    rescue = g[~g['uniprot_id'].isin(used)].sort_values(['oof_rf_rank', 'uniprot_id'], ascending=[False, True]).head(k)
    return set(pd.concat([main, rescue], ignore_index=True)['uniprot_id'])


def grade_row(row: dict) -> Tuple[str, str, str]:
    # Recommendation grade is intended for prioritizing wet-lab validation, not truth labeling.
    product_ok = row['best_product_tanimoto'] >= 0.72 and row['substrate_match']
    exact_or_near = row['original_known_product_tanimoto'] >= 0.80 and row['original_known_substrate_match']
    has_enzyme = bool(row['enzyme_id_clean']) or row['sequence_present']
    oechem = row['product_source_flag'] == 'OEChem_product'
    no_specific = not bool(row['enzyme_id_clean']) and not row['sequence_present']

    if no_specific:
        return 'C', '仅可评价反应靶标，缺少具体酶序列或酶编号。', '补充拟验证酶序列后再进入首批筛选。'
    if exact_or_near and row['enzyme_in_positive_dataset']:
        if oechem:
            return 'A-', '目标酶与已知数据库记录高度一致，但产物 SMILES 来源为 PubMed/OEChem，需核验结构标注。', '可作为优先复验对象；建议同步复核产物结构。'
        return 'A', '目标酶与已知反应记录高度一致，酶层面与反应层面证据一致。', '适合作为首批验证或阳性对照。'
    if product_ok and row['sequence_present'] and row['seq_top_jaccard'] >= 0.55:
        return 'B+', '反应层面支持较强，目标序列与候选酶库存在较近邻。', '可纳入首批验证，但应低于 A 类条目。'
    if product_ok and has_enzyme:
        return 'B', '反应层面支持较强，但酶层面证据不足或未进入当前候选池。', '可作为候补验证对象。'
    if row['best_product_tanimoto'] >= 0.55 and has_enzyme:
        return 'C+', '存在中等程度产物相似性，但底物/产物或酶层面证据不充分。', '不建议作为首批核心实验，可作为探索性条目。'
    return 'C', '当前证据不足以支持优先验证。', '需补充文献、序列来源或重新定义反应靶标。'


def main():
    intentions = pd.read_csv(INTENTIONS)
    known = pd.read_csv(KNOWN, sep='\t')
    allseq = pd.read_csv(ALLSEQ, sep='\t').drop_duplicates('Entry')
    seq_dict = dict(zip(allseq['Entry'].astype(str), allseq['Sequence'].astype(str)))
    seq_kmers = {k: kmer_set(v) for k, v in seq_dict.items()}

    # Prepare known reaction table with product fingerprints.
    krows = []
    for idx, r in known.iterrows():
        mol, psmiles, pc = rxn_product_mol(r.get('smiles_seq', ''))
        f = fp(mol)
        krows.append({
            'known_idx': idx,
            'Entry': str(r.get('Entry', '')),
            'EC number': r.get('EC number', ''),
            'rhea_id': str(r.get('rhea_id', '')),
            'smiles_seq': r.get('smiles_seq', ''),
            'known_product_smiles': psmiles,
            'known_product_carbons': pc,
            'known_substrate_class': rxn_substrate_class(r.get('smiles_seq', '')),
            'fp': f,
        })
    kdf = pd.DataFrame(krows)

    # Load zero-shot learned rescue scores. Only needed columns.
    usecols = ['gate_id','reaction_id','uniprot_id','reaction_similarity','oof_rf_rank','oof_hgb_rank','cage_rank_score_fill0','label']
    meta = pd.read_csv(META, usecols=usecols)
    meta = meta[meta['gate_id'].eq('recall_union_core')].copy()

    out_rows = []
    top_rows = []
    seq_rows = []

    for _, ir in intentions.iterrows():
        number = int(ir['number'])
        enzyme_id = clean_id(ir.get('original_enzyme', ''))
        product = str(ir.get('product', ''))
        product_source_flag = 'OEChem_product' if 'OEChem_product' in str(ir.get('P.S.', '')) else ''
        intent_class = intent_substrate_class(ir.get('substrate', ''))
        imol, ipsmiles, ipc = rxn_product_mol(ir.get('smiles', ''))
        ifp = fp(imol)
        seq = '' if pd.isna(ir.get('enzyme_seq', '')) else str(ir.get('enzyme_seq', '')).strip()
        sequence_present = bool(seq and seq.lower() != 'nan')

        # Best known reaction/product matches.
        sims = []
        for _, kr in kdf.iterrows():
            sim = tanimoto(ifp, kr['fp'])
            smatch = intent_class and kr['known_substrate_class'] == intent_class
            ecsame = same_ec(ir.get('ec', ''), kr.get('EC number', ''))
            score = sim + (0.08 if smatch else 0) + (0.05 if ecsame else 0)
            sims.append((score, sim, bool(smatch), bool(ecsame), kr))
        sims.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = sims[0]
        best_same = next((x for x in sims if x[2]), best)

        for rank, x in enumerate(sims[:8], start=1):
            _, sim, smatch, ecsame, kr = x
            top_rows.append({
                'number': number, 'rank': rank, 'query_product': product,
                'matched_rhea_id': kr['rhea_id'], 'matched_entry': kr['Entry'],
                'product_tanimoto': sim, 'substrate_match': smatch, 'ec_match': ecsame,
                'known_substrate_class': kr['known_substrate_class'],
                'known_product_carbons': kr['known_product_carbons'],
            })

        # Original enzyme in positive dataset.
        enzyme_known = kdf[kdf['Entry'].eq(enzyme_id)] if enzyme_id else kdf.iloc[0:0]
        original_known_best_sim = 0.0
        original_known_rhea = ''
        original_known_submatch = False
        if not enzyme_known.empty:
            e_sims = []
            for _, kr in enzyme_known.iterrows():
                sim = tanimoto(ifp, kr['fp'])
                smatch = intent_class and kr['known_substrate_class'] == intent_class
                e_sims.append((sim + (0.08 if smatch else 0), sim, bool(smatch), kr))
            e_sims.sort(key=lambda x: (x[0], x[1]), reverse=True)
            original_known_best_sim = e_sims[0][1]
            original_known_submatch = e_sims[0][2]
            original_known_rhea = e_sims[0][3]['rhea_id']

        # Sequence nearest neighbors in candidate sequence database.
        seq_top_entry = ''
        seq_top_j = 0.0
        if sequence_present:
            ks = kmer_set(seq)
            if ks:
                vals = [(entry, jaccard(ks, kms)) for entry, kms in seq_kmers.items()]
                vals.sort(key=lambda x: x[1], reverse=True)
                seq_top_entry, seq_top_j = vals[0]
                for rank, (entry, score) in enumerate(vals[:8], start=1):
                    seq_rows.append({'number': number, 'rank': rank, 'query_enzyme': enzyme_id, 'neighbor_entry': entry, 'kmer_jaccard': score})

        # Zero-shot candidate status for nearest Rhea reaction in current workflow.
        best_rhea = best_same[4]['rhea_id'] if best_same[2] else best[4]['rhea_id']
        zg = meta[meta['reaction_id'].eq(best_rhea)]
        z_rank_rxn = np.nan
        z_rank_rf = np.nan
        z_in_top10 = False
        z_in_top20 = False
        z_cage_rank_score = np.nan
        if enzyme_id and not zg.empty and enzyme_id in set(zg['uniprot_id'].astype(str)):
            tmp = zg.copy()
            tmp['rxn_rank'] = tmp['reaction_similarity'].rank(method='first', ascending=False)
            tmp['rf_rank'] = tmp['oof_rf_rank'].rank(method='first', ascending=False)
            er = tmp[tmp['uniprot_id'].astype(str).eq(enzyme_id)].iloc[0]
            z_rank_rxn = int(er['rxn_rank'])
            z_rank_rf = int(er['rf_rank'])
            z_cage_rank_score = float(er['cage_rank_score_fill0'])
            z_in_top10 = enzyme_id in build_zero_shot_panel(tmp, B=10, k=5)
            z_in_top20 = enzyme_id in build_zero_shot_panel(tmp, B=20, k=10)

        row = {
            'number': number,
            'substrate': ir.get('substrate', ''),
            'product': product,
            'enzyme_id_raw': ir.get('original_enzyme', ''),
            'enzyme_id_clean': enzyme_id,
            'id_type': 'UniProt-like' if is_uniprot_like(enzyme_id) else ('GenBank/WP' if enzyme_id else 'EC-only/no-enzyme'),
            'ec': ir.get('ec', ''),
            'product_source_flag': product_source_flag,
            'intent_substrate_class': intent_class,
            'query_product_carbons': ipc,
            'sequence_present': sequence_present,
            'enzyme_in_sequence_db': enzyme_id in seq_dict,
            'enzyme_in_positive_dataset': not enzyme_known.empty,
            'original_known_product_tanimoto': original_known_best_sim,
            'original_known_rhea_id': original_known_rhea,
            'original_known_substrate_match': original_known_submatch,
            'best_rhea_id': best[4]['rhea_id'],
            'best_entry': best[4]['Entry'],
            'best_product_tanimoto': best[1],
            'best_known_substrate_class': best[4]['known_substrate_class'],
            'substrate_match': best[2],
            'ec_match_to_best': best[3],
            'best_same_substrate_rhea_id': best_same[4]['rhea_id'],
            'best_same_substrate_entry': best_same[4]['Entry'],
            'best_same_substrate_product_tanimoto': best_same[1],
            'seq_top_entry': seq_top_entry,
            'seq_top_jaccard': seq_top_j,
            'workflow_mapped_rhea_id': best_rhea,
            'in_recall_union_core_for_mapped_rhea': bool(enzyme_id and enzyme_id in set(zg['uniprot_id'].astype(str))),
            'rxn_similarity_rank_in_mapped_rhea': z_rank_rxn,
            'rf_rescue_rank_in_mapped_rhea': z_rank_rf,
            'cage_rank_score_in_mapped_rhea': z_cage_rank_score,
            'in_zero_shot_top10_panel': z_in_top10,
            'in_zero_shot_top20_panel': z_in_top20,
        }
        grade, rationale, action = grade_row(row)
        row['recommendation_grade'] = grade
        row['rationale_short'] = rationale
        row['recommended_action'] = action
        out_rows.append(row)

    result = pd.DataFrame(out_rows)
    top = pd.DataFrame(top_rows)
    seqn = pd.DataFrame(seq_rows)

    result.to_csv(OUT / 'wetlab_intention_assessment.csv', index=False)
    top.to_csv(OUT / 'wetlab_intention_top_reaction_matches.csv', index=False)
    seqn.to_csv(OUT / 'wetlab_intention_sequence_neighbors.csv', index=False)

    # Markdown report.
    order = {'A': 0, 'A-': 1, 'B+': 2, 'B': 3, 'C+': 4, 'C': 5, 'D': 6}
    res_sorted = result.sort_values(['recommendation_grade','number'], key=lambda s: s.map(order).fillna(9) if s.name=='recommendation_grade' else s)
    lines = []
    lines.append('# 二萜与倍半萜合成意向实验的计算判定报告')
    lines.append('')
    lines.append('## 1. 数据与前提')
    lines.append('本报告依据上传的 20 条底物—产物—酶对应记录进行计算判定。表中主产物、副产物及多酶记录已拆分为一一对应条目；有 UniProt 编号的酶采用 UniProt ID，缺少 UniProt ID 的记录采用 GenBank/WP 编号。部分记录的产物 SMILES 标注为 OEChem_product，表示该产物结构来自 PubMed/OEChem 风格标注而非 ChEBI 标注。')
    lines.append('')
    lines.append('## 2. 方法')
    lines.append('对每条记录，首先从 reaction SMILES 中提取主要碳骨架产物，使用 RDKit Morgan fingerprint 计算其与本地 Rhea/TPS 正例库产物的 Tanimoto 相似度。底物类别按照 FPP-like 与 GGPP-like 进行归类，并作为反应层面匹配的约束。对有酶序列的记录，使用 3-mer Jaccard 与候选 TPS 序列库进行近邻检索。对具有 UniProt-like 编号的记录，进一步检查其是否存在于已知正例库、候选序列库，以及在当前 zero-shot workflow 映射 Rhea 反应下是否进入 recall_union_core reservoir。')
    lines.append('')
    lines.append('zero-shot 工作流采用既有结果：recall_union_core 作为高召回候选池，reaction similarity 作为主排序信号，RandomForest/HGB meta-ranker 作为 learned rescue 信号。')
    lines.append('')
    lines.append('## 3. 汇总结果')
    summary = result['recommendation_grade'].value_counts().sort_index()
    for g, n in summary.items():
        lines.append(f'- {g}: {int(n)} 条')
    lines.append('')
    lines.append('## 4. 分项判定表')
    lines.append('')
    lines.append('| 编号 | 底物 | 产物 | 酶编号 | 等级 | 最相近 Rhea | 产物相似度 | 原酶已知支持 | 序列近邻 | 判定依据 | 建议 |')
    lines.append('|---:|---|---|---|---|---|---:|---|---|---|---|')
    for _, r in res_sorted.iterrows():
        enz = r['enzyme_id_clean'] or '未给出'
        known_support = '是' if r['enzyme_in_positive_dataset'] and r['original_known_product_tanimoto'] >= 0.8 else ('部分' if r['enzyme_in_positive_dataset'] else '否')
        seq_support = f"{r['seq_top_entry']} ({r['seq_top_jaccard']:.2f})" if r['sequence_present'] else '无序列'
        lines.append('| {number} | {substrate} | {product} | {enz} | {grade} | {rhea} | {sim:.3f} | {known} | {seq} | {rat} | {act} |'.format(
            number=int(r['number']), substrate=r['substrate'], product=str(r['product']).replace('|','/'), enz=enz,
            grade=r['recommendation_grade'], rhea=r['workflow_mapped_rhea_id'], sim=float(r['best_same_substrate_product_tanimoto'] if r['substrate_match'] else r['best_product_tanimoto']),
            known=known_support, seq=seq_support, rat=str(r['rationale_short']).replace('|','/'), act=str(r['recommended_action']).replace('|','/')
        ))
    lines.append('')
    lines.append('## 5. 结论')
    lines.append('A/A- 类条目主要对应已知酶—反应关系或高度一致的反应/酶证据，适合作为首批验证对象或阳性对照。B/B+ 类条目具备较强反应层面支持，但酶层面证据仍需结合序列来源、表达可行性和产物检测条件进一步核验。C 类条目主要受限于缺少具体酶序列、仅有 EC 信息、或产物结构标注来源不一致，不宜直接作为首批核心验证对象。')
    lines.append('')
    lines.append('## 6. 输出文件')
    lines.append('- `results/wetlab_intentions/wetlab_intention_assessment.csv`')
    lines.append('- `results/wetlab_intentions/wetlab_intention_top_reaction_matches.csv`')
    lines.append('- `results/wetlab_intentions/wetlab_intention_sequence_neighbors.csv`')

    (OUT / 'wetlab_intention_assessment_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(result[['number','substrate','product','enzyme_id_clean','recommendation_grade','workflow_mapped_rhea_id','best_same_substrate_product_tanimoto','original_known_product_tanimoto','seq_top_entry','seq_top_jaccard','rationale_short']].to_string(index=False))
    print('\noutputs:')
    for p in ['wetlab_intention_assessment.csv','wetlab_intention_top_reaction_matches.csv','wetlab_intention_sequence_neighbors.csv','wetlab_intention_assessment_report.md']:
        print(str(OUT / p))

if __name__ == '__main__':
    main()
