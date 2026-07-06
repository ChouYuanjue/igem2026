# Terpene CAGE 全酶库检索实验报告

## 实验目的
给定 10 条萜类 Rhea 反应，把所有萜类合酶作为候选酶，不做候选池筛选，直接构造 10 reactions × all terpene synthases 的 pair，使用 EnzymeCAGE 对所有 pair 打分并做排序，随后计算 Top-1 / Top-5 / Top-10 recall、MRR，以及每条 reaction 的真实酶排名与 Top-10 推荐列表。

## 数据来源和数据规模
- `positive_labels`: `1640` 行，列名 `Entry, EC number, rhea_id, smiles_seq, Sequence`
- `candidate_enzymes`: `1391` 行，列名 `Entry, Sequence`
- `selected_reactions`: `10` 行，列名 `RHEA_ID, SMILES`
- 候选酶总数: `1391`
- 10 条 reaction 中有 positive label 的数量: `10`
- 10 条 reaction 中没有 positive label 的数量: `0`
- 预计 pair 数: `13910`

## 为什么不做候选池筛选
这次实验目标是做全库检索，而不是候选池内重排序。直接对所有 terpene synthase 构造 pair 可以避免先验筛选把真正的正例提前排除，也更接近“检索”场景本身，能更真实地衡量 EnzymeCAGE 在大候选空间里的区分能力。

## 为什么只用 P2Rank top1
本实验要求单 pocket 设置。使用 P2Rank top1 可以把 pocket 选择固定下来，避免多 pocket 聚合带来的额外设计自由度，同时也更省计算成本，更适合先做一版全库 screening 基线。

## 结构下载成功率
- 成功结构数: `1381` / `1391`
- 成功率: `1381/1391 (99.28%)`

## P2Rank 成功率
- 成功 pocket 数: `1379` / `1381`
- 成功率: `1379/1381 (99.86%)`
- P2Rank 失败数: `10`

## CAGE 成功打分 pair 数
- 成功打分 pair 数: `13790` / `13910`
- 覆盖率: `13790/13910 (99.14%)`

## Top-1 / Top-5 / Top-10 recall
- Top-1 recall: `0.0000`
- Top-5 recall: `0.0000`
- Top-10 recall: `0.0000`
- MRR: `0.0037`
- median best positive rank: `387.0`

## 每条 reaction 的命中情况
| reaction_id | rhea_id | status | n_candidates | n_positive_enzymes | best_positive_rank | top1_hit | top5_hit | top10_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reaction_01 | RHEA:22408 | ok | 1391 | 24 | 387.0 | False | False | False |
| reaction_02 | RHEA:17653 | ok | 1391 | 45 | 129.0 | False | False | False |
| reaction_03 | RHEA:58164 | ok | 1391 | 2 | 178.0 | False | False | False |
| reaction_04 | RHEA:32551 | ok | 1391 | 12 | 724.0 | False | False | False |
| reaction_05 | RHEA:12869 | ok | 1391 | 16 | 481.0 | False | False | False |
| reaction_06 | RHEA:32691 | ok | 1391 | 22 | 223.0 | False | False | False |
| reaction_07 | RHEA:54060 | ok | 1391 | 1 | 689.0 | False | False | False |
| reaction_08 | RHEA:60020 | ok | 1391 | 2 | 1095.0 | False | False | False |
| reaction_09 | RHEA:31811 | ok | 1391 | 6 | 94.0 | False | False | False |
| reaction_10 | RHEA:31023 | ok | 1391 | 1 |  | False | False | False |

## Top-10 推荐酶列表
- `reaction_01` / `RHEA:22408`: best positive rank `387`; top10 enzymes `P9WER1, Q0C8A0, L0MZK0, Q5AR34, A0A0U5GHG9, A0A0F7TZD6, A0A1E1FFL0, A0A097ZPD9, A0A1Y0BRF4, A0A3G9H185`; scores `0.1265, 0.0402, 0.0376, 0.0298, 0.0250, 0.0238, 0.0223, 0.0206, 0.0203, 0.0192`
- `reaction_02` / `RHEA:17653`: best positive rank `129`; top10 enzymes `P9WER1, Q0C8A0, L0MZK0, Q5AR34, A0A0U5GHG9, A0A0F7TZD6, A0A1Y0BRF4, A0A3G9H185, A0A1E1FFL0, A0A1L9WUX8`; scores `0.1842, 0.0971, 0.0758, 0.0597, 0.0493, 0.0427, 0.0423, 0.0413, 0.0388, 0.0373`
- `reaction_03` / `RHEA:58164`: best positive rank `178`; top10 enzymes `A0A060KY90, A0A067DDU9, A0A067ECN5, A0A067FI21, A0A067SEC9, A0A067Z9B6, A0A075F9Z3, A0A075FA51, A0A075FAK4, A0A075FBG7`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_04` / `RHEA:32551`: best positive rank `724`; top10 enzymes `A0A097ZPD9, E0D7H6, A0A1E1FFL0, A0A0F7TZD6, A0A2I1BSZ6, A0A0U5GHG9, Q5AR34, D5SJ87, A0A3G9GR23, A0A2P2GK84`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_05` / `RHEA:12869`: best positive rank `481`; top10 enzymes `A0A097ZPD9, A0A1E1FFL0, A0A0F7TZD6, A0A2I1BSZ6, A0A140JWS6, A0A0U5GHG9, Q5AR34, E0D7H6, A0A097ZPD5, A0A2P2GK84`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_06` / `RHEA:32691`: best positive rank `223`; top10 enzymes `A0A060KY90, A0A067DDU9, A0A067ECN5, A0A067FI21, A0A067SEC9, A0A067Z9B6, A0A075F9Z3, A0A075FA51, A0A075FAK4, A0A075FBG7`; scores `1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000`
- `reaction_07` / `RHEA:54060`: best positive rank `689`; top10 enzymes `A0A060KY90, A0A067DDU9, A0A067ECN5, A0A067FI21, A0A067SEC9, A0A067Z9B6, A0A075F9Z3, A0A075FA51, A0A075FAK4, A0A075FBG7`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_08` / `RHEA:60020`: best positive rank `1095`; top10 enzymes `A0A097ZPD9, A0A2I1BSZ6, A0A1E1FFL0, Q0C8A0, A0A0F7TZD6, A0A2P2GK84, A0A0U5GHG9, D5SJ87, A0A140JWS6, Q5AR34`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_09` / `RHEA:31811`: best positive rank `94`; top10 enzymes `A0A140JWS6, A0A097ZPD9, Q0C8A0, A0A1E1FFL1, A0A2I1BSZ6, A0A097ZPD5, H1VN83, A0A1E1FFL0, E0D7H6, A0A0F7TZD6`; scores `0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000`
- `reaction_10` / `RHEA:31023`: best positive rank `NA`; top10 enzymes `B8NHE1, P9WER0, A0A8H7XQ49, A0A8H8CFI5, A0A5S9I252, A0A084R1K7, Q2U0K2, S0ECT9, A0A8H7XP09, A0A8H7XMX0`; scores `0.0784, 0.0401, 0.0147, 0.0135, 0.0106, 0.0078, 0.0068, 0.0067, 0.0062, 0.0062`

## 失败记录
- ID 解析失败记录数: `0`
- P2Rank 失败记录数: `10`

### failed_p2rank_pockets.csv 前几行
```csv
UniprotID,structure_path,error
A0A097ZPE6,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/A0A097ZPE6.cif_residues.csv
P0CJ42,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/P0CJ42.cif_residues.csv
A0A097ZPE6,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/A0A097ZPE6.cif_residues.csv
P0CJ42,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/P0CJ42.cif_residues.csv
A0A097ZPE6,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/A0A097ZPE6.cif_residues.csv
P0CJ42,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/P0CJ42.cif_residues.csv
A0A097ZPE6,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/A0A097ZPE6.cif_residues.csv
P0CJ42,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/P0CJ42.cif_residues.csv
A0A097ZPE6,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/A0A097ZPE6.cif_residues.csv
P0CJ42,,No residues assigned to pocket rank 1 in /home/runnel/igem-pocket/data/terpene_cage_screen/_p2rank_stage/raw/P0CJ42.cif_residues.csv
```

## 结果解释
这次实验把搜索空间完全展开到全体 terpene synthase，因此指标更能反映模型在大候选空间中的排序能力。若 Top-k 指标较高，说明模型不仅能分辨反应类型，还能把真实酶稳定推到候选前列；若 MRR 偏低，则说明正确酶虽然有机会进入前列，但整体排序仍不够稳定。

## 局限性
1. 依赖 AlphaFold 结构和 P2Rank top1 pocket，结构误差或 pocket 选择误差都会影响最终分数。
2. 只看单 pocket，无法利用多 pocket 的互补信息。
3. 全库检索会放大数据覆盖问题，如果某些酶没有结构或 pocket，相关 pair 会被排除出打分集合。
4. 真实标签来自已知 Rhea 注释，未标注正例并不等价于真正的负例。
