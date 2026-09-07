# BiME-Rank 资产审计（2026-09-07）

唯一机器入口：`reproducibility/bime_rank/canonical.json`。所有路径相对 `/home/s241850073/igem2026`。
本次基准 Git 为 `420eb90b9fd46381c48e72a84f5928697acf3d0e`，分支 `model/zero-shot-known-recovery`；开始时工作区干净。
核对了服务器文件、manifest、实际代码引用、SHA、文件时间和在线 `/api/status`。没有执行训练、推理评测、删除数据或改变排序算法。

## 实际结构与入口

| 实际目录 | 整理后职责 |
|---|---|
| `configs/production_routes/terpene_v1.yaml` | 当前磁盘 production route，BiME-Rank v2 |
| `projects/active/terpene_screening/` | 模型与评测源码；现有旧状态文档转向本审计 |
| `scripts/catalyst_finder/` | 当前服务命名空间，保留兼容路径 |
| `reproducibility/bime_rank/` | 唯一 claim 索引、SHA 锁、依赖图、分类、清理建议、验证记录 |
| `reproducibility/bime_rank/source_snapshots/` | 原先只在 ignored results 中的复现脚本快照与来源 SHA |
| `docs/archive/bime_rank/20260907/` | 五份 superseded 文档及修复前路由 SHA 记录 |
| `results/bime_rank_unified_v1/` | 当前结构/上下文/分层证据及必须保留的专家否决证据 |
| `results/catalyst_clean_mainline_v1/`、`results/unified_safe_system_v1/` | 当前仍实际依赖的基础模型、E2R V3/V4 ranker，保持原位 |
| `results/clipzyme_native_extension_v1/` | 冻结公平评测、native embedding、输入支持证明 |
| `results/requested_r2e20_bime_v2_20260906/` | 当前 success-first 推荐与原始多阶段结果；旧包在 `historical/pre_asset_audit_20260907/` |
| `data/`、`external_models/`、`local_candidate_libraries/` | 原始数据、候选身份、模型与特征；具体依赖见锁清单 |

资产目录扫描覆盖 461,514 个文件，逻辑文件大小合计 114.84 GB，分为 1138 个目录/文件组。此数字不是 `du` 实际占块量；符号链接单列。
`inventory.json` 是全资产目录组清单；`protected_files.json` 是当前已识别依赖的文件级清单；未识别组明确为 `unclassified_preserve`，不是已确认垃圾。

## 唯一 canonical 清单

| Claim ID | Primary | 口径 |
|---|---|---|
| `production` | `configs/production_routes/terpene_v1.yaml` | Current disk production route v2; live process revision is separately audited |
| `candidate_universe` | `data/catalyst_candidate_universes/general_merged/manifest.json` | 185918 proteins / 11081 reactions; not Enzyme-405 augmented pool |
| `clipzyme_r2e` | `results/bime_rank_unified_v1/r2e_structure_external_confirmation_v1/summary.json` | Strict double-cold 144 queries / 166202 shared protein candidates |
| `clipzyme_e2r` | `results/clipzyme_native_extension_v1/e2r_strict650_clipzyme_v4_fair_v1/summary.json` | Strict double-cold 248 queries / 10131 shared reaction candidates |
| `enzyme405` | `results/bime_rank_unified_v1/enzyme405_complete226_augmented_v1/summary.json` | 226 queries; frozen 186170 augmented global pool projected to immutable per-query support; seeds40-44 official comparator |
| `multi_seed` | `results/bime_rank_unified_v1/multiseed_scaling_v1/summary.json` | Nested 1/2/3/5 known-positive inputs, same hidden target; distinct from random training seeds |
| `expert_admission` | `projects/active/terpene_screening/BIME_RANK_EXPERT_ADMISSION_V1.json` | Promoted CLIP and seed context; rejected homology, reciprocal and CAGE experts are retained as evidence |
| `cost_aware` | `projects/active/terpene_screening/BIME_RANK_COST_AWARE_HIERARCHY_V1_RESULT.json` | Execution policy and retention evidence, not a new ranking algorithm |
| `r2e_seed_retention` | `results/bime_rank_unified_v1/r2e_seed_context_retention_v1/summary.json` | One-known-positive strict temporal retention |
| `e2r_seed_retention` | `results/bime_rank_unified_v1/e2r_seed_context_retention_v1/summary.json` | One-known-positive strict temporal retention |
| `wetlab_success_first` | `results/requested_r2e20_bime_v2_20260906/MANUAL_SUCCESS_FIRST_SUMMARY.json` | 20 rows / 17 reactions / 16 primary constructs / 27 including backup; 7 manual overrides; predictions not activity measurements |
| `judge_evidence` | `projects/active/terpene_screening/BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json` | Current numeric presentation authority; judge-facing TeX is tracked under `docs/release/bime_rank/` and validated against canonical evidence |

## 实际调用链和模型身份

`serve.py` 引入双向 routing graph → `core/routing.py` 解析 `terpene_v1.yaml` → `rank_open_world.py` 按适用条件调用 BiME R2E/E2R runtime。
R2E 使用 bounded-center ESM-C 与 EnzGFM+center 的冻结 LambdaRank，结构输入可用时接入 CLIPZyme，已知阳性条件下使用冻结 seed-context。
E2R 的 V3 fallback 实际读取 `experts/{enzgfm,esmc,equalblock,rdkitplus}/summary.json`、模型、蛋白和反应特征；V4 在结构可用时融合 CLIP，支持已知反应上下文。
因此 V3/V4、Catalyst bundle 名与旧样式物理路径不能批量替换。它们是当前依赖身份，不代表当前系统仍停留在旧版。`dependency_graph.json` 展开这些 manifest 依赖。

磁盘 general universe 是 185,918 proteins / 11,081 reactions。Enzyme-405 的 186,170 是该评测独立的 augmented universe，不能覆盖 production 候选库。CLIP 结构资产的可支持蛋白数和严格公平集的 166,202 也有不同筛选条件。

在线进程 PID 919268 上报 build revision `c59c3b71f6bc`，早于本次审计的 Git HEAD；`live_observation.json` 保留观测。因此本次确认的是磁盘 production 配置与资产，不把在线旧进程视为已完整加载新代码。没有重启服务或动隧道。

## 数字身份与历史资产

- 严格 CLIP R2E Top-20：8.33% → 15.97%；E2R：8.47% → 15.32%，各自对应索引中的固定 query/candidate support。
- 两个 CLIP strict claim 的上游 support 来自 Rhea release128→141 v2；原冻结 protocol/builder 保持字节身份不变。release128/release141 `rhea2uniprot_sprot.tsv` 作为 external data 以 URL/bytes/SHA 锁定；独立的 `rebuild_rhea128_to141_strict_support_v2.py` 复用原映射逻辑，并从 tracked clean2023 构造一致性 witness，可在不依赖历史 compact cache 的情况下确定性重建 1,122-pair / 208-reaction support，`test_pairs.csv` 与冻结文件逐字节一致、SHA 为 `9a53a465...`。
- Enzyme-405 最终 augmented SR@10：CAGE 五 seed 均值 51.33% → BiME 54.42%；MRR 0.25166 → 0.28641。原来 49.12% 那一版已 superseded。置信区间保留在该 primary，不能把点估计领先写成所有指标显著领先。
- TPS practical legacy_exact Catalyst、Selenzyme frozen Catalyst V3 与两条 MARTS strict route confirmation 均已从 current canonical 降级；current judge 不再展示这些前身模型的定量结果，避免与当前 BiME-v2 混写。
- 旧 `CATALYST_TPS_MARTS_R2E_SYMMETRY_CONFIRM_V1` 只是 2026-07 的内部 MARTS R2E 路由确认；当前 production/runtime、BiME-Rank V2 scorecard 和评委报告均不使用其数字。2026-09-07 复核还发现 protocol 中声明的 baseline 与保留的逐查询 paired evidence 不能完整对齐，因此该实验从 canonical release claim 降为 historical/supplemental evidence，不要求为科研 release 补跑。
- 同期 MARTS E2R dual-kernel confirmatory 也属于 pre-BiME Catalyst 内部路由 lineage；虽然其生成器和结果仍可复现，但当前 production、V2 scorecard 和 judge release 均不使用该数字，因此同样降为 historical/supplemental，而不是 current canonical。
- multi-seed 是多已知阳性输入的 1/2/3/5-seed 对照，不等于五个训练随机种子。EnzymeCAGE seeds40–44 是另一种 seed。
- 当前 wet-lab 包有 20 行 / 17 反应 / 16 主构建 / 27 含备选构建，7 个人工覆盖。它是 success-first 预测推荐包，未发现可据此认定的实测活性结果。
- homology、reciprocal consistency、CAGE Top20 未准入的结果是当前方法的负证据，保留为 canonical admission 依赖。

`canonical.json:superseded` 给出旧路径→当前 claim 的映射。旧评测数据原位保留，已明确 superseded 的目录增加 `ASSET_STATUS.json`；历史复现脚本仍可读取旧输入，但不能用作当前报告入口。
五份旧文档原文移入 archive，原位置变为重定向。旧 JSON ledger/provenance 明确标记 superseded。专家 admission 的最初 staging 审计保留原身份，另加 current production pointer。
不能整目录删除旧 Enzyme-405：当前 augmented evaluator 还引用其 `protein_feature_audit.csv`，它是复现输入而非当前分数来源。

## 实际修复与验证

1. 修复 production-v2 audit 中生产/候选 route 的过期 SHA，保留修复前记录。
2. 修复 wet-lab 包 summary 的两个 FASTA 文件名，重新打包；旧 ZIP 和旧 summary 原样归档。
3. 新增显式选择的 canonical claim 索引、完整依赖图、关键文件大小/mtime/SHA、目录库存及历史映射。
4. 保存 ignored one-off Python 源码的逐字快照和原路径，防止复现只依赖未跟踪脚本。
5. 新增只读 resolver，摘要变化或 superseded primary 会直接失败，不会回退到 latest/旧结果。
6. 当前资产校验通过：canonical primary、8 个 route 引用 SHA、生产/候选语义一致、20/17 行数、16/27 FASTA 序列数、ZIP 内容一致；现有路由/上下文/分层契约测试 11 项及新增 resolver 负向测试 4 项通过，共 15 项。CLIP provenance 与 Enzyme-405 pairs SHA 也核对通过。

运行方式（不会重跑实验）：

```bash
.venv/bin/python scripts/maintenance/resolve_bime_asset.py --verify
.venv/bin/python scripts/maintenance/resolve_bime_asset.py enzyme405
.venv/bin/python scripts/maintenance/validate_bime_assets.py
```

只有明确审阅了资产变更才运行 `build_bime_asset_index.py` 刷新锁；它扫描文件和计算小文件 SHA，不选模、不自动晋升结果。不要用刷新锁掩盖意外文件变化。

## 清理、歧义与范围

- 本次实际删除 0 个数据文件。可再生 Python bytecode 合计 9.84 MB，路径列在 `cleanup_proposal.json`；本次也未删除。
- 没有确认任何同时满足“重复、无当前消费者、可廉价精确重建”的大 cache。equalblock 约 2.38 GB 的 general 特征虽可由两源重建，但当前 E2R 正在引用，不能清。
- 旧文件未被 canonical 选择，不等于无价值。库存里尚未完成逐项语义判定的长尾目录明确保留；这部分不声称已经全量确认为历史实验。
- 依赖扫描留下 5 个 `data/sota/*` 路径，来自 Horizyn YAML；运行时代码确实使用该 YAML 生成指纹，但训练/测试字段是否被当前外部组件执行路径读取尚未做昂贵重放验证。保留记录，不凭空改路径。
- 大 embedding/checkpoint 没有全部重新计算内容 SHA；锁清单明确区分已重算 SHA 和只记录 metadata 的文件，原 manifest 中已有的内容摘要保留。精确恢复仍需要服务器原始大资产或其备份，Git 仅保存代码和索引。
- 基准审计时项目树中尚无当前 `BiME_Rank_Judge_Report`。随后 release continuation 已将资料库 `BiME_Rank_Judge_Report_20260907_v9_1.tex` 逐字节导入 `docs/release/bime_rank/`，并仅按 canonical primary 修正发现的数值漂移；生成 PDF 仍作为构建产物而非数值权威。旧 `zz_model_workflow_report.tex` 属历史材料。
- 在线服务 revision 与磁盘不同；另外在线 association count 246,283 与候选 manifest 的 246,610 属不同观测，尚未证明统计语义相同，未覆盖任何一方。

## Release continuation（2026-09-07）

在上述基准审计之后继续完成了两项只整理、不重跑实验的 release 收尾：

1. 将 420 个经过静态依赖、进程/链接和 canonical 保护检查后确认可移出主视野的历史实验目录从 `results/` 搬到本地 `archive/experiments/pre_release_20260907/`，共 3351 个文件、3,649,366,326 bytes；删除仍为 0。另有 30 个存在依赖或身份不充分的候选 fail-closed 留在原位。移动清单、阻断原因和恢复工具分别见 `archive_moves.json`、`archive_plan.json` 和 `scripts/maintenance/manage_bime_archive.py`。
2. 将资料库最新评委 TeX/Bib 纳入 `docs/release/bime_rank/`。原始 `v9_1` TeX SHA-256 为 `8d1ea50f98b6a998fe22d3072ca9d95a2fca2774e470af96f29f7e9e64621c0f`。导入后发现其 CLIPZyme R2E 表仍含旧结果 11.11% / 21.53%，故以 `clipzyme_r2e` primary 为唯一依据校准为 15.97% / 22.22%，并同步 Hit@50 配对 CI 为 [+6.25,+19.44] pp。`validate_bime_judge_report.py` 当前 validator 已扩展为对 CLIP 双向、Enzyme-405、one-seed/multi-seed、cost-aware 与 expert-admission 的 current 展示数字做 fail-closed 校验，并显式禁止旧 Catalyst/MARTS 数字回流。

评委稿现在是受保护的 release narrative asset，但仍不是独立数值权威；`judge_evidence` 的 primary 继续是 `BIME_RANK_RETRIEVAL_CAPABILITY_SCORECARD_V2.json`，所有展示值必须能回到对应 canonical claim primary。

## 轻量叙事建议（本轮不实施）

1. 以后文稿表格由 claim ID 读取 primary，自动填入协议、候选支持和指标，减少手写数字漂移。
2. 将通用零样本、已知阳性上下文、实验可行性决策三层分别展示；wet-lab recommendation 单独保留人工决策来源。
3. 专家准入与否决放一张简表，cost-aware 作为执行层说明，不额外包装成新模型。
