# 萜类合酶检索项目：跨服务器快速复刻说明

本文说明 Git 仓库中保存了什么、没有保存什么，以及怎样在另一台 Linux 服务器上恢复可运行的生产检索与湿实验工作流。

## 1. 复刻目标

一次正常的 `git clone` 加 bootstrap 应恢复：

- 反应 → 酶、酶 → 反应的全部自动路由；
- 五组自训练生产集成权重；
- E2R Top-20 双核协同资产；
- 经验可靠性校准器；
- 当前库、MARTS 注册库和 UniProt rescue 所需的聚合表示与元数据；
- 持久开放注册表；
- 批量发现、受控 UniProt 扩展和六板湿实验的最终可展示产物；
- 训练、评测、验证和报告生成代码；
- 固定版本的外部 Horizyn 源代码及其官方检查点恢复方式。

## 2. 最快恢复方式

```bash
git clone git@github.com:ChouYuanjue/igem2026.git
cd igem2026
bash scripts/bootstrap_terpene_runtime.sh
```

完整检查：

```bash
bash scripts/bootstrap_terpene_runtime.sh --full-check
```

已有虚拟环境时：

```bash
VENV_DIR=/path/to/venv bash scripts/bootstrap_terpene_runtime.sh --skip-install
```

只校验现有文件：

```bash
bash scripts/bootstrap_terpene_runtime.sh --verify-only
```

## 3. 已提交的不可轻易再生成资产

### 3.1 自训练生产权重

```text
results/terpene_production_models/
├── marts_adapted_drfp_pu/
├── marts_adapted_drfp_pu_r2e075/
├── marts_adapted_drfp_pu_r2e_exact_residual/
├── marts_adapted_drfp_pu_e2r/
├── marts_adapted_drfp_pu_e2r_hardneg128/
└── marts_dual_kernel_e2r_top20/
```

其中提交：

- 15 个普通生产集成 checkpoint；
- 3 个 exact-residual checkpoint；
- 自训练的 `reaction_feature_distiller.pt`；
- 反应特征矩阵、辅助表示、注册表和训练 pair；
- 双核支持矩阵、相似矩阵和锁定参数；
- 配置、训练历史、许可证和审计文件。

### 3.2 聚合表示

仓库只保存程序实际读取的聚合矩阵与 ID 映射，不保存成千上万个重复的单条 `.npy`：

```text
data/terpene_embeddings/esmc600m_mean/{embeddings.npy,entries.csv,...}
data/terpene_embeddings/marts_unseen_esmc600m/{embeddings.npy,entries.csv,...}
data/terpene_embeddings/uniprot_tps_primary_esmc600m/{embeddings.npy,entries.csv,...}
```

### 3.3 数据与注册表

提交：

- 当前萜类合酶数据和关联表；
- MARTS 归一化数据；
- 50% identity folds、反应聚类及适配特征；
- 持久开放注册表；
- UniProt 主候选元数据、体系结构 contract 与必要审计输入。

### 3.4 最终展示和实验产物

提交：

- 全注册表 Top-3/10/20 排名与泄漏审计；
- 受控 UniProt rescue 排名；
- 候选发现面板；
- 初始、平衡、随机化孔板；
- 结果模板；
- 六板合并采购 manifest 与 FASTA；
- UniProt rescue campaign；
- 可靠性校准器和关键评测摘要。

这些文件使另一台服务器无需先重新训练或重新跑完整批量任务，就能直接用于前端展示和后续实验。

## 4. 没有提交的内容

### 4.1 Horizyn 官方 checkpoint

`horizyn_v1_0_dev.ckpt` 大约 192 MiB，是 Dayhoff Labs 发布的外部模型，不是本项目训练资产。它有稳定官方来源：

- 源代码：`https://github.com/dayhofflabs/horizyn.git`
- 固定 commit：`e6655e732f574c8bfa0488b9bc5068b67e382745`
- Zenodo DOI：`10.5281/zenodo.20348783`
- 目标 SHA-256：`31bb9b6d73241b7807050377799de8b4bfb17f42a6cd652c8b17b65faf754c25`
- 许可证：PolyForm Noncommercial 1.0.0

bootstrap 会下载、校验并放到：

```text
results/terpene_production_models/
  marts_adapted_drfp_pu_r2e_exact_residual/
  horizyn_v1_0_dev.ckpt
```

这样避免 GitHub 单文件限制，也避免把可稳定下载的第三方大文件永久复制进 Git 历史。

### 4.2 ESM-C 600M 基础模型

现有实体查询使用已提交的聚合 embedding，不需要重新下载 ESM-C。只有输入全新酶序列或注册新酶时需要 ESM-C 600M；`esm` 会在首次调用时从官方模型源下载并缓存。

### 4.3 外部下载缓存和可重算中间文件

不提交：

- `downloads/clipzyme/*.zip` 等外部缓存；
- 每个蛋白各一份的重复向量文件；
- UniProt 原始下载、MMseqs 临时数据库和可从提交的主候选重新建立的冗余副本；
- `__pycache__`、测试缓存、日志和临时文件；
- 大规模消融实验的全部中间 checkpoint。

## 5. 环境

已验证环境：

```text
Linux
Python 3.12.3
PyTorch 2.4.0 + CUDA 12.4
```

核心版本见：

```text
requirements-terpene-runtime.txt
```

PyTorch CUDA wheel 与驱动有关。目标服务器可先安装适合自身 CUDA 的 PyTorch 2.4.0，再运行：

```bash
bash scripts/bootstrap_terpene_runtime.sh
```

## 6. 完整性校验

Git 中的每个便携资产都有 SHA-256：

```text
reproducibility/terpene_runtime_manifest.json
```

运行：

```bash
.venv/bin/python scripts/verify_terpene_runtime.py
```

校验内容包括：

- 文件是否存在；
- SHA-256 是否一致；
- 关键 `.npy/.npz` 形状是否一致；
- 外部 Horizyn checkpoint 是否已恢复且哈希正确。

## 7. 最小烟雾测试

当前反应找酶：

```bash
.venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-enzymes --reaction-id RHEA:54512 --top-k 3 \
  --output /tmp/r2e_top3.csv
```

已有酶找反应：

```bash
.venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-reactions --enzyme-id 7S5L_A --top-k 20 \
  --output /tmp/e2r_top20.csv
```

外部反应会验证 Horizyn 实时编码：

```bash
.venv/bin/python projects/active/terpene_screening/rank_open_world.py \
  rank-enzymes --query-id smoke_external_reaction \
  --reaction-smiles 'CCO>>CC=O' --top-k 10 \
  --output /tmp/r2e_external_top10.csv
```

## 8. 复刻边界

本提交优先保证：

1. 生产推理可直接运行；
2. 路由与可靠性结果可复现；
3. 前端展示使用的完整结果可直接读取；
4. 湿实验设计与反馈入口可继续使用；
5. 所有自训练或难以重算的资产不丢失。

它不试图把当前 46GB 工作目录原样塞入 Git。可稳定重新下载或由已提交资产确定性重建的内容，通过脚本、固定版本、校验和和文档恢复。

## 9. 生产内核 v1、注册表快照与完整质量门禁

`reproducibility/terpene_runtime_manifest.json` 已升级到 version 3，并将
`configs/production_routes/terpene_v1.yaml` 作为生产契约纳入 SHA-256 校验。
可靠性校准器同时绑定 route ID、模型包版本和方向候选集合哈希；任一不匹配
都会输出 `incompatible_calibrator`，而不是沿用旧分数。

首次从旧仓库复刻时，兼容注册表会自动迁移为不可变快照。生产读取通过
`data/terpene_open_world_registry/CURRENT` 原子切换，旧的
`proteins/*.csv|npy` 和 `reactions.csv` 仍作为兼容镜像保留。执行：

```bash
.venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py snapshot
.venv/bin/python projects/active/terpene_screening/manage_open_world_registry.py status
```

完整质量门禁：

```bash
bash scripts/run_terpene_quality_gate.sh
bash scripts/run_terpene_quality_gate.sh --full
```

普通门禁执行编译、82 项测试、portable manifest、五个神经部署、双核资产、
系统健康、机制特征准备和时间切分 readiness。完整门禁额外执行真实查询
smoke、三条冻结 golden route，以及 R2E/E2R Top-3/10/20 的单查询—批处理
逐候选一致性检查。

每个 CLI 排名 CSV 都会同时产生 `<output>.audit.json`，记录 route、模型、
候选宇宙、注册表、输入质量和可靠性状态。跨服务器复刻后应同时保留 CSV
和审计侧车，避免只保存候选列表而失去生产上下文。
