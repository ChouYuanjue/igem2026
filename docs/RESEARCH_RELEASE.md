# BRIDGE 科研发布合同

发布分支为 `master`。当前科研方法身份为 **BRIDGE — Broad Retrieval with Inference-Driven Gated Experts**；当前用户侧系统为 **Starase Navigator**。BiME-Rank 是直接前身和冻结基线，FIBRE 是已经归档的研究分支。

机器可读的科研资产清单继续由 `reproducibility/research_release_manifest.json` 管理。当前 BRIDGE claim 入口为 `reproducibility/bridge/canonical.json`；它允许当前方法名称引用迁移前产生的不可变历史结果，而无需改写原文件名、schema 或哈希。

## 1. 三个发布角色

### `bridge-method`

只包含方法定义、Broad/专家契约、适用性与缺失中性规则。该包不携带项目训练权重和 benchmark 数据，也不发布性能数字。

### `bridge-reproduction`

冻结当前 claim 需要的数据、候选宇宙、项目训练权重、第三方恢复合同、评测器和历史结果。benchmark 数字只能从这一边界或其直接引用的不可变证据产生。

### `starase-application`

用于当前全信息应用，可使用完整数据库、结构/机制/上下文证据和 TPS/家族专项能力。该包面向实际候选生成和实验规划，不把应用态结果转化为 benchmark claim。

## 2. 当前与历史来源

当前源码：

- `projects/active/bridge/`
- `scripts/starase_navigator/`
- `frontend/starase_navigator/`
- `configs/production_routes/`

冻结前身与历史证据：

- `reproducibility/bime_rank/`：BiME-Rank 与后续冻结实验；
- `archive/fibre/20261003/`：FIBRE 原实现、理论和旧工具；
- `archive/terpene_screening/`：更早 TPS/Catalyst 支线。

历史文件存在不赋予其当前 authority。当前方法身份由 `projects/active/bridge/docs/` 和 BRIDGE release profiles 决定。

## 3. 为什么保留 `FIBRE_*` 结果文件

BRIDGE 定名发生在最后一轮实验之后。已有结果文件的名称、schema、内部路径与 SHA-256 已经成为实验 provenance 的一部分，因此不批量改名。

规则如下：

1. 旧结果保持原字节和原路径；
2. `reproducibility/bridge/canonical.json` 给出当前 claim 名称；
3. 新实验开始使用 BRIDGE 命名；
4. FIBRE 方法源码只从 archive 复现。

这保证“改叙事”不会变成“重写实验历史”。

## 4. 项目自训练资产

`reproducibility/bime_rank/model_assets.json` 仍是项目自训练权重和第三方模型恢复合同的权威清单。BRIDGE reproduction validator 要求所有项目自训练资产保持文件大小和 SHA-256 一致。

大规模 ESM-C、EnzGFM 等基础表示矩阵可以按固定模型版本、候选顺序和构建脚本重建；不适合普通 Git 的第三方大 checkpoint 继续通过固定来源和校验和恢复。

`data/` 与 `results/` 仍采用 deny-by-default 策略。只有 manifest 明确列出的科研资产进入 release；开发机上的任意缓存和临时结果不会因为位于这些目录就获得发布身份。

## 5. claim 与模型选择边界

模型开发遵循：

- development 用于选择；
- frozen/confirmation 用于一次性确认；
- external/temporal 只能确认或否决，不能揭示后回头调参；
- 不同候选宇宙和不同 cold 轴的数字不能直接混表解释；
- 家族专项能力在匹配领域测试中判断，不能由稀疏全局平均值替代。

BRIDGE 当前关键 claim 对应的不可变结果见 `reproducibility/bridge/canonical.json`。

## 6. 当前验证

便携源码合同：

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
```

生成 BRIDGE release manifests：

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/build_bridge_release_manifests.py
```

当前源码与产品回归：

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q projects/active/bridge scripts
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests
```

BiME-Rank 的冻结 release/extended reproduction 仍由原有 `run_reproduction_tests.py` 维护，因为其目的在于验证历史基线和 claim 来源，不代表当前方法仍叫 BiME-Rank。

## 7. 迁移规则

FIBRE 迁移点为 2026-10-03：

- 原 `projects/active/fibre/` 完整快照 → `archive/fibre/20261003/implementation/`；
- 原方法/理论文档随快照归档；
- 原 FIBRE workflow/release tooling 的精确副本 → `archive/fibre/20261003/tooling/`；
- 当前运行 namespace → `projects/active/bridge/`；
- 当前方法文档 → `projects/active/bridge/docs/`；
- 当前 claim namespace → `reproducibility/bridge/`。

迁移不能改变历史实验数据、冻结模型权重或结果哈希。
