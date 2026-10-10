# BRIDGE 复现边界

## 1. 当前方法与历史记录分离

当前方法源码位于 `projects/active/bridge/`。FIBRE 原实现已完整归档于 `archive/fibre/20261003/implementation/`。

BRIDGE 没有重写既有实验结果文件。迁移前生成的 `BRIDGE_*` JSON 继续作为不可变历史证据，由 `reproducibility/bridge/canonical.json` 统一映射到当前 BRIDGE claim 名称。

## 2. 为什么保留旧文件名

冻结结果文件名、schema、内部路径和哈希属于实验来源的一部分。批量改名会制造新的文件身份，也会让旧 commit、报告和审计脚本难以对应。因此当前规则是：

- 方法身份和现役源码使用 BRIDGE；
- 历史实验结果保持原字节和原名称；
- 当前 claim 通过 BRIDGE canonical map 指向历史证据；
- 新生成的结果从现在开始使用 BRIDGE 名称。

## 3. 当前复现入口

基础检查：

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q projects/active/bridge scripts
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests
```

当前排名入口：

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py rank-enzymes \
  --reaction-id RHEA:54512 --top-k 10 --output /tmp/bridge-r2e.csv
```

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py rank-reactions \
  --enzyme-id 7S5L_A --top-k 20 --output /tmp/bridge-e2r.csv
```

正式关系留出评价使用 E2R 开发 3,900/1,316 与最终双向 21,505 条相同关系，当前主表、独立消融与 CAGE 原生门控计算见上述 evaluation.md 的复现入口。R2E 在唯一的 5,216 条验证关联上重新拟合关系门控和校准功能专家，然后重算正式测试逐关系排名及四项消融；E2R 联合排序主模型保持冻结。 运行时情景门控同样在唯一验证集内，以查询为折并将支持和预测目标分离；复现入口为 reproducibility/bime_rank/scripts/fit_bridge_episodic_single_validation_v1.py。

## 4. 历史 BRIDGE 复现

历史 FIBRE 研究只能从 `archive/fibre/20261003/` 或冻结 commit 复现。该路径用于审计和工程史，不得重新成为当前生产或方法定义的 authority。
