# BRIDGE 科研核心

**BRIDGE — Broad Retrieval with Inference-Driven Gated Experts** 是当前酶—反应检索方法。

BRIDGE 以 Broad Retrieval 提供全候选空间的稳定基础顺序，再根据当前查询的信息可用性、生化适用域和方向性证据决定哪些专家可以参与。专家只提供受限修正；缺失或不适用的专家保持静音，不会把“缺少证据”解释成负证据。

## 当前结构

- `core/`：候选宇宙、路由、适用域、来源与生产契约。
- `runtime/`：Broad 排序、BiME-Rank 遗留强基线、结构专家和当前专家运行接口。
- `evidence/`：结构、机制、家族、上下文等可插拔证据。
- `application/`：当前全信息应用态和 TPS 专项能力。
- `pipelines/`：数据、证据与应用资产构建。
- `portable/`：便携数据和基础检索构建工具。
- `docs/`：当前 BRIDGE 方法、工程史、评测和复现说明。

部分源码仍包含冻结历史资产名或 schema 名，例如 `BRIDGE_*` 结果文件、旧结果目录和 BiME-Rank 模型包。这些名称用于保证历史哈希、结果路径和复现合同连续，不能据此判断当前方法身份。

## 方法核心

最终排序写成：

\[
S_{\mathrm{BRIDGE}}(q,e)
= S_{\mathrm{Broad}}(q,e)
+ \sum_k g_k(q)\,\Delta_k(q,e).
\]

- `S_Broad`：始终存在的广域基础顺序；
- `g_k(q)`：第 `k` 个专家对当前查询的适用性与权限；
- `Delta_k`：通过准入后允许施加的有限修正。

BRIDGE 的工程原则来自长期试错：Broad 负责普遍覆盖；专家提供局部知识；强基础序受到保护；缺失信息不产生惩罚；专家价值按查询和方向判断。

## 历史边界

FIBRE 的交互图册、条件催化模式和统一关系核心已经归档到：

`archive/fibre/20261003/`

BiME-Rank 保留为 BRIDGE 的直接前身和冻结基线。早期 TPS/CAGE、Broad、连续学习、反应中心、BRIDGE 等完整分支见 `docs/engineering.md`。

## 阅读入口

1. `docs/method.md`：当前方法定义与叙事主线。
2. `docs/engineering.md`：完整工程决策树，包括废弃分支。
3. `docs/evaluation.md`：当前评测口径和关键结果。
4. `docs/status.md`：当前实现边界和历史资产关系。
5. `docs/reproducibility.md`：冻结证据、旧文件名和复现规则。
