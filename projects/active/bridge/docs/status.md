# BRIDGE 当前状态

## 当前身份

当前科研方法：**BRIDGE — Broad Retrieval with Inference-Driven Gated Experts**。

当前用户产品：**Starase Navigator**。

直接前身：**BiME-Rank**。

已归档研究分支：**FIBRE**。

## 当前排序结构

1. Broad Core 生成全候选空间基础顺序。
2. 查询适用性层判断专家是否有资格参与。
3. 通过准入的专家产生有限修正。
4. 缺失、不适用或未获排序权限的专家贡献严格为零。
5. 专项专家仅在明确家族范围内激活。

## 固定七类综合测试

当前冻结记录中：

- Broad Core：MRR 约 0.2093，Hit@10 约 36.87%，Hit@100 约 59.41%，Hit@1000 约 77.91%；
- 完整专家系统：MRR 约 0.2369，Hit@10 约 41.54%，Hit@100 约 64.92%，Hit@1000 保持约 77.91%。

专家主要改善头部排序，Broad 保持总体候选覆盖。

## CAGE 分层对比

统一七类实验中：

- 原始 CAGE：MRR 约 0.1008，Hit@10 约 20.53%；
- Broad Top-1000 后直接使用通用 CAGE：MRR 约 0.0257，Hit@10 约 5.44%；
- 完整系统：MRR 约 0.2369，Hit@10 约 41.54%。

Broad Top-1000 能覆盖约 77.91% 查询的至少一个阳性，而通用 CAGE 可原生评分的阳性查询约 25.08%。该结果支持“结构专家覆盖范围必须显式建模”。

## 专项专家

家族专项 CAGE 在各自官方测试中存在局部增益；TPS 专项修正只在稀疏、明确的 TPS 适用域中激活。综合七类测试中专项专家只激活很少查询，因此全局平均变化很小，这符合设计目标。

## 历史资产名

若当前代码、结果目录或 JSON 中仍出现 `BRIDGE_*`，其含义是“冻结历史证据或迁移前资产名”。当前方法身份只由 `projects/active/bridge/`、`projects/active/bridge/docs/` 与 `reproducibility/bridge/` 定义。
