# 文档地图

仓库文档按“当前方法、冻结复现、历史归档”分层。文件名中的日期、`current`、`final` 或旧方法名都不能单独决定权威性。

## 当前 BRIDGE 文档

- `../projects/active/bridge/README.md` — 当前方法总入口。
- `../projects/active/bridge/docs/method.md` — BRIDGE 主叙事与方法定义。
- `../projects/active/bridge/docs/engineering.md` — 全量工程决策树和废弃分支。
- `../projects/active/bridge/docs/evaluation.md` — 评测协议和当前关键结果。
- `../projects/active/bridge/docs/status.md` — 当前实现边界。
- `../projects/active/bridge/docs/reproducibility.md` — BRIDGE 与历史证据的复现关系。
- `../reproducibility/bridge/canonical.json` — 当前 claim 到冻结历史证据的机器映射。
- `project_structure.md` — 当前仓库结构。
- `RESEARCH_RELEASE.md` — 科研发布边界。

## 冻结前身与复现资产

`../reproducibility/bime_rank/` 保存 BiME-Rank 及其后的冻结实验、候选宇宙、模型资产、评测脚本和历史结果。BiME-Rank 是 BRIDGE 的直接前身，仍是重要比较与复现基线。

其中部分后期结果生成于 BRIDGE 名称最终冻结之前，因此文件仍带 `FIBRE_*` 名称。当前 claim 通过 `../reproducibility/bridge/canonical.json` 引用这些不可变文件。

## FIBRE 历史归档

`../archive/fibre/20261003/` 保存 FIBRE 的完整源码快照、方法/理论文档和旧工具链。FIBRE 只用于工程史与审计，不再定义当前模型。

## 其它历史文档

- `archive/pre_bridge/20261003/` — BRIDGE 定稿前的长篇检索报告和旧叙事。
- `archive/bime_rank/20260907/` — BiME-Rank 当时的冻结可读报告。
- `../archive/terpene_screening/` — 更早 TPS/Catalyst 研究支线。

## 稳定支持文档

仍与当前运行契约相关的早期协议文档可以保留，例如 taxonomy scope、conformal retrieval set、数据 schema 和第三方依赖说明。它们提供局部协议来源，不承担当前方法身份。

## 历史 demotion 审计

BiME-Rank release 仍保留以下 Git 历史审计表：

- `../reproducibility/bime_rank/historical_source_demotions.json`
- `../reproducibility/bime_rank/historical_research_source_demotions.json`
- `../reproducibility/bime_rank/historical_artifact_demotions.json`

它们描述哪些来源被降为历史 lineage。开发机本地、明确不进入版本控制的附加路径继续由 `.git/info/exclude` 管理。CI 的冻结验证产物仍使用 `bime-rank-release-validation-<commit>` 命名，以保证历史 release 可追溯。
