# 项目结构

当前仓库明确分开 **BRIDGE 科研核心**、**COMPASS 产品实现**、**冻结复现证据** 与 **历史研究归档**。

| 路径 | 角色 | 当前权威性 |
| --- | --- | --- |
| `projects/active/bridge/` | BRIDGE 科研核心与当前检索运行接口 | 当前方法实现 |
| `scripts/starase_navigator/` | 智能体、语义规划、检索编排、证据获取 | 当前应用实现 |
| `frontend/starase_navigator/` | COMPASS 前端 | 当前产品界面 |
| `configs/production_routes/` | 部署路由和模型合同 | 当前生产配置 |
| `reproducibility/bridge/` | BRIDGE claim 映射和新实验命名空间 | 当前 claim 入口 |
| `reproducibility/bime_rank/` | BiME-Rank 与后续冻结历史实验 | 冻结复现/历史证据 |
| `archive/fibre/20261003/` | FIBRE 完整源码、理论、旧工具链 | 历史审计 |
| `archive/terpene_screening/` | TPS/Catalyst 更早支线 | 历史审计 |
| `scripts/maintenance/` | release、manifest、资产和仓库校验 | 当前维护工具 |

## BRIDGE 核心

`projects/active/bridge/` 按责任组织：

- `core/`：候选宇宙、路由、适用域、来源和生产契约；
- `runtime/`：Broad 排序、BiME 遗留强基线与专家运行接口；
- `evidence/`：结构、机制、家族、上下文证据；
- `application/`：全信息应用态和 TPS 专项能力；
- `pipelines/`：数据与证据构建；
- `portable/`：便携数据和基础检索构建；
- `docs/`：当前方法、工程史、评测和复现说明；
- `release/`：`bridge-method`、`bridge-reproduction`、`starase-application` 三种发布边界。

BRIDGE 的默认排序权属于 Broad。专家是否运行、是否拥有排序权限、能修改多大范围，由查询适用性、证据可用性、方向和冻结准入共同决定。

## 历史实现兼容

迁移前的 FIBRE 源码已完整移动到 `archive/fibre/20261003/implementation/`。当前 BRIDGE 包中仍可能读取历史名称的结果目录、schema 或冻结模型资产；这些名称承担兼容和 provenance 作用，不代表 FIBRE 仍是现役方法。

## BiME-Rank 边界

BiME-Rank 仍是 BRIDGE 的直接前身。R2E LambdaRank、E2R Anchored LambdaMART、结构专家准入、seed context 和 cost-aware hierarchy 等能力构成 BRIDGE 的重要基础。其冻结材料继续保存在 `reproducibility/bime_rank/`，不回写历史文件身份。

## FIBRE 边界

FIBRE 的统一交互图册、条件催化模式、统一关系核心已经结束。该分支的有效工程结论——missing-neutral、插件化、查询级适用性和保护 Broad——被 BRIDGE 吸收；原方法定义和理论只存在于 archive。

## 验证

```bash
PYTHONPATH=. .venv/bin/python -m compileall -q projects/active/bridge scripts
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
```
