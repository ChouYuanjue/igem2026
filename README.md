# Starase Atlas — BRIDGE 与 Starase Navigator

本仓库是 NJU-China iGEM 2026 Starase Atlas 的软件与模型仓库。当前科研方法为 **BRIDGE — Broad Retrieval with Inference-Driven Gated Experts**，用户侧产品为 **Starase Navigator**。

## 方法概览

BRIDGE 从大规模开放候选空间出发。Broad Retrieval 先给出稳定的全局基础顺序；随后根据当前查询、信息可用性、生化适用域和方向性证据，决定哪些专家能够参与。专家只提供受限修正，缺失或不适用的信息保持中性。

\[
S_{\mathrm{BRIDGE}}(q,e)
=
S_{\mathrm{Broad}}(q,e)
+
\sum_k g_k(q)\Delta_k(q,e).
\]

这套结构来自项目的完整工程演化：

**TPS + CAGE → 开放候选空间 → Broad Retrieval → BiME-Rank → 查询级专家适用性 → BRIDGE。**

TPS 最终成为稀疏启用的领域专家；CAGE 从早期主排序器逐步转为家族专项能力。BiME-Rank 保留为 BRIDGE 的直接前身和冻结基线。

FIBRE 的交互图册、条件催化模式、统一关系核心等路线已经结束，完整源码、理论和工具链归档于 `archive/fibre/20261003/`。FIBRE 在当前文档中仅作为工程历史出现。

## 仓库入口

| 内容 | 路径 |
| --- | --- |
| BRIDGE 总览 | `projects/active/bridge/README.md` |
| 当前方法 | `projects/active/bridge/docs/method.md` |
| 完整 Engineering 决策树 | `projects/active/bridge/docs/engineering.md` |
| 当前评测 | `projects/active/bridge/docs/evaluation.md` |
| 当前状态 | `projects/active/bridge/docs/status.md` |
| 当前复现规则 | `projects/active/bridge/docs/reproducibility.md` |
| BRIDGE claim 映射 | `reproducibility/bridge/canonical.json` |
| BiME-Rank 冻结前身 | `reproducibility/bime_rank/` |
| FIBRE 历史归档 | `archive/fibre/20261003/` |
| Starase Navigator | `scripts/starase_navigator/`、`frontend/starase_navigator/` |
| 生产路由 | `configs/production_routes/default.yaml` |

## 安装

测试环境使用 Python 3.12。

```bash
git clone <THE-OFFICIAL-IGEM-GITLAB-CLONE-URL>
cd <repository>
python3.12 -m venv .venv
./scripts/bootstrap_terpene_runtime.sh
```

大型数据库、基础模型、embedding 与派生矩阵通过固定版本和 release manifest 恢复，不把任意开发机缓存混入科研发布。

## 运行 Starase Navigator

```bash
./scripts/starase_navigator/manage.sh start
curl -fsS http://127.0.0.1:8791/api/status | .venv/bin/python -m json.tool
```

自然语言是主要控制入口。系统根据任务选择候选空间、Broad 路由、专家深度和证据获取方式；用户无需选择内部模型版本。

## 直接运行 BRIDGE 检索

反应找酶：

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py \
  rank-enzymes --reaction-id RHEA:54512 --top-k 10 \
  --output /tmp/bridge-r2e.csv
```

酶找反应：

```bash
PYTHONPATH=. .venv/bin/python projects/active/bridge/runtime/cli.py \
  rank-reactions --enzyme-id 7S5L_A --top-k 20 \
  --output /tmp/bridge-e2r.csv
```

## 科研发布与复现

BRIDGE 当前有三个隔离发布角色：

| 发布角色 | 用途 | 可否支撑 benchmark claim |
| --- | --- | --- |
| `bridge-method` | 方法定义、路由和专家契约 | 否 |
| `bridge-reproduction` | 冻结数据、权重、评测与 claim 证据 | 是 |
| `starase-application` | 当前全信息应用与实验辅助 | 否 |

验证：

```bash
PYTHONPATH=. .venv/bin/python scripts/maintenance/validate_bridge_release_profiles.py --source-only
PYTHONPATH=. .venv/bin/python scripts/maintenance/build_bridge_release_manifests.py
PYTHONPATH=. .venv/bin/python -m pytest -q projects/active/bridge/tests
PYTHONPATH=. .venv/bin/python -m pytest -q scripts/starase_navigator/tests
```

当前 BRIDGE 数值仍有部分来源文件带 `FIBRE_*` 历史文件名。它们保持原名称和字节以维持实验 provenance，由 `reproducibility/bridge/canonical.json` 映射到当前 claim。

## 许可证与引用

项目源码使用 MIT License。第三方模型、数据与软件遵循其上游许可证，详见 `THIRD_PARTY_NOTICES.md`。仓库引用信息见 `CITATION.cff`。

### BiME-Rank 冻结前身回归

BRIDGE 迁移不删除 BiME-Rank 的历史复现入口：

```bash
.venv/bin/python reproducibility/bime_rank/run_reproduction_tests.py --tier release
.venv/bin/python reproducibility/bime_rank/run_reproduction_tests.py --tier extended
```
