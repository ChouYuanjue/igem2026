# TerpeneNavigator 第二轮：Conformal Retrieval Sets 与循环权重网格（2026-08-05）

## 1. 第二轮目标与边界

本轮继续增强开放世界 TPS 双向检索，但不改变已经冻结的生产模型、自动路由、候选宇宙、原始 `score` 或 `rank`。新增内容分成两类：

1. **Conformal Retrieval Sets**：把单一 Top-K 列表扩展为带目标覆盖率的候选集合诊断；
2. **循环一致性权重网格**：系统地评估 0–0.20 的循环证据权重和适用域门控，只有独立确认面板稳定受益时才允许提出新路由。

本轮结论是：Conformal 集合可以作为生产旁路契约上线；循环网格没有产生路由晋级候选，因此继续保持证据层，不修改生产排序。

## 2. Conformal Retrieval Sets 定义

版本：

```text
terpene-conformal-retrieval-sets-v1
```

方法：

```text
normalized_best-positive-rank_split-conformal
```

对每个校准查询，令候选宇宙大小为 `N`，最佳已知正例排名为 `r+`。非一致性分数定义为：

```text
s = (r+ - 1) / (N - 1)
```

对于目标错误覆盖率 `alpha`，使用有限样本 split-conformal 次序统计量：

```text
qhat = 第 ceil((n + 1) * (1 - alpha)) 小的校准分数
```

在生产候选宇宙 `N_prod` 中，集合大小为：

```text
K_conf = ceil(qhat * (N_prod - 1) + 1)
```

最终 conformal retrieval set 是生产排名的前 `K_conf` 个候选。该构造的目标是：在锁定的、query-disjoint、double-cold 协议及可交换性假设下，以边际方式覆盖**至少一个已知正例**。它不表示集合中每个候选的活性概率，也不保证真实实验一定阳性。

## 3. 校准数据和防泄漏设计

校准输入来自已锁定的双冷查询级不确定性资产：

```text
results/terpene_open_world_uncertainty_rrf_routing/e2r_query_uncertainty_features.csv
```

关键规则：

- 以唯一 `query_id` 为校准单位；
- 同一查询在多个双冷 cell 中出现时，使用最差 `best_positive_rank` 合并；
- 通过 SHA-256 确定性分成 calibration/test，两部分查询完全不重叠；
- benchmark 候选宇宙为 1,421 个蛋白和 453 个反应；
- 通过归一化 rank 阈值运输到当前绑定的生产宇宙：2,085 个蛋白和 753 个反应；
- 校准器同时绑定 route ID、candidate universe SHA-256 和 model bundle version；
- 任一绑定不匹配时输出 `incompatible_calibrator`，不继续生成集合。

共得到 1,215 个 query calibration units、6 个方向/目标校准器。支持 `alpha=0.20/0.10/0.05`，对应 80%/90%/95% 目标覆盖率。

## 4. 全局校准结果

测试集验证使用一侧 99% 二项容差：只要观察到的 miss 数不超过名义 `alpha` 下的 99% 上界，就认为与目标覆盖率相容。该门槛不会把有限测试集的正常抽样波动错误判为校准失败。

| 方向 | 目标 | alpha | 目标覆盖 | 生产集合大小 | 占候选宇宙 | 测试实测覆盖 | miss / 99%允许上限 |
|---|---|---:|---:|---:|---:|---:|---:|
| 反应 → 酶 | top10 | 0.05 | 95.0% | 1638 | 78.6% | 97.0% | 3/11 |
| 反应 → 酶 | top10 | 0.10 | 90.0% | 1476 | 70.8% | 88.0% | 12/18 |
| 反应 → 酶 | top10 | 0.20 | 80.0% | 972 | 46.6% | 77.0% | 23/30 |
| 反应 → 酶 | top20 | 0.05 | 95.0% | 1719 | 82.4% | 97.0% | 3/11 |
| 反应 → 酶 | top20 | 0.10 | 90.0% | 1509 | 72.4% | 89.0% | 11/18 |
| 反应 → 酶 | top20 | 0.20 | 80.0% | 1055 | 50.6% | 75.0% | 25/30 |
| 反应 → 酶 | top3 | 0.05 | 95.0% | 1638 | 78.6% | 97.0% | 3/11 |
| 反应 → 酶 | top3 | 0.10 | 90.0% | 1476 | 70.8% | 88.0% | 12/18 |
| 反应 → 酶 | top3 | 0.20 | 80.0% | 972 | 46.6% | 77.0% | 23/30 |
| 酶 → 反应 | top10 | 0.05 | 95.0% | 584 | 77.6% | 92.9% | 8/12 |
| 酶 → 反应 | top10 | 0.10 | 90.0% | 464 | 61.6% | 89.4% | 12/19 |
| 酶 → 反应 | top10 | 0.20 | 80.0% | 309 | 41.0% | 84.1% | 18/33 |
| 酶 → 反应 | top20 | 0.05 | 95.0% | 379 | 50.3% | 95.6% | 5/12 |
| 酶 → 反应 | top20 | 0.10 | 90.0% | 306 | 40.6% | 85.8% | 16/19 |
| 酶 → 反应 | top20 | 0.20 | 80.0% | 194 | 25.8% | 79.6% | 23/33 |
| 酶 → 反应 | top3 | 0.05 | 95.0% | 529 | 70.3% | 98.2% | 2/12 |
| 酶 → 反应 | top3 | 0.10 | 90.0% | 439 | 58.3% | 91.2% | 10/19 |
| 酶 → 反应 | top3 | 0.20 | 80.0% | 281 | 37.3% | 81.4% | 21/33 |

所有 18 个全局方向/目标/alpha 组合均通过有限样本容差检查。结果同时暴露了当前任务难度：R2E 的 90% retrieval set 通常需要约 1,476–1,509 个蛋白，而 E2R 的 90% set 约 306–464 个反应。系统不会把这种大集合包装成“高精度小面板”。

## 5. Applicability-aware Mondrian 分组

查询只根据推理时可见的适用域诊断分组：

```text
strong   = reference_library / in_domain
moderate = near_domain
weak     = weakly_supported / far_out_of_domain
```

一个分组只有在满足以下条件时才启用：

- calibration 查询数不少于 20；
- test 查询数不少于 20；
- 通过 99% 二项容差；
- test 实测覆盖率不低于名义目标。

否则自动回退到全局校准器。当前共有 21 个分组/alpha 组合通过启用门槛。部分示例：

| 方向 | 目标 | alpha | 分组 | 生产集合大小 | 测试实测覆盖 |
|---|---|---:|---|---:|---:|
| 酶 → 反应 | top10 | 0.05 | strong | 579 | 100.0% |
| 酶 → 反应 | top10 | 0.10 | moderate | 447 | 93.3% |
| 酶 → 反应 | top10 | 0.10 | strong | 514 | 92.3% |
| 酶 → 反应 | top10 | 0.20 | moderate | 308 | 93.3% |
| 酶 → 反应 | top10 | 0.20 | strong | 334 | 80.8% |
| 酶 → 反应 | top20 | 0.05 | strong | 368 | 100.0% |
| 酶 → 反应 | top20 | 0.10 | strong | 274 | 94.4% |
| 酶 → 反应 | top20 | 0.20 | strong | 155 | 91.7% |
| 酶 → 反应 | top3 | 0.05 | moderate | 551 | 100.0% |
| 酶 → 反应 | top3 | 0.05 | strong | 356 | 100.0% |
| 酶 → 反应 | top3 | 0.10 | moderate | 506 | 95.4% |
| 酶 → 反应 | top3 | 0.10 | strong | 283 | 100.0% |
| 酶 → 反应 | top3 | 0.20 | strong | 171 | 86.7% |
| 反应 → 酶 | top10 | 0.05 | weak | 1638 | 100.0% |
| 反应 → 酶 | top10 | 0.10 | weak | 1520 | 98.4% |
| 反应 → 酶 | top20 | 0.05 | weak | 1763 | 100.0% |
| 反应 → 酶 | top20 | 0.10 | weak | 1572 | 95.4% |
| 反应 → 酶 | top20 | 0.20 | weak | 1136 | 84.6% |
| 反应 → 酶 | top3 | 0.05 | weak | 1638 | 100.0% |
| 反应 → 酶 | top3 | 0.10 | weak | 1520 | 98.4% |
| 反应 → 酶 | top3 | 0.20 | weak | 1058 | 82.3% |

分组只改变集合阈值，不改变生产候选排序。

## 6. CLI、API 和批处理行为

默认模式为注解：

```bash
.venv/bin/python projects/active/terpene_screening/rank_open_world.py   rank-enzymes --reaction-smiles 'CCO>>CC=O'   --query-id EXTERNAL_RXN --top-k 10   --conformal-mode annotate --conformal-alpha 0.10   --output /tmp/external_r2e.csv
```

CSV 仍只返回请求的 Top-10，但会记录：

- `conformal_set_size`；
- `conformal_set_fraction`；
- `conformal_set_truncated`；
- `conformal_set_member`；
- 校准器、绑定状态、适用域分组和验证覆盖率。

需要完整集合时显式使用：

```bash
--conformal-mode expand
```

`expand` 会保持原始 `ranking_objective` 和 route，只把返回深度扩到锁定阈值。第一次短查询确定的适用域分组和 `qhat` 会被冻结，扩展过程不会反向改变自身集合定义。

真实外部反应 `CCO>>CC=O` 的 Top-10 R2E smoke：

- route 保持 `r2e-external-top10-v1`；
- `weak` Mondrian 组；
- 90% conformal set 为 1,520 个蛋白；
- annotate 返回 10 行并标记 truncated；
- expand 返回 1,520 行；
- 两种模式 Top-1 均为 `C5I9X1`，原始排序未改变。

API 查询对象新增：

```text
query.conformal_retrieval_set
```

注册表批处理使用同一校准器，但固定为 annotate-only，避免一个查询自动生成上千行而破坏标准 Top-3/10/20 产物。

## 7. 循环一致性第二轮网格

入口：

```bash
.venv/bin/python scripts/evaluate_terpene_cycle_rerank_grid.py
```

完整实验使用：

- 6 个注册外部酶查询；
- 6 个注册外部反应查询；
- 每个方向 3 个 development、3 个 confirmation；
- Top-3/10/20 三条生产 objective route；
- 前向 Top-20、反向 Top-50；
- 循环权重 0、0.05、0.10、0.15、0.20；
- `all`、`applicability >= 0.60`、`applicability >= 0.80` 三种门控。

这是“注册表已知关联代理面板”，不是新的 double-cold 独立确认集，因此只有筛选价值，不能替代正式生产晋级验证。

开发面板中，只有 E2R Top-20 的权重 0.20 出现很小的 MRR 增益 `+0.00463`，Hit@3/10/20 均未变化。到独立确认面板后：

- 所有选择配置新增命中数为 0；
- 丢失命中数为 0；
- MRR 增益为 0；
- 0 个配置达到生产晋级条件；
- 18 个方向/目标/预算判断全部为 `evidence_only_no_route_change`。

因此循环一致性仍用于解释、审计和候选证据，不加入生产 `score` 或 `rank`。

## 8. 生产安全边界

当前实现保证：

- Conformal 默认只注解；
- `expand` 只扩展同一路由的返回前缀；
- 当前库实体标记 `not_applicable_current_entity`；
- 候选宇宙、route 或 model bundle 不匹配时拒绝旧校准器；
- masked/few-shot/manual override 等未校准干预不会冒用默认集合保证；
- conformal set 不是候选活性概率；
- 循环网格结果不自动修改生产路由；
- 原有 golden 排名、score source 和单/批一致性必须继续通过。

## 9. 关键资产

```text
projects/active/terpene_screening/core/conformal.py
scripts/prepare_terpene_conformal_retrieval_sets.py
scripts/evaluate_terpene_cycle_rerank_grid.py
results/terpene_conformal_retrieval_sets/calibrators.json
results/terpene_conformal_retrieval_sets/validation_metrics.csv
results/terpene_cycle_rerank_grid_v2/summary.json
results/terpene_cycle_rerank_grid_v2/confirmation_metrics.csv
```

这些资产将进入 portable runtime manifest 和服务器部署清单 v4。
