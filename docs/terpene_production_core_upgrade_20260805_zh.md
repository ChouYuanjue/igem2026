# 萜类合酶检索生产内核 v1 实施报告（2026-08-05）

## 1. 约束与结论

本次修改不改变项目核心作用：系统仍在开放候选空间中完成反应到酶、酶到
反应的双向 Top-K 检索，继续支持 zero-shot、few-shot、外部实体、可靠性、
受控候选扩展和湿实验闭环。没有把系统改造成固定 Rhea 分类器、通用酶注释
平台或生成式序列设计工具。

三条修改前冻结查询已逐字段回归通过；注册表单查询与批处理在 R2E/E2R 的
Top-3、Top-10、Top-20 六个组合中候选顺序、route ID 和 score source 全部
一致。

## 2. 生产路由契约

生产路由从路径和分支硬编码中抽出到：

```text
configs/production_routes/terpene_v1.yaml
```

每次结果新增 route ID/version、model bundle、candidate universe version/hash、
registry version。旧 `score_source`、模型目录和 CSV 列仍保留，已有消费者可
继续读取。

## 3. 输入与特征安全

外部反应增加 `strict|warn|fallback` 策略。DRFP 失败会形成明确审计状态；
strict 直接拒绝，warn/fallback 标记 `fallback_used`，不再静默伪装成正常
输入。外部蛋白记录长度、非法字符、模糊残基比例、低复杂度和序列 SHA-256。

ESM-C、模型 ensemble 和 schema 使用进程缓存；序列与反应特征使用内容寻址
缓存。缓存是可重建资产，不进入 Git。

## 4. 注册表事务

每次 init/add/remove 先在新目录构建完整蛋白矩阵、entries、metadata、反应表
和 manifest，校验后原子切换 `CURRENT`。切换完成后更新历史路径兼容镜像。
任何中断都不会让生产读取到一半新、一半旧的注册表。

当前服务器快照：

```text
registry-20260805T043639628709Z-30b5b663c65d
694 registered proteins
240 registered reactions
```

## 5. 可靠性绑定

六个外部 zero-shot 校准器绑定：

- route ID；
- model bundle version；
- 方向候选 ID 集合 SHA-256。

当前 R2E 候选集合哈希为
`1b5ae5a5d13b7448003a901f971158f6df53c6d7fcab46fbd7cc3b1f4d55681d`；
E2R 为
`039e83bbc7d2a91d68bd427a106dce61984ac02f77e9e1f53342efe23799653e`。
候选宇宙或路由变化后，旧校准器会被拒绝。

## 6. 统一调用接口

`rank_open_world.py` 暴露 `build_parser()` 和 `execute_ranking()`；
`core.engine.RetrievalEngine` 与 HTTP API 都直接调用它们。API 默认只读并禁止
模型路径覆盖，避免前端重新实现路由或任意切换生产模型。

## 7. 湿实验七态反馈

新增七态：未测试、控制失败、表达失败、检测不确定、表达合格但无目标产物、
目标产物阳性、其他产物阳性。只有目标阳性和表达合格阴性进入原有训练反馈；
控制失败、表达失败、检测不确定和其他产物观察均不会被错误写成负例。

## 8. 机制与时间评测 readiness

机制流程已生成 504 个 mechanism、18 个步骤类型维度、3395 个步骤和 79.99%
MARTS pair 覆盖的非学习基线。它只允许作为 residual/kernel 辅助研究，进入
生产仍需冻结 R2E 提升验证。

时间审计只有 195/2833 行可严格恢复有效年份，覆盖 6.88%，另有 22 行包含
未来数字 token。默认不生成 temporal split，也不允许据此报告时间外推指标。

## 9. 比赛证据层

在不改变生产 `score`、`rank` 和 route 的前提下，所有共享排名新增
`terpene-candidate-evidence-passport-v1`。查询级开放世界适用域由最近库相似度、
ensemble 共识、Top-K 稳定性、排名方差和边界分离度组成；候选级护照记录证据
强度、真实证据路径和警告。代理分数均明确标注为诊断证据，不解释为催化概率。

新增 `scripts/analyze_terpene_cycle_consistency.py`，可对 Top-N 候选运行反向生产
检索，检查 `reaction -> enzyme -> reaction` 或 `enzyme -> reaction -> enzyme`
闭环。外部实体通过临时只读候选扩展参与反向检查。循环 RRF 只写独立研究结果，
默认不进入生产排序。完整说明见
`docs/terpene_competition_evidence_layer_20260805_zh.md`。

## 10. 第二轮 Conformal Retrieval Sets

新增 `terpene-conformal-retrieval-sets-v1`，以 query-disjoint 双冷查询的最佳
正例归一化 rank 构建有限样本 split-conformal 候选集合。六个外部 zero-shot
校准器绑定 route ID、model bundle 和方向候选宇宙哈希。默认 `alpha=0.10`
且只注解集合大小；显式 `expand` 才扩展同一路由返回前缀。

90% 全局集合在 R2E 为 1,476–1,509 / 2,085 个蛋白，在 E2R 为
306–464 / 753 个反应。大集合被原样报告，不伪装成高精度小面板。第二轮循环
权重网格在 12 个注册已知关联代理查询上没有确认任何新增命中，因此生产 route
保持不变。完整报告见
`docs/terpene_second_round_conformal_cycle_20260805_zh.md`。

## 11. 验证入口

```bash
bash scripts/run_terpene_quality_gate.sh
bash scripts/run_terpene_quality_gate.sh --full
```

验证包括：85 项测试、portable runtime manifest、五个神经部署、双核资产、
route/reliability/conformal/candidate-set 健康、HTTP smoke、golden 回归、单/批
一致性，以及第二轮循环网格 smoke。
