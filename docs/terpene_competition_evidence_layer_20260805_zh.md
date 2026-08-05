# TerpeneNavigator 比赛证据层 v1（2026-08-05）

## 1. 设计目标

本层用于将经过验证的双向开放世界检索系统升级为可审计的科学决策系统，
而不是替换现有生产模型。所有扩展默认只追加证据字段，不改变原始 `score`、
`rank`、route、候选宇宙或可靠性校准结果。

当前落地三项：

1. Candidate Evidence Passport：候选证据护照；
2. Open-World Applicability：开放世界适用域；
3. Bidirectional Cycle Consistency：双向循环一致性分析。

三者均明确声明为排名诊断或证据强度代理，不解释为催化活性概率。

## 2. Candidate Evidence Passport

每个单查询、批处理和 HTTP/API 结果都带有版本化护照：

```text
terpene-candidate-evidence-passport-v1
```

查询级字段包括：

- `query_applicability_score`；
- `query_applicability_tier`；
- `query_applicability_recommendation`；
- 分解后的适用域组成；
- `diagnostic_applicability_not_activity_probability` 解释标签。

候选级字段包括：

- `candidate_evidence_score`；
- `candidate_evidence_tier`；
- `candidate_evidence_paths`；
- `candidate_evidence_warnings`；
- `evidence_strength_not_activity_probability` 解释标签。

候选证据等级为：

```text
priority_candidate
supported_candidate
review_candidate
exploratory_candidate
```

证据路径会按真实生产路线记录神经检索、RRF、邻域迁移、双核协同、ensemble
共识和已验证可靠性校准，不虚构未运行的机制或结构证据。

## 3. 开放世界适用域

适用域代理只使用现有查询诊断：

- 最近库实体相似度；
- ensemble Top-1 共识；
- ensemble Top-K 集合稳定性；
- ensemble Top-K 成员支持；
- Top-1 排名稳定性；
- Top-K 边界分离度。

其版本为：

```text
terpene-open-world-applicability-v1
```

输出等级：

```text
reference_library
in_domain
near_domain
weakly_supported
far_out_of_domain
```

它回答“模型对此查询有多少表示空间与排序稳定性支持”，不回答“候选发生反应
的概率是多少”。库内查询保留 `reference_library` 标签；注册表和真正外部查询
仍根据诊断进入不同难度等级。

真实 smoke 中：

- 当前反应 `RHEA:54512`：`reference_library`；
- 一个注册酶批查询：`near_domain`，score 约 0.774；
- 一个注册反应批查询：`in_domain`，score 约 0.887；
- 外部反应 `CCO>>CC=O`：`weakly_supported`，score 约 0.417。

这说明适用域不会把所有输入机械标成高可信。

## 4. 双向循环一致性

分析入口：

```bash
.venv/bin/python scripts/analyze_terpene_cycle_consistency.py \
  --direction reaction_to_enzyme \
  --reaction-id RHEA:54512 \
  --top-k 20 \
  --cycle-top-n 5 \
  --reverse-top-k 50 \
  --output results/terpene_cycle_consistency/example.csv
```

对于 `reaction_to_enzyme`：

```text
reaction -> candidate enzyme -> reverse-ranked reaction
```

对于 `enzyme_to_reaction`：

```text
enzyme -> candidate reaction -> reverse-ranked enzyme
```

真正 external query 会通过临时、只读候选扩展把原始外部实体加入反向候选
宇宙，因此可以检查语义闭环，而无需持久写入注册表。

输出包括：

- `cycle_reverse_rank`；
- `cycle_recovered`；
- `cycle_consistency_score`；
- 反向 route 和 score source；
- `cycle_reranked_rank`；
- 明确的非概率解释标签。

脚本还计算一个有界、研究用途的循环 RRF 排名。默认循环权重为 0.15，允许
范围为 0–0.25。它只写独立分析结果，`production_ranking_modified=false`；只有
在冻结验证集上确认提升后才允许考虑进入生产路线。

当前真实 smoke：

- `RHEA:54512` 正向前两名均在反向 Top-10 找回原反应；
- recovery fraction 为 1.0；
- 平均循环一致性为 0.9125；
- 外部 `CCO>>CC=O` 的正向首名在反向第 8 位找回原反应，循环分数约 0.806。

这些只是流程与语义闭环示例，不是总体性能声明。

## 5. 接口一致性

Evidence Passport 已接入：

- `rank_open_world.py` CLI CSV；
- CLI `.audit.json`；
- `core.engine.RetrievalEngine`；
- HTTP 排名响应；
- `rank_registry_batch.py` 两方向 ranking CSV；
- batch query summary CSV。

API 中查询证据位于：

```text
query.evidence_passport
```

候选证据位于：

```text
candidates[i].evidence_passport
```

## 6. 安全边界

当前版本保证：

- 不修改原始 score；
- 不修改原始 rank；
- 不改变自动路由；
- 不改变模型和候选集合；
- 不覆盖现有经验可靠性校准；
- 不把代理分数称为生化概率；
- 循环 rerank 只存在于独立实验脚本。

生产测试明确断言护照注解前后候选 ID、rank 和 score 完全相同。

## 7. 后续指标实验

下一阶段应在严格冻结 split 上比较：

1. 原始生产排名；
2. 0.05、0.10、0.15、0.20 循环权重；
3. 仅对 `near_domain`/`in_domain` 查询启用循环 rerank；
4. Top-30/50 内局部重排；
5. Hit@3/10/20、MRR、平均反向恢复率和计算成本。

只有预先指定权重在独立确认 split 上保持提升，才可新增 production route。
否则循环一致性继续作为解释、审计和候选优先级证据。
