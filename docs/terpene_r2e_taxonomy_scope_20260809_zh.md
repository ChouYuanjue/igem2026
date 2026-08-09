# R2E 酶候选 Taxonomy Scope：设计、实现与验证

日期：2026-08-09
状态：已部署到模型服务器；默认工作流保持不变
适用方向：**仅 Reaction → Enzyme（R2E）**

## 1. 目标

在原有开放世界 R2E 检索之外，增加三种可明确选择的酶候选范围：

- `all`：正常生产候选宇宙；
- `eukaryote`：只允许真核酶进入预测；
- `prokaryote`：只允许原核酶进入预测。

这不是在最终 Top-K 上做前端过滤，而是在模型打分之前改变候选蛋白矩阵：

```text
Reaction query
  → reaction representation
  → deployed protein universe (2,085)
  → taxonomy sieve
       all         → 2,085
       eukaryote   → 1,340
       prokaryote  →   180
  → production route/model
  → uncertainty diagnostics
  → ranked candidates
```

E2R 不存在酶候选池，因此**没有** taxonomy scope 控件、query gate 或额外 route suffix；E2R 生产语义保持不变。

## 2. 为什么使用候选宇宙约束，而不是训练三套模型

真核/原核限制描述的是实验对象范围，而不是一个新的预测任务。将其设计为 candidate-universe constraint 有几个优点：

1. 复用已经冻结验证的生产模型，不引入三套难以维护的权重；
2. 路由、模型分数和候选范围三者职责分离；
3. restriction 在打分前发生，所有 direct、few-shot、ensemble disagreement、CAGE rescue 等后续逻辑看到的是同一个受限候选集合；
4. `candidate_universe_hash` 会随 scope 改变，可被校准器绑定检查捕获；
5. 默认 `all` 完全保持原生产结果，可由 golden regression 直接证明无回归。

## 3. Taxonomy registry

版本：

```text
terpene-enzyme-taxonomy-scope-v1
```

文件：

```text
data/terpene_taxonomy_scope/protein_taxonomy_scope.csv
data/terpene_taxonomy_scope/summary.json
```

可复现生成器：

```text
scripts/prepare_terpene_taxonomy_scope.py
```

### 3.1 本地数据源

生成器只使用项目中已经固定的本地数据，不在查询时联网：

1. 生产候选清单：
   `results/terpene_production_models/marts_adapted_drfp_pu/protein_registry.csv`
2. MARTS 酶 kingdom/species 元数据：
   `data/terpene_marts/marts_enzymes.tsv`
3. 已有本地 UniProt TPS 快照：
   `data/terpene_uniprot_expansion/uniprot_tps_normalized.tsv`

### 3.2 分类优先级

优先使用 MARTS 中与 production ID 直接对应的 kingdom 标签。

对于没有直接 kingdom 的生产蛋白，只在以下条件同时满足时使用本地 UniProt organism 做保守补齐：

- accession 在本地 UniProt TPS 快照中存在；
- organism 的 genus 可以在 MARTS 中找到；
- 该 genus 在 MARTS 中只对应一个 kingdom；
- 存在 kingdom 冲突的 genus 不进行推断。

目前检测到 `scytonema` genus 存在冲突，因此被排除在 genus inference 之外。

每个 registry row 都记录：

- `taxonomy_scope`
- `kingdom`
- `species`
- `taxonomy_source`
- `taxonomy_confidence`

因此前端和审计文件可以说明某一候选为什么被纳入，而不是只有一个不可追踪的布尔标记。

## 4. 分类定义与候选数量

### 4.1 真核

以下 MARTS kingdom family 被归入 `eukaryote`：

- Plantae
- Fungi
- Animalia（所有当前子标签）
- Amoebozoa

数量：**1,340 / 2,085**。

### 4.2 原核

以下类别被归入 `prokaryote`：

- Bacteria
- Archaea
- Cyanobacteria

数量：**180 / 2,085**。

### 4.3 Other 与 Unknown

- Viruses：`other`，6 条；不会为了方便被塞入原核。
- 无支持分类：`unknown`，559 条。

在 restricted scope 中，`other` 与 `unknown` 都默认排除。

这意味着系统宁可缩小候选空间，也不会用蛋白名称、长度、motif 或模型 embedding 去“猜”其真核/原核属性。

## 5. Candidate universe hash

三个当前固定集合的 SHA-256 identifier-set hash：

```text
all         2,085  1b5ae5a5d13b7448003a901f971158f6df53c6d7fcab46fbd7cc3b1f4d55681d
eukaryote   1,340  0c2fec494d7f1c5008eab20d0ca05023fc495adb528e0742136c215a6519e73e
prokaryote    180  16c9d28d6b65815b7f36eb54a233f8a6bde149acf496d4859ca2e44ad8df8f9d
```

受限模式不仅 route ID 不同，candidate-universe hash 也不同。

## 6. R2E 在线执行语义

CLI/API 参数：

```text
enzyme_taxonomy_scope = all | eukaryote | prokaryote
```

执行顺序：

1. 装载 current protein embeddings；
2. 合并 registered protein embeddings；
3. 若存在 temporary external candidate，则先合并；
4. 根据 taxonomy registry 计算可保留 index；
5. restricted mode 直接切片 `protein_features` 与 `protein_ids`；
6. 后续模型只看到切片后的矩阵；
7. few-shot similarity、direct score、ensemble uncertainty、排序与 CAGE candidate selection 都使用同一受限 ID 集合；
8. provenance 用受限 ID 集合重新生成 candidate-universe hash 与 size。

因此：

```text
post-hoc filter = false
pre-score candidate restriction = true
```

## 7. Few-shot 组合规则

Few-shot 与 taxonomy restriction 可以组合，例如：

```text
r2e-current-top10-v1+fewshot+eukaryote-only
```

但种子必须属于选定的生物域：

- eukaryote scope + eukaryotic seed：允许；
- eukaryote scope + prokaryotic seed：拒绝；
- restricted scope + locally unknown seed：拒绝。

原因是使用一个被 scope 排除的 seed 去定义相似度空间，会制造“候选只限真核、指导样本却来自原核”的不透明混合协议。

## 8. Route provenance

受限模式产生显式后缀：

```text
+eukaryote-only
+prokaryote-only
```

示例：

```text
r2e-current-top10-v1+eukaryote-only
r2e-external-top10-v1+prokaryote-only
r2e-current-top10-v1+fewshot+eukaryote-only
r2e-external-top3-v1+masked+prokaryote-only
```

其中最后一类来自 registry batch discovery：已知 enzyme association 会在批处理中屏蔽，因此同时出现 `+masked` 与 taxonomy suffix。

## 9. Calibration 边界

### 9.1 Empirical reliability

现有 empirical reliability calibrator 是在 unrestricted candidate universe 和锁定 external double-cold 协议上验证的。

因此 restricted R2E external query 返回：

```text
empirical_reliability_status = not_applicable_taxonomy_restricted
```

不输出伪造的“活性概率”或未验证 reliability tier。

### 9.2 Conformal Retrieval Set

现有 conformal set 同样绑定：

- route ID
- model bundle
- exact candidate-universe hash

restricted mode 改变了 route 与 candidate universe，因此返回：

```text
conformal_status = not_applicable_taxonomy_restricted
conformal_binding_status = not_applicable
```

前端会禁用 restricted R2E 的 conformal target 控件，并解释需要独立校准。

未来如果希望给真核/原核模式恢复 conformal coverage，需要分别建立 query-disjoint calibration panel，而不是把原 2,085 候选的 qhat 按比例缩放。

## 10. Batch workflow

`rank_registry_batch.py` 增加 R2E 专用参数：

```text
--r2e-enzyme-taxonomy-scope all|eukaryote|prokaryote
```

E2R batch 不读取该参数。

restricted R2E batch 强制使用新的 `--output-dir`；如果试图覆盖 canonical：

```text
results/terpene_registry_batch
```

程序直接拒绝执行。

这样 canonical unrestricted batch、eukaryote batch 与 prokaryote batch 可以并列比较而不会互相覆盖。

## 11. Frontend 设计

R2E Query Composer 新增独立步骤：

```text
03 · Which enzyme sources may participate?
```

三种入口：

- All enzymes — 2,085
- Eukaryotes only — 1,340
- Prokaryotes only — 180

E2R 切换后这个步骤完全消失。

### 11.1 Route Atlas

所有 R2E route 在 candidate universe 与 strategy router 之间加入：

```text
Taxonomy Sieve
```

模块视觉不是普通矩形占位，而表现为：

```text
many candidate particles
       → sieve
       → retained particles
```

运行 restricted route 时动态 metric：

```text
2085 → 1340 · eukaryote only
2085 → 180 · prokaryote only
```

### 11.2 Evidence banner

restricted R2E 结果顶部显式显示：

- pre-filter candidate size
- post-filter candidate size
- excluded count
- unknown count
- calibration unavailable explanation

候选表同时展示 candidate kingdom 与 taxonomy source 对应的 biological-domain 信息。

## 12. 全路由展示完整性

Route Board 当前展示：

- 12 条 manifest 基础生产路由；
- 2 条 few-shot execution path；
- R2E / E2R mask 相关路径；
- R2E / E2R temporary-universe；
- R2E / E2R manual override；
- R2E eukaryote-only；
- R2E prokaryote-only；
- conditional CAGE rescue。

总可见路径：**23**。

其中 R2E registry `+masked` 路径被单独标注为 `BATCH WORKFLOW`，避免误导为交互式 portal 控件。

## 13. 已完成验证

### 13.1 真实 API

当前 `RHEA:54512` Top-10：

```text
all:
  route = r2e-current-top10-v1
  universe = 2085

eukaryote:
  route = r2e-current-top10-v1+eukaryote-only
  universe = 1340
  excluded = 745

prokaryote:
  route = r2e-current-top10-v1+prokaryote-only
  universe = 180
  excluded = 1905
```

真实外部 reaction `CCO>>CC=O` + eukaryote：

```text
route = r2e-external-top10-v1+eukaryote-only
empirical_reliability_status = not_applicable_taxonomy_restricted
conformal_status = not_applicable_taxonomy_restricted
```

### 13.2 Batch smoke

1 query / Top-3：

```text
eukaryote:
  r2e-external-top3-v1+masked+eukaryote-only
  2085 → 1340

prokaryote:
  r2e-external-top3-v1+masked+prokaryote-only
  2085 → 180
```

返回候选全部属于对应 scope。

### 13.3 Frozen regression

默认 `all`：

- 三条 frozen golden route 全通过；
- E2R/R2E Top-3/10/20 单查询—批处理候选、route、score source 全一致。

因此默认生产行为没有改变。

### 13.4 Headless frontend E2E

只使用 headless Playwright，无 GUI 测试。

验证：

- 23 route cards；
- 5 route groups；
- 2 taxonomy path cards；
- E2R taxonomy controls = 0；
- eukaryote route 与 `2085 → 1340` visual metric 正确；
- prokaryote route 与 `2085 → 180` visual metric 正确；
- few-shot + eukaryote 同时点亮 taxonomy 与 seed module；
- 10/10 eukaryote candidates biological domain 正确；
- 10/10 prokaryote candidates biological domain 正确；
- mobile document width = viewport width = 390；
- console errors = 0；
- page errors = 0；
- request failures = 0。

## 14. 已知边界

1. 559 个 production proteins 仍为 locally unknown；restricted mode 选择保守排除。
2. 当前不在查询时访问 UniProt/NCBI，因此结果可复现，也不会因远端 taxonomy 数据更新而静默改变。
3. temporary external enzyme 如果没有进入 pinned taxonomy registry，在 restricted mode 下会被视为 unknown 并排除。
4. restricted mode 尚没有单独的 empirical-reliability 或 conformal calibrator。
5. 本功能解决的是候选生物域约束，不等同于表达宿主适配、亚细胞定位、密码子偏好或实验可构建性；这些应作为后续实验效用层的独立维度。

## 15. 后续可选升级

只有在需要进一步扩大已分类覆盖时，才建议引入新的 pinned taxonomy snapshot：

- 固定日期的 UniProt taxonomy export；或
- 固定版本 NCBI Taxonomy dump。

升级应生成新的 taxonomy-scope version 和新的 candidate hashes，并重新评估 restricted double-cold panel。不要在在线查询路径中调用远端 taxonomy API 后直接改变候选集合，否则无法保证比赛演示与实验批次可复现。
