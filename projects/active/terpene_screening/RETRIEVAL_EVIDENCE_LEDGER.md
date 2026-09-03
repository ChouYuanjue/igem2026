# Catalyst Retrieval Evidence Ledger

这份文件只管理**数字的身份**，不再按实验历史叙述。

## 先纠正三个混淆

1. `RF/CAGE rescue` 是 Catalyst 自己的历史 TPS hybrid，**不是外部 baseline**。
2. Rhea128→141 表里的 base/anchor 也是 Catalyst 内部基线，**不是外部模型 baseline**。
3. EnzymeCAGE 论文表、EnzGFM 论文表如果没有本地复现，只能标 `paper-only`；不能与本地实测混成一种证据。

## 当前证据总账

| 场景 | 对照身份 | 是否双方本地实测 | 是否同 support | 能否直接做 baseline delta | 结论 |
|---|---|---:|---:|---:|---|
| TPS 513×1391 日用 R2E | 旧 RF/HGB+CAGE hybrid | 是 | 是 | **否：这是我们自己的旧系统** | 只能证明内部迭代；外部 baseline 缺失 |
| TPS pure EnzymeCAGE | raw EnzymeCAGE probability | 是 | **否，coverage 不完整** | 否 | 513-query end-to-end Hit@10 12.09%，native-evaluable 361-query Hit@10 17.17%；只能做覆盖/迁移诊断 |
| TPS strict R2E | fold-local RF/HGB+CAGE hybrid | 是 | 是 | **否：内部旧系统** | 8.96/16.81% vs 0.56/1.54% 是内部进步，不是外部 superiority |
| TPS strict E2R | previous production route | 是 | 是 | 内部 delta 可以 | 独立确认 Hit@20 34.77→43.37%，MRR 0.0764→0.0874；属于生产路线改进 |
| TPS exact-entity visibility | 同一系统 visibility slices | 是 | 是 | 不需要 | 能力切片，不应强塞 baseline |
| TPS few-shot | 3-mer vs ESM-C similarity | 是 | 是 | 可以，但这是简单算法基线，不是外部模型 | 适合展示同源扩展/远缘扩展能力 |
| Enzyme-405 complete226 | **official EnzymeCAGE seeds40–44** | **是** | **是** | **是** | 当前最干净纯 EnzymeCAGE 外部对比；CAGE SR@10 51.33±1.66%，Catalyst 49.12%；MRR/MAP 则 Catalyst 略高 |
| Enzyme-405 full295 | EnzymeCAGE paper table | CAGE 否 / Catalyst 是 | 协议对应但非本地同跑 | 否 | paper-only context |
| Orphan-335 | author Selenzyme score | **是** | **是** | **是** | 干净的 reaction-novel R2E 外部对比，但 positive-protein novelty/coverage 不足，属 secondary |
| CLIPZyme common support | official CLIPZyme checkpoint | **是** | **是** | **是** | 干净本地 baseline alignment，但 common support/adapted cell，属 secondary |
| EnzGFM native ReactZyme | EnzGFM-1.5B paper mean | baseline 否 / Catalyst 是 | split 对齐 | 不做 paired/local delta | paper-only；且是 seen-reaction sequence-divergence |
| Rhea128→141 general | Catalyst internal base/anchor | 是 | 是 | 只能做内部 delta | 主外部绝对泛化结果；不要把 anchor 叫外部 baseline |
| common CAGE reservoir + broad neural | pure CAGE vs integrated broad neural | 是 | 是 | **不用于公平泛化** | 神经 Hit@10≈98–99% 是 exposure/recovery 诊断，主表忽略 |


## TPS 候选宇宙必须分代

TPS 历史上存在两套不同 universe：

- **旧 current-library：513 reactions × 1,391 proteins**，主要用于数据库补全、exact-entity、早期 strict R2E；
- **当前 MARTS：453 reactions × 1,421 proteins**，用于 domain adaptation 与 `confirmatory20260726` 的双向 strict 评测。

不能把旧 universe 的 R2E 与新 universe 的 E2R 拼成一张“对称 TPS 双向表”。当前 1,421×453 universe 上两方向均有本地测量：R2E `adapted_direct` 在 233 query-cells 上 Hit@10 **6.87%**、Hit@20 **16.74%**、MRR **0.03985**；E2R 在 279 query-cells 上已有 finalized fusion，Hit@20 **34.77%→43.37%**、MRR **0.0764→0.0874**。因此当前真正的缺口是：**R2E 尚无与 E2R 同等级的 finalized route confirmation**。

## 真正缺漏

- **TPS 513×1391 R2E：缺完整 same-support 外部 baseline。** 纯 CAGE 本地有分数，但 support 不完整；不能用 RF/CAGE 顶替。
- **TPS strict double-cold R2E：缺 aligned external baseline。**
- TPS strict E2R 没有外部 baseline，但已有独立内部确认，足够支撑“路线改进”，不是外部 SOTA claim。

## 可以忽略/降级

- pure CAGE v1/v2：和 v3 重复。
- Enzyme-405 早期 100-query reconstruction：被 complete226 取代。
- legacy broad-Rhea projected cold：相对 current clean2023 已不冷。
- common-reservoir 98–99% neural recovery：明显是 integrated-association recovery，不是外部泛化。

## 最完善、对称的外部对比

1. **Enzyme-405 complete226: EnzymeCAGE vs Catalyst，双方本地实测、同 support。**
2. **Orphan-335: author Selenzyme vs Catalyst，同 author pool。**
3. **CLIPZyme common support: official checkpoint vs Catalyst，同 score matrix。**

其中只有第 1 个是纯 EnzymeCAGE；它不能冒充 TPS 513×1391 baseline。
