# BRIDGE Family/Domain specialist slice

口径：仅统计生产路径中 Family/Domain 专家实际激活的 R2E query；Before 为 BRIDGE - Family/Domain，After 为完整 BRIDGE。MRR/Hit 使用每个 query 的最佳 held-out positive rank，与 R2E direct 主表一致。

| Expert slice | Activated q | Held-out edges | Query MRR Before -> After | Delta MRR | Hit@10 Before -> After | Hit@100 Before -> After | Edge rank changes (up/down) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P450 family-CAGE | 10 | 106 | 0.11674 -> 0.11674 | +0.00000 | 10.00% -> 10.00% | 40.00% -> 40.00% | 0 / 1 |
| Phosphatase family-CAGE | 1 | 1 | 0.00167 -> 0.00167 | +0.00000 | 0.00% -> 0.00% | 0.00% -> 0.00% | 0 / 0 |
| Terpene family-CAGE | 0 | 0 | -- | -- | -- | -- | 0 / 0 |
| TPS specialist | 11 | 13 | 0.01047 -> 0.02820 | +0.01773 | 0.00% -> 18.18% | 18.18% -> 36.36% | 7 / 0 |
| **All activated Family/Domain** | **22** | **120** | **0.05837 -> 0.06724** | **+0.00887** | **4.55% -> 13.64%** | **27.27% -> 36.36%** | **7 / 1** |

说明：P450 的 query-level 最佳正例排名未变化，但 106 条 held-out 关系中有 1 条由 rank 16 变为 17；TPS specialist 在 13 条 held-out 关系中改善 7 条、无恶化，并使其 11 个激活 query 的 MRR 从 0.01047 提升到 0.02820。
