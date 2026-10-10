# BRIDGE：统一主实验表

训练关系 218,537；唯一验证来源 5,216 条关系；R2E 和 E2R 使用同一组 21,505 条正式测试关联。四类消融及 CAGE 门控组合均以当前冻结模型重新评价。Balanced 按四类暴露和节点度数层等权取原始宏平均。

### R2E：反应 → 酶 · Direct：自然分布
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.00522 | 0.49% | 0.99% | 2.46% |
| Broad Retrieval | 0.01413 | 1.22% | 2.22% | 7.39% |
| Broad Retrieval + CAGE Reranking | 0.00537 | 0.46% | 1.20% | 3.30% |
| CAGE Gate + BRIDGE Reranking | 0.01218 | 1.24% | 1.92% | 3.17% |
| BRIDGE | 0.02590 | 2.68% | 4.05% | 8.04% |
| 去除Functional | 0.02481 | 2.58% | 3.78% | 7.77% |
| 去除Structure/Mechanism | 0.02637 | 2.72% | 4.10% | 7.75% |
| 去除Long-term Relation Context | 0.01762 | 1.51% | 2.99% | 8.73% |
| 去除Family/Domain | 0.02589 | 2.68% | 4.04% | 8.03% |

### R2E：反应 → 酶 · Balanced：难度宏平均
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.06294 | 5.66% | 11.42% | 24.62% |
| Broad Retrieval | 0.09412 | 8.67% | 15.12% | 32.60% |
| Broad Retrieval + CAGE Reranking | 0.04056 | 4.56% | 8.96% | 19.75% |
| CAGE Gate + BRIDGE Reranking | 0.16423 | 17.47% | 23.60% | 29.87% |
| BRIDGE | 0.20441 | 22.94% | 31.13% | 40.56% |
| 去除Functional | 0.20374 | 22.95% | 31.07% | 40.44% |
| 去除Structure/Mechanism | 0.20146 | 22.10% | 30.88% | 40.01% |
| 去除Long-term Relation Context | 0.11772 | 10.18% | 21.84% | 35.06% |
| 去除Family/Domain | 0.20439 | 22.94% | 31.11% | 40.54% |

### E2R：酶 → 反应 · Direct：自然分布
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.07090 | 8.45% | 9.74% | 10.22% |
| Broad Retrieval | 0.03357 | 2.87% | 7.62% | 24.10% |
| Broad Retrieval + CAGE Reranking | 0.01810 | 2.21% | 3.53% | 4.16% |
| CAGE Gate + BRIDGE Reranking | 0.11286 | 12.94% | 14.94% | 15.20% |
| BRIDGE | 0.14980 | 15.80% | 25.50% | 45.42% |
| 去除Functional | 0.11524 | 12.35% | 18.30% | 39.94% |
| 去除Structure/Mechanism | 0.13370 | 13.82% | 22.63% | 45.19% |
| 去除Long-term Relation Context | 0.09727 | 9.65% | 18.28% | 40.33% |
| 去除Family/Domain | 0.14980 | 15.80% | 25.50% | 45.42% |

### E2R：酶 → 反应 · Balanced：难度宏平均
| 方法 | MRR | Hit@3 | Hit@10 | Hit@100 |
| --- | ---: | ---: | ---: | ---: |
| EnzymeCAGE | 0.08566 | 8.92% | 11.51% | 12.44% |
| Broad Retrieval | 0.06317 | 5.50% | 13.43% | 44.55% |
| Broad Retrieval + CAGE Reranking | 0.03246 | 3.70% | 5.26% | 6.26% |
| CAGE Gate + BRIDGE Reranking | 0.11113 | 11.72% | 14.21% | 14.29% |
| BRIDGE | 0.28897 | 30.26% | 40.34% | 60.69% |
| 去除Functional | 0.27668 | 29.33% | 37.65% | 57.05% |
| 去除Structure/Mechanism | 0.19998 | 20.38% | 29.86% | 60.10% |
| 去除Long-term Relation Context | 0.20449 | 20.70% | 32.60% | 54.62% |
| 去除Family/Domain | 0.28897 | 30.26% | 40.34% | 60.69% |

> 当前结果为测试信息已参与方案迭代后的探索性复评，尚需独立外部验证。
