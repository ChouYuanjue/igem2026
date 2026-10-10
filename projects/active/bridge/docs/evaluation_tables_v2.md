# BRIDGE：广域关系评测

正式训练截止关系图为 clean2023 的 218,537 条催化关联。E2R 的查询门控开发集包含 3,900 条学习关系与 1,316 条内部验证关系。R2E 和 E2R 共同使用 21,505 条训练期未见正式测试关系。Direct 为关系级平均，Balanced 为四类新颖度与各自节点度数层等权的原始宏平均。

完整的 EnzymeCAGE、Broad、CAGE 门控路径、BRIDGE 以及四组证据消融在同一正式测试集上的统计表见 [统一主表](evaluation_tables_v3.md)，评测协议及复现入口见 [评测说明](evaluation.md)。
