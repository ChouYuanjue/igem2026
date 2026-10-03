# BRIDGE portable configuration

`config/bridge.example.yaml` 是当前便携工作流示例。它用于重建基础候选表示、Broad 检索所需输入和可选证据，不定义历史 FIBRE 交互图册。

当前方法说明见：

- `projects/active/bridge/docs/method.md`
- `projects/active/bridge/docs/engineering.md`
- `projects/active/bridge/docs/reproducibility.md`

旧 FIBRE portable workflow 的精确副本已经归档到 `archive/fibre/20261003/tooling/`。

运行：

```bash
snakemake -s workflow/Snakefile \
  --configfile config/bridge.example.yaml \
  --cores 4
```

配置只声明数据、特征和构建资源。专家是否具有当前查询的排序权限，由 BRIDGE runtime 的适用性与准入合同决定，不由 portable 配置静态指定。
