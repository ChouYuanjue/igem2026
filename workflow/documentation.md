# Portable BRIDGE workflow

该 Snakemake 工作流重建 BRIDGE 运行所需的便携数据与基础特征。它服务于 Broad candidate space 和可选专家输入准备，不复活历史 FIBRE interaction-atlas 方法。

默认配置：`config/bridge.example.yaml`。

```bash
snakemake -s workflow/Snakefile --configfile config/bridge.example.yaml --cores 4
```

环境文件位于 `workflow/envs/bridge-*.yaml`。运行设备优先读取 `BRIDGE_DEVICE`；旧 `FIBRE_DEVICE` 仅作为迁移兼容变量。特征缓存优先读取 `BRIDGE_FEATURE_CACHE`。

历史工作流源码已保存于 `archive/fibre/20261003/tooling/`。
