# FIBRE 数学建模文档

`FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex` 保留历史文件名，正文现已重写为当前冻结 FIBRE 检索器的直接数学说明。

主线只包含真实进入模型的对象：

- 反应 `multiview` 输入（冻结维度 8270）；
- ESM-C 600M mean 蛋白输入（1152 维）；
- 128 维通用交互分量与 8 个 32 维专家分量；
- R2E / E2R 查询侧门控；
- 簇屏蔽的多阳性双向对比损失；
- Top-3 / Top-10 / Top-20 排名 surrogate；
- gate balance、低熵、专家 diversity 和 0.02 overlap consistency；
- legacy、strict double-cold 与 185,918 蛋白广域消融结果。


使用 XeLaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
```
