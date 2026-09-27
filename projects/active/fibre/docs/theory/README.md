# 催化相容核数学文档

`FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex` 当前承载催化相容核的中文数学讲义。文件名沿用历史路径，正文主理论已经回到生产模型真实结构：反应是有向化学变换，蛋白提供潜在催化能力，模型学习两者之间的任务层催化相容核。

正文依次给出物理催化语义、原始分子输入、需求—能力双线性配对、平方可积核的有限秩分解、多阳性双向排序目标、双未见可定义性、多基底专家、有界双线性修正和有限秩实验更新。

使用 XeLaTeX 编译：

```bash
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
xelatex -interaction=nonstopmode -halt-on-error FIBRE_INTERACTION_ATLAS_THEORY_ZH.tex
```

生成的 PDF 为构建产物，仓库中的规范源文件为 `.tex`。
