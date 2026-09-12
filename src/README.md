# 按版本和实验找代码

每个阶段都是一个实际文件夹，里面同时放代码、结果、图表和报告。编号用于阅读顺序；是否采用，以“当前主方案”“候选”“历史对照”等状态为准。

| 题目 | 当前使用 | 完整过程 |
|---|---|---|
| 第一、二问 | [第二问自适应求解](q1_q2_geometry/02_第二问自适应求解/README.md) | [3个阶段](q1_q2_geometry/README.md) |
| 第三问 | [lean主方案](q3_model_v2/00_主方案_lean/README.md) | [8个阶段](q3_model_v2/README.md) |
| 第四问 | [25站shared主方案](q4_model/02_主方案25站_shared/README.md) | [6个阶段](q4_model/README.md) |

例如打开第四问的 [share25候选](q4_model/05_候选share25/README.md)，便能同时看到：

```text
05_候选share25/
  README.md     本阶段做了什么、当前地位、如何运行
  code/         候选求解、实验脚本、接口及本地检查
  reports/      模型改进、接入验证及候选演练说明
  results/      开发、留出、压力与接口实验
  figures/      对应图表
```

共享几何代码在前序或公共组中只保留一份。各组README列出依赖，项目运行工具自动准备它们。不要直接执行组内历史CMD/PS1，日常使用项目根目录的入口。

论文稿及跨题目的材料仍放在[论文导航](../docs/README.md)，第三问更早的两份完整版本见[历史版本](../archive/README.md)。逐文件旧路径对应保存在 `studies.json` 中，由工具读取；通常无需手工编辑。
