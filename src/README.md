# 按版本和实验找代码

各阶段保留实验脚本、结果、图表和报告。第四问的运行模块集中在 `q4_model/` 根目录，共用的第三问几何与通信模块在 `q3_model_v2/` 根目录；具体文件见各组导航。编号用于阅读顺序；是否采用，以“当前主方案”“候选”“历史对照”等状态为准。

| 题目 | 当前使用 | 完整过程 |
|---|---|---|
| 第一、二问 | [第二问自适应求解](q1_q2_geometry/02_第二问自适应求解/README.md) | [3个阶段](q1_q2_geometry/README.md) |
| 第三问 | [lean主方案](q3_model_v2/00_主方案_lean/README.md) | [8个阶段](q3_model_v2/README.md) |
| 第四问 | [25站share25_prune最终方案](q4_model/05_候选share25/README.md) | [6个阶段](q4_model/README.md) |

例如打开第四问的 [share25候选](q4_model/05_候选share25/README.md)，便能同时看到：

```text
05_候选share25/
  README.md     本阶段做了什么、当前地位、如何运行
  code/         实验脚本、版本检查及本地检查
  reports/      模型改进、接入验证及候选演练说明
  results/      开发、留出、压力与接口实验
  figures/      对应图表
```

运行源码只保留一份。第四问可按[命令行说明](q4_model/README.md)直接运行客户端；实验脚本仍通过 `tools/run_study.py` 准备依赖。根目录CMD入口继续可用，不要直接执行组内历史CMD/PS1。

论文稿及跨题目的材料仍放在[论文导航](../docs/README.md)，第三问更早的两份完整版本见[历史版本](../archive/README.md)。逐文件旧路径对应保存在 `studies.json` 中，由工具读取；新增顶层运行模块时需同步登记，以便原工具继续复现实验。
