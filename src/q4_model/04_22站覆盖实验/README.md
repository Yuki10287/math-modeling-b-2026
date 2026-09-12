# 第四问：22站覆盖

**状态：研究对照与负结果。** 更紧覆盖、情景费用等探索；未入选主方案。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

本组入口和文件用途见下表；结论以保存的实验汇总为准。

22站与后续share25的比较过程共同记录在[模型改进实验](../05_候选share25/reports/第四问模型改进实验.md)。

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |

主要文件：

- [analyze_cell_experiment.py](code/analyze_cell_experiment.py): 结果分析、核查或绘图。
- [benchmark_cells.py](code/benchmark_cells.py): 实验或压力测试驱动。
- [cell_cover.py](code/cell_cover.py): 模型或运行依赖。
- [cell_solver.py](code/cell_solver.py): 模型或运行依赖。
- [cell_validation.py](code/cell_validation.py): 模型或运行依赖。
- [context_policy.py](code/context_policy.py): 模型或运行依赖。
- [stress_cells.py](code/stress_cells.py): 实验或压力测试驱动。
- [test_cell_cover.py](code/test_cell_cover.py): 本地行为检查。
- [test_context_policy.py](code/test_context_policy.py): 本地行为检查。

## 使用了哪些前序成果

[第四问：25站shared](../02_主方案25站_shared/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q4_cells -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q4_cells --tests
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
