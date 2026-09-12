# 第三问：闭环预测

**状态：探索阶段。** 闭环情景与预测诊断，尚未替换主方案。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [本阶段原始说明](reports/README.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |

主要文件：

- [benchmark_closed.py](code/benchmark_closed.py): 实验或压力测试驱动。
- [benchmark_conditional.py](code/benchmark_conditional.py): 实验或压力测试驱动。
- [bootstrap.py](code/bootstrap.py): 模型或运行依赖。
- [closed_preview.py](code/closed_preview.py): 模型或运行依赖。
- [conditional_preview.py](code/conditional_preview.py): 模型或运行依赖。
- [conditional_worlds.py](code/conditional_worlds.py): 模型或运行依赖。
- [diagnose_conditional.py](code/diagnose_conditional.py): 结果分析、核查或绘图。
- [diagnose_forecasts.py](code/diagnose_forecasts.py): 结果分析、核查或绘图。
- [rollout_solver.py](code/rollout_solver.py): 模型或运行依赖。
- [scenario_model.py](code/scenario_model.py): 模型或运行依赖。
- [summarize.py](code/summarize.py): 结果分析、核查或绘图。
- [test_closed_loop.py](code/test_closed_loop.py): 本地行为检查。
- [test_conditional_preview.py](code/test_conditional_preview.py): 本地行为检查。
- [test_conditional_worlds.py](code/test_conditional_worlds.py): 本地行为检查。
- [test_scenario_model.py](code/test_scenario_model.py): 本地行为检查。
- [timing_replay.py](code/timing_replay.py): 模型或运行依赖。

## 使用了哪些前序成果

[第三问：time/lean](../00_主方案_lean/README.md)、[第三问：正负反馈与扫描前瞻](../06_正负反馈与扫描前瞻/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q3_closed -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q3_closed --tests
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
