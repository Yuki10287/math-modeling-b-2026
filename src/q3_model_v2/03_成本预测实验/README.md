# 第三问：成本预测

**状态：未晋升的候选。** 研究预测费用与实际开销的差距。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [成本预测诊断与本地验证](reports/成本预测诊断与本地验证.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [analyze_cost_diagnostic.py](code/analyze_cost_diagnostic.py): 结果分析、核查或绘图。
- [audit_forecast_results.py](code/audit_forecast_results.py): 结果分析、核查或绘图。
- [benchmark_forecast.py](code/benchmark_forecast.py): 实验或压力测试驱动。
- [diagnose_cost.py](code/diagnose_cost.py): 结果分析、核查或绘图。
- [forecast_policy.py](code/forecast_policy.py): 模型或运行依赖。
- [forecast_solver.py](code/forecast_solver.py): 模型或运行依赖。
- [plot_forecast_results.py](code/plot_forecast_results.py): 结果分析、核查或绘图。
- [summarize_forecast.py](code/summarize_forecast.py): 结果分析、核查或绘图。
- [test_forecast.py](code/test_forecast.py): 本地行为检查。

## 使用了哪些前序成果

[第三问：time/lean](../00_主方案_lean/README.md)、[第三问：联合调度](../02_联合调度实验/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q3_cost -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q3_cost --tests
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
