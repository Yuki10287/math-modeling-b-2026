# 第三问：正负反馈与扫描前瞻

**状态：开发负结果。** 没有取得足以替换lean的稳定收益。

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

- [analyze_results.py](code/analyze_results.py): 结果分析、核查或绘图。
- [benchmark_exploration.py](code/benchmark_exploration.py): 实验或压力测试驱动。
- [bootstrap.py](code/bootstrap.py): 模型或运行依赖。
- [exploration_solver.py](code/exploration_solver.py): 模型或运行依赖。
- [make_probe_stress.py](code/make_probe_stress.py): 实验或压力测试驱动。
- [probe_policy.py](code/probe_policy.py): 模型或运行依赖。
- [scan_preview.py](code/scan_preview.py): 模型或运行依赖。
- [test_probe_policy.py](code/test_probe_policy.py): 本地行为检查。
- [test_scan_preview.py](code/test_scan_preview.py): 本地行为检查。
- [validate_probe_stress.py](code/validate_probe_stress.py): 实验或压力测试驱动。

## 使用了哪些前序成果

[第三问：time/lean](../00_主方案_lean/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q3_probe -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q3_probe --tests
```

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
