# 第三问：time/lean

**状态：当前主方案。** 官方运行保持time/lean；包含模型、主实验、接口与验证。

这里把本阶段的代码、实验产物和说明放在一起；文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [官方测试操作说明](reports/官方测试操作说明.md)
- [本轮改进与实验结果](reports/本轮改进与实验结果.md)
- [模型推导与适用边界](reports/模型推导与适用边界.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段源代码和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 早期记录或顶层验证汇总 | [records/](records/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [audit_holdout_replays.py](code/audit_holdout_replays.py): 结果分析、核查或绘图。
- [baseline_solver.py](code/baseline_solver.py): 模型或运行依赖。
- [belief_model.py](code/belief_model.py): 模型或运行依赖。
- [benchmark.py](code/benchmark.py): 实验或压力测试驱动。
- [coverage_checks.py](code/coverage_checks.py): 模型或运行依赖。
- [coverage_model.py](code/coverage_model.py): 模型或运行依赖。
- [create_release.py](code/create_release.py): 模型或运行依赖。
- [environment.py](code/environment.py): 模型或运行依赖。
- [geometry.py](code/geometry.py): 模型或运行依赖。
- [local_policy.py](code/local_policy.py): 模型或运行依赖。
- [official_client.py](code/official_client.py): 接口程序。
- [package_official.py](code/package_official.py): 模型或运行依赖。
- [plot_results.py](code/plot_results.py): 结果分析、核查或绘图。
- [recovery.py](code/recovery.py): 模型或运行依赖。
- [scan_planning.py](code/scan_planning.py): 模型或运行依赖。
- [solver.py](code/solver.py): 模型或运行依赖。
- [summarize_release.py](code/summarize_release.py): 结果分析、核查或绘图。
- [test_client.py](code/test_client.py): 本地行为检查。
- [v1_solver.py](code/v1_solver.py): 模型或运行依赖。
- [validate_model.py](code/validate_model.py): 模型或运行依赖。
- [verify_main_solution.py](code/verify_main_solution.py): 版本与证据校验。
- [write_report.py](code/write_report.py): 模型或运行依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q3_main -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q3_main --tests
```

官方演练使用项目根目录的[运行第三问.cmd](../../../运行第三问.cmd)，由你手动开启对应模拟器演练。个人日志统一保存在 `local_data/official_runs/`。

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
