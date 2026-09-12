# 第三问：time/lean

**状态：当前主方案。** 官方运行保持time/lean；包含模型、主实验、接口与验证。

本阶段的实验脚本、产物和说明保留在本目录；正式运行需要的8个求解模块，以及供第四问复用的几何与通信模块，集中在上一级目录，下方链接指向其实际位置。文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [官方测试操作说明](reports/官方测试操作说明.md)
- [本轮改进与实验结果](reports/本轮改进与实验结果.md)
- [模型推导与适用边界](reports/模型推导与适用边界.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 正式运行模块 | [上一级第三问目录](../README.md)，入口为 [official_client.py](../official_client.py) |
| 本阶段实验脚本和本地检查 | [code/](code/) |
| 直接命令行入口回归 | [tests/test_direct_entry.py](../tests/test_direct_entry.py) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 早期记录或顶层验证汇总 | [records/](records/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [audit_holdout_replays.py](code/audit_holdout_replays.py): 结果分析、核查或绘图。
- [baseline_solver.py](../baseline_solver.py): 模型或运行依赖。
- [belief_model.py](../belief_model.py): 模型或运行依赖。
- [benchmark.py](code/benchmark.py): 实验或压力测试驱动。
- [coverage_checks.py](code/coverage_checks.py): 模型或运行依赖。
- [coverage_model.py](../coverage_model.py): 模型或运行依赖。
- [create_release.py](code/create_release.py): 模型或运行依赖。
- [environment.py](code/environment.py): 模型或运行依赖。
- [geometry.py](../geometry.py): 模型或运行依赖。
- [local_policy.py](../local_policy.py): 模型或运行依赖。
- [official_client.py](../official_client.py): 接口程序。
- [package_official.py](code/package_official.py): 模型或运行依赖。
- [plot_results.py](code/plot_results.py): 结果分析、核查或绘图。
- [recovery.py](../recovery.py): 模型或运行依赖。
- [scan_planning.py](../scan_planning.py): 模型或运行依赖。
- [solver.py](../solver.py): 模型或运行依赖。
- [summarize_release.py](code/summarize_release.py): 结果分析、核查或绘图。
- [test_client.py](code/test_client.py): 本地行为检查。
- [v1_solver.py](../v1_solver.py): 模型或运行依赖。
- [validate_model.py](code/validate_model.py): 模型或运行依赖。
- [verify_main_solution.py](code/verify_main_solution.py): 版本与证据校验。
- [write_report.py](code/write_report.py): 模型或运行依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看直接入口的参数（只显示帮助，不代表完整运行已通过）：

```powershell
python -X utf8 src/q3_model_v2/official_client.py --help
```

本地检查：

```powershell
python -X utf8 -m unittest discover -s src/q3_model_v2/tests -v
python -X utf8 tools/run_study.py q3_main --tests
```

由队员手动启动问题3、等待接口就绪后，可直接运行（会连接模拟器）：

```powershell
python -X utf8 src/q3_model_v2/official_client.py --robot-id YOUR_TEAM_ID --log NEW_LOG.jsonl
```

替换队号与本次新日志名；日志父目录须存在，不能覆盖已有日志。此入口直接加载同目录模块，不依赖临时目录。项目根目录的[运行第三问.cmd](../../../运行第三问.cmd)仍保留，它会生成 `local_data/official_runs/q3/` 下的新日志；详细流程见[操作说明](reports/官方测试操作说明.md)。助手仅运行本地检查，不自行连接官方模拟器。

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
