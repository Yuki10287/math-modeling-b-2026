# 第四问：share25候选

**状态：已新增独立的25站保序删点候选，完成本地客户端验证。** 新入口为 `q4_share25_prune_client.py`，日志标记 `share25_prune`。原share25的第六批官方演练51/51全清、平均6594.44秒；新删点入口尚无官方实测成绩。

本阶段的实验脚本、产物和说明保留在本目录；运行模块集中在上一级目录，下方链接指向其实际位置。文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [25站保序删点候选运行说明](reports/25站保序删点候选运行说明.md)：本次新增的独立客户端，日志版本为share25_prune。
- [整体模型继续改进实验汇总](reports/整体模型继续改进实验汇总.md)：新22站、保序删点及独立本地验证；原入口继续保留。
- [新22站覆盖与整局配对](reports/新22站覆盖与整局配对实验.md)
- [保序删除必失败光学点](reports/保序删除必失败光学点开发实验.md)
- [第六批官方候选结果分析](../03_官方演练与时间诊断/reports/官方演练第六批share25结果分析.md)
- [第六批结果后的模型实验汇总](reports/第六批结果后的模型实验汇总.md)
- [连续朝向半径积分评分开发实验](reports/连续朝向半径积分评分开发实验.md)
- [光学矩形长宽比开发实验](reports/光学矩形长宽比开发实验.md)
- [细分覆盖与服务停点扫描本地实验](reports/细分覆盖与服务停点扫描本地实验.md)
- [第四问share25候选演练说明](reports/第四问share25候选演练说明.md)
- [第四问模型改进实验](reports/第四问模型改进实验.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段实验脚本和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [assess_task_candidate.py](code/assess_task_candidate.py): 结果分析、核查或绘图。
- [benchmark_task_sharing.py](code/benchmark_task_sharing.py): 实验或压力测试驱动。
- [plot_cell_experiment.py](code/plot_cell_experiment.py): 结果分析、核查或绘图。
- [prepare_share25_release.py](code/prepare_share25_release.py): 模型或运行依赖。
- [q4_share25_client.py](../q4_share25_client.py): 接口程序。
- [stress_task_sharing.py](code/stress_task_sharing.py): 实验或压力测试驱动。
- [task_sharing_solver.py](../task_sharing_solver.py): 模型或运行依赖。
- [test_share25_client.py](code/test_share25_client.py): 本地行为检查。
- [validate_share25_http.py](code/validate_share25_http.py): 模型或运行依赖。
- [verify_share25_release.py](code/verify_share25_release.py): 版本与证据校验。

## 使用了哪些前序成果

[第四问：25站shared](../02_主方案25站_shared/README.md)、[第四问：22站覆盖](../04_22站覆盖实验/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q4_share -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q4_share --tests
```

官方演练可使用[第四问命令行](../README.md)，或项目根目录的[运行第四问候选.cmd](../../../运行第四问候选.cmd)，由你手动开启对应模拟器演练。CMD日志保存在 `local_data/official_runs/`；命令行日志保存到 `--log` 指定位置。

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
