# 第四问：share25候选

**状态：第四问最终方案为独立的25站保序删点版本 `share25_prune`。** 入口为 `q4_share25_prune_client.py`，本批第八次演练72/72全清、平均5998.69秒；原share25第六批51/51全清、平均6594.44秒。两批不同场景，不能将均值差当作配对改善；`shared`保留为基线与备用版本。

本阶段的实验脚本、产物和说明保留在本目录；运行模块集中在上一级目录，下方链接指向其实际位置。文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [第八批保序删点官方演练分析](../03_官方演练与时间诊断/reports/官方演练第八批保序删点结果分析.md)：实测、条件费用核算、下一步模型方向及非凸细化无新增收益的诊断。
- [固定扫描站联合测向开发与留出实验](reports/固定扫描站联合测向开发与留出实验.md)：第八批后的本地研究；开发快0.92%，独立留出慢3.01%、最差慢22.28%，不晋升、不修改现用入口。
- [反馈分支清除费用模型本地实验](reports/反馈分支清除费用模型本地实验.md)：保留原10测点，补充负反馈后保序清除费用和未排除分支风险；5布局开发快0.68%、10布局留出快0.71%，留出5快5慢、最差慢6.90%，计算开销增加，保留研究、不接入现用入口。
- [两步反馈联合规划本地原型](reports/两步反馈联合规划本地原型.md)：在附近两个已知源、三个扫描站中试验反馈后的任务重选；开发五布局一步平均快3.72%，但独立留出十布局平均慢2.76%，两步慢3.12%，因此不接入客户端，保留为失败边界和后续模型研究材料。
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
