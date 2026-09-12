# 第四问：25站shared

**状态：当前主方案。** 双环覆盖、联合路线与停点复用；含接口和压力验证。

本阶段的实验脚本、产物和说明保留在本目录；运行模块集中在上一级目录，下方链接指向其实际位置。文件夹序号表示研究过程，不代表性能排名。

## 先看说明

- [第四问压力测试与接口验证](reports/第四问压力测试与接口验证.md)
- [第四问官方演练操作说明](reports/第四问官方演练操作说明.md)
- [第四问覆盖与联合路线改进](reports/第四问覆盖与联合路线改进.md)

## 文件对应

| 内容 | 位置 |
|---|---|
| 本阶段实验脚本和本地检查 | [code/](code/) |
| 模型说明、实验报告和操作说明 | [reports/](reports/) |
| 逐批实验、冻结选择与结果汇总 | [results/](results/) |
| 本阶段图表 | [figures/](figures/) |

主要文件：

- [benchmark_joint.py](code/benchmark_joint.py): 实验或压力测试驱动。
- [joint_solver.py](../joint_solver.py): 模型或运行依赖。
- [local_http_fixture.py](code/local_http_fixture.py): 模型或运行依赖。
- [polar_cover.py](../polar_cover.py): 模型或运行依赖。
- [q4_official_client.py](../q4_official_client.py): 接口程序。
- [stress_check.py](code/stress_check.py): 实验或压力测试驱动。
- [summarize_joint_q4.py](code/summarize_joint_q4.py): 结果分析、核查或绘图。
- [summarize_readiness.py](code/summarize_readiness.py): 结果分析、核查或绘图。
- [test_joint.py](code/test_joint.py): 本地行为检查。
- [test_q4_client.py](code/test_q4_client.py): 本地行为检查。
- [test_thresholds.py](code/test_thresholds.py): 本地行为检查。
- [verify_release.py](code/verify_release.py): 版本与证据校验。

## 使用了哪些前序成果

[第四问：公共几何与定位](../00_公共几何与定位/README.md)、[第四问：31站初版](../01_初版31站/README.md)。源代码只保留一份，运行工具自动准备这些依赖。

## 怎么运行

在项目根目录执行。不要直接双击 `code` 内的历史启动文件或把一个实验组单独复制出去。

查看本组主要脚本的参数：

```powershell
python -X utf8 tools/run_study.py q4_main -- --help
```

本地检查：

```powershell
python -X utf8 tools/run_study.py q4_main --tests
```

官方演练可使用[第四问命令行](../README.md)，或项目根目录的[运行第四问.cmd](../../../运行第四问.cmd)，由你手动开启对应模拟器演练。CMD日志保存在 `local_data/official_runs/`；命令行日志保存到 `--log` 指定位置。

原始记录中的旧路径用于溯源，运行工具保留其相对路径含义；查看文件时使用本页链接。新实验输出应指定本组 `results` 下的新目录，不覆盖历史批次。
