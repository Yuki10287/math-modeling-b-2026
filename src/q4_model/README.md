# 第四问：版本与实验过程

运行模块集中在本目录，实验脚本、结果、图表和报告仍保留在下表各阶段目录中。各组README链接到同一份运行源码。

| 阶段 | 当前地位 | 研究内容 |
|---|---|---|
| [00_公共几何与定位](00_公共几何与定位/README.md) | 共享基础 | 各版本共用的几何、定位、路线与独立审计。 |
| [01_初版31站](01_初版31站/README.md) | 历史对照 | 混合源首版覆盖与局部服务策略。 |
| [02_主方案25站_shared](02_主方案25站_shared/README.md) | 当前主方案 | 双环覆盖、联合路线与停点复用；含接口和压力验证。 |
| [03_官方演练与时间诊断](03_官方演练与时间诊断/README.md) | 实测复盘 | 两批shared官方记录（5次与4次）、时间开销及目标差距；含新批次回放诊断。 |
| [04_22站覆盖实验](04_22站覆盖实验/README.md) | 研究对照与负结果 | 更紧覆盖、情景费用等探索；未入选主方案。 |
| [05_候选share25](05_候选share25/README.md) | 下一轮候选 | 服务停点共享与任务恢复；本地优于shared，未自动替换。 |

默认入口保持25站shared。share25是独立候选入口；22站实验和负结果没有混入默认方案。

在仓库根目录打开命令行。手动启动官方问题4演练并等待接口就绪后，运行对应客户端：

**share25候选：**

```powershell
python -X utf8 src/q4_model/q4_share25_client.py --robot-id 202619002152 --log q4-share25-practice-01.jsonl
```

**shared主方案：**

```powershell
python -X utf8 src/q4_model/q4_official_client.py --robot-id 202619002152 --log q4-shared-practice-01.jsonl
```

队伍编号需与模拟器登录一致；每次演练使用新的日志编号。命令行日志写入 `--log` 指定的位置，相对路径以当前命令目录为准；同时保留同名加 `.beliefs.json` 的轨迹文件。端口默认2026，其他端口可追加 `--url http://127.0.0.1:端口`。

根目录的[运行第四问候选.cmd](../../运行第四问候选.cmd)和[运行第四问.cmd](../../运行第四问.cmd)继续可用，会先核验发布文件。命令行运行前也可按各组操作说明进行只读版本核验。

本目录两个客户端及其运行依赖须与相邻的 `src/q3_model_v2` 一起保留。实验、压力测试和版本检查脚本仍通过 `tools/run_study.py` 运行，例如：

```powershell
python -X utf8 tools/run_study.py q4_share -- --help
python -X utf8 tools/run_study.py q4_main --tests
```
