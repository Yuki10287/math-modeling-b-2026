# 2026数学建模B题求解

**第一次接手论文，请先读[论文交接指南](docs/论文交接指南.md)。** 指南给出原题与附件、阅读顺序、当前版本、数据口径、第四问整合材料和正式测试待办；不需要此前聊天记录。

现在按**题目 → 版本或实验阶段**找文件。同一次研究的代码、实验结果、图表和说明放在同一个文件夹中。

| 从哪里开始 | 内容 |
|---|---|
| [原题与两个附件](docs/题目材料/README.md) | 原件、便于搜索的文本副本及文件校验清单 |
| [第一、二问](src/q1_q2_geometry/README.md) | 第一问几何核、早期预设测点、当前第二问自适应求解 |
| [第三问](src/q3_model_v2/README.md) | lean主方案、局部选点、联合调度、成本预测、全局价值、官方复盘、反馈与闭环预测 |
| [第四问](src/q4_model/README.md) | 公共几何、初版31站、shared基线、官方复盘、22站实验、share25_prune最终方案 |
| [论文和总体结论](docs/README.md) | 跨题目的论文稿、总体版本选择、证据索引与文献复盘 |
| [早期完整版本](archive/README.md) | 第三问最初原型与上一版完整程序 |

例如，第四问 `05_候选share25` 里面就能找到该候选的求解代码、配对实验、压力结果、接口验证和候选说明，不用再跨总的src、docs目录寻找。

## 日常运行

- 第二问从[独立求解示例](src/q1_q2_geometry/02_第二问自适应求解/reports/README.md)开始。
- [运行第三问.cmd](运行第三问.cmd)：当前time/lean主方案。
- [运行第四问.cmd](运行第四问.cmd)：第四问最终方案 `share25_prune`。
- [运行第四问候选.cmd](运行第四问候选.cmd)：独立share25候选，先看[候选说明](src/q4_model/05_候选share25/reports/第四问share25候选演练说明.md)。

第三、四问由你手动开启相应官方演练，再按窗口提示开始。个人日志保存在 `local_data/official_runs/`。请保留完整项目，使用根目录入口；组内的历史启动文件仅用于版本溯源。

## 本地检查与实验

Python 3.11或更新版本，在项目根目录执行：

```powershell
python -m pip install -r src/q4_model/02_主方案25站_shared/code/requirements.txt
python -X utf8 tools/check_structure.py
python -X utf8 tools/run_study.py q12_current --tests
python -X utf8 tools/run_study.py q3_main --tests
python -X utf8 tools/run_study.py q4_main --tests
```

这些检查只使用本地文件和测试服务。各组README给出对应实验入口；运行方式及新增实验方法见[工具说明](tools/README.md)。

## 文件保存与同步

求解代码在各组 `code` 中修改，实验产物在同组 `results`、`figures` 中查找，报告在同组 `reports` 中阅读。`tmp` 是自动生成的运行目录和工作缓存，无需编辑。

代码、报告和已保存的实验分析纳入Git。题目PDF在项目根目录，两个原始附件及检索文本在 `docs/题目材料/`，随仓库同步。个人日志、官方加密日志、结果压缩包和其他本地缓存不进入Git；重新分析这些原始日志时需要队内另行传递对应文件。另一台电脑获取完整仓库后，用 `git pull --ff-only` 更新即可。只阅读论文材料无需安装Python。

目录迁移细节与验证见[整理说明](docs/目录调整说明.md)，全阶段导航见[src/README](src/README.md)。
