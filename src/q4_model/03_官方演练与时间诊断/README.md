# 第四问：官方演练与时间诊断

**状态：实测复盘。** 本组同时保存旧5次和新4次演练的代码、数据、报告与诊断。两批均为shared；share25候选的材料见[候选阶段](../05_候选share25/README.md)。

## 按批次查找

| 批次 | 运行版本与结果 | 报告 | 原始分析产物 |
|---|---|---|---|
| 旧批：测试结果(2).zip，5次 | shared；63/63清除；合并503.51秒/源 | [五次结果分析](reports/官方演练五次结果分析.md) | [逐动作与汇总](results/official_q4_20260911_analysis.json)、[对应图表](figures/official_q4_practice_costs.png) |
| 新批：测试结果(4).zip，4次 | shared；50/50清除；合并504.33秒/源 | [第四批结果分析](reports/官方演练第四批结果分析.md) | [逐动作与汇总](results/official_q4_20260912_batch4_analysis.json)、[条件回放诊断](results/official_q4_20260912_batch4_replay_detail.json) |

新批平均整局6304.11秒。5000秒参考针对整局；旧[400秒目标评估](results/target400_assessment.json)针对单源，二者不能混用。不同批次布局不同，均值变化不能当作算法改进幅度。

新批前三次严格回放通过，第四次在两处等价路线选择上沿用日志方向后通过条件核查；条件核查与严格复现分别记录。具体证据和局限见新批报告。

## 代码与产物对应

| 代码 | 用途和对应产物 |
|---|---|
| [analyze_official_archive.py](code/analyze_official_archive.py) | 读取用户保存的ZIP，核对版本、计时、几何证据、源数和严格回放；生成新批逐动作与汇总JSON |
| [replay_fourth.py](code/replay_fourth.py) | 新批单场等价路线回放、轨迹与证书核对；生成回放补充JSON。原生成脚本见[历史快照](results/source_snapshots/batch4_replay/README.md) |
| [analyze_official_results.py](code/analyze_official_results.py) | 原五次演练分析及共享回放、几何核查方法；生成旧批汇总JSON |
| [assess_time_target.py](code/assess_time_target.py) | 旧批400秒/源目标的费用与结构分析 |
| [plot_official_results.py](code/plot_official_results.py) | 旧批五次结果图表；图表仍对应旧批，未用新批替换 |

新批结果和报告单独保存，旧数据及图表保持原有口径。完整文件列表见[结果导航](results/README.md)。

## 本地复核

在项目根目录执行。查看新批分析程序的参数：

```powershell
python -X utf8 tools/run_study.py q4_official --script analyze_official_archive.py -- --help
```

本机原始包存放在 `local_data/q4_official_20260912_batch4/测试结果(4).zip`，属于个人原始材料，不进入Git。保存的去身份化分析结果和报告进入Git。

如果本机保留该包，可以复核四次记录，输出使用一个不存在的新文件：

```powershell
python -X utf8 tools/run_study.py q4_official --script analyze_official_archive.py -- --zip "local_data/q4_official_20260912_batch4/测试结果(4).zip" --source-counts 13 11 15 11 --omni-counts 5 1 1 7 --directional-counts 8 10 14 4 --output "src/q4_model/03_官方演练与时间诊断/results/my-batch4-review.json"
```

这些数量对应本批01—04，来源于用户补充；不得用于其他测试包。该命令只分析已保存日志，不启动或连接官方模拟器，也不改写求解策略。若复核旧批目录格式，显式选择 `--script analyze_official_results.py`。

单独复核新批第04次的条件回放：

```powershell
python -X utf8 tools/run_study.py q4_official --script replay_fourth.py -- --zip "local_data/q4_official_20260912_batch4/测试结果(4).zip" --analysis "src/q4_model/03_官方演练与时间诊断/results/official_q4_20260912_batch4_analysis.json" --case q4-practice-04 --output "src/q4_model/03_官方演练与时间诊断/results/my-batch4-replay.json"
```

原始包、成员指纹和动作数量须与分析文件对应，输出须不存在；核验失败时返回非零。两份新结果已通过上述整理后入口离线复现，见[归档复核记录](results/batch4_organization_check.json)。

求解与公共几何仍依赖[25站shared主方案](../02_主方案25站_shared/README.md)和其共享模块，由运行工具准备。
