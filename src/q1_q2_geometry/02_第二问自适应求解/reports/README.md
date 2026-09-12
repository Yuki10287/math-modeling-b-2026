# 第一问几何核与第二问独立求解

整理后请从[本阶段导航](../README.md)阅读，命令均从项目根目录执行，使用根目录的启动入口。

第二问现在有完整的独立流程：**首测输入 → 选择第二测点 → 输入实际二测反馈 → 输出定位区域、直径和最小包围圆**。它只处理几何定位，不调用官方接口，也不执行移动、检测或清除。

论文以[问题一至三建模算法初稿](../../../../docs/论文整理/问题一至三建模算法论文初稿.md)为准。第二问采用纯几何目标，第三问在共享几何结构上引入任务时间。

## 文件对应

| 文件 | 用途 |
|---|---|
| [geometry.py](../../00_第一问几何核/code/geometry.py) | 第一问几何核，另保留早期预设测点示例 |
| [q2_solver.py](../code/q2_solver.py) | 第二问独立程序；复用第一问几何核，无第三问运行依赖 |
| [test_q2_solver.py](../code/test_q2_solver.py) | 数学不变量、异常反馈、原策略回归与命令行检查 |
| [benchmark_q2.py](../code/benchmark_q2.py) | 固定布局和误差场的二次观测专项对照 |
| [examples](../examples) | 可直接运行的首测和二测输入示例 |
| [第二问专项验证](第二问专项验证.md) | 新对照结果、配对胜负及适用边界 |
| [历史预设测点实验说明](../../01_早期预设测点实验/reports/历史预设测点实验说明.md) | 原7个预设点及63个敏感性案例，保留历史口径 |

旧 `results.json` 不被新程序改写。其时间列属于历史上对第三问的探索，不用于第二问评价。

## 运行示例

在项目根目录使用 Python 3.11 或更新版本：

```powershell
python -m pip install -r src/q1_q2_geometry/00_第一问几何核/code/requirements.txt
python -X utf8 tools/run_study.py q12_current --script q2_solver.py -- plan --input src/q1_q2_geometry/02_第二问自适应求解/examples/first_observation.json --output tmp/q2_example/plan.json
python -X utf8 tools/run_study.py q12_current --script q2_solver.py -- update --plan tmp/q2_example/plan.json --input src/q1_q2_geometry/02_第二问自适应求解/examples/second_observation.json --output tmp/q2_example/result.json
```

本地示例首点为 `(0,0)`、首测示向度为 `0°`；所选第二点约为 `(758.739150,-487.574260)`。示例二测读数为 `63.67°`，得到直径约 **51.99 米**、包围圆半径约 **25.99 米**。二测示例只用于复现，不能当作真实测试反馈。

输出文件拒绝覆盖，重跑请使用新输出文件名。返回码 `0` 表示正常完成；`1` 表示写出了观测矛盾或无候选等结果；`2` 表示输入、读写或数值计算错误。

## 输入自己的观测

首测 JSON：

```json
{"position": [0, 0], "status": "direction", "angle": 0}
```

查看计划中的 `selected_point`，在该点取得反馈后填写二测 JSON。`position` 必须与计划点一致（容差 `1e-6` 米）；`angle` 为实际示向度，以度为单位、从 x 轴正向逆时针，程序按模360处理。

- 正常示向：提供 `position`、`status: "direction"` 和 `angle`。
- 近距离提示：提供 `position` 和 `status: "near"`。
- 无信号：提供 `position` 和 `status: "no_signal"`。

首测为 `near` 时直接输出几何位置约束，无需二测；首测 `no_signal` 不满足本程序的第二问输入条件。计划点已经通过接收认证，二测若出现 `no_signal`，输出 `inconsistent` 说明矛盾，不把它当作定位成功。

`update` 从公开首测重建区域，不信任保存计划里可被修改的顶点。不要把错误或缺失的反馈自行填成正常示向。

## 输出含义

`plan` 输出首测区域 `region`、可靠候选区域的数学表达、通过认证的有限 `candidates`、各点预测直径 `geometry_score_m`，以及选中点 `selected_point`。有限候选是候选区域内的离散采样，不是整个连续区域。

`update` 输出：

- `region.vertices`：按边界顺序给出的保守定位多边形顶点；
- `region.diameter_m`：区域直径；
- `region.enclosing_circle`：最小包围圆圆心及半径（浮点数值解）；
- `diameter_ratio`：二测直径与首测直径的比值；
- `status`：正常示向、近距离或观测矛盾等状态。

近距离反馈同时输出精确的 `near_constraint`（圆心及5米半径）。多边形是圆盘的外逼近，直径可能略大于10米；不能把这个离散结果误读为违反近距离提示。

## 算法与假设

正常首测区域由原点1800米源位置圆、半角1°测向楔形、首点1500米接收圆相交得到。圆用128边外接正多边形近似，正常示向中的5米排除空洞暂不扣除。外逼近非空不证明原精确约束一定相容；外逼近为空则能够识别矛盾。

利用“接收半径至少1000米”和“首测已成功”检查可靠接收：新点到某个可能源更近时沿用首测成功证据，否则要求仍在1000米保证范围内。检查包括裁剪半平面产生的新顶点，采用999.999米余量。

默认 `geometry` 沿用原型18点构造、最多5个顶点加顶点均值、误差 `-1°/0°/+1°`。以有限情景中最大的预测后验直径评分，同分选择距离较短的点。没有速度或动作费用参数。预测只加入新楔形；实际更新再加入1500米接收约束，近距离反馈改用5米约束。

对照策略共用同一候选集：`midpoint` 选最接近“包围圆圆心加0.3倍半径侧向偏移”的候选；`nearest` 选离首点最近的候选。通过 `plan --policy midpoint` 或 `--policy nearest` 使用。它们都不读取真值。

这是有限离散的几何优化，没有连续全局最优保证。若候选构造未找到可用点，返回 `no_candidate`，不宣称问题无可行解；直径不超过 `1e-8` 米按数值退化直接返回 `localized`。

128边圆的单边界最大外扩为 `R(sec(π/128)-1)`：1500米圆约0.452米、1800米圆约0.542米、5米圆约0.00151米。这不是定位交集误差的统一上界。旧预设示例用720边，精度口径不同。

第二问采用题设1°；第三问冻结的工程实现用 `1.0050001°` 兼容先加误差再保留两位小数的反馈。新本地实验用不超过0.99°的固定误差后再舍入，保证反馈误差仍在1°内。

## 验证与复现

```powershell
python -X utf8 tools/run_study.py q12_current --tests
python -X utf8 tools/run_study.py q12_current --script benchmark_q2.py -- --output-dir tmp/q2_benchmark_repeat
```

基准目录必须不存在。它只产生本地观测并评价二测后几何结果，真值只用于反馈生成和独立审计。保存结果见[专项验证报告](第二问专项验证.md)。

第一问历史检查可调用 `geometry.checks()`。运行 `geometry.py` 主入口会重新生成旧 `results.json`；新第二问程序不需要执行该入口。
