# 第三问：从模型改进到本地验证

本目录是一份独立的新版本；原始资料、初版代码和 `q3_improved` 上一版均保留。

**需要手动运行官方第三问测试：**先看 [官方测试操作说明](官方测试操作说明.md)，双击 `运行第三问.cmd`。入口已通过12项本地HTTP替身检查，未实际连接官方模拟器；冻结的求解模型没有更改。启动文件会等待你手动启动问题3并确认倒计时结束。

先读 **`本轮改进与实验结果.md`** 看整局结果，再读 **`模型推导与适用边界.md`** 看推导与仍然存在的模型局限。图表位于 `figures/`，逐局原始反馈和完整轨迹位于 `results/holdout-*/cases/`。

默认方案为 `lean`：持续维护所有已发现源的位置集合；利用固定未知接收半径产生的正负测点半平面；沿途筛选有价值的检测并共享测向；在当前位置即可保证清除的其他源及时清除；联合比较测向与有限光学覆盖；加入短横向测向候选；按剩余覆盖区域调整扫描点。最多16源的数量上界也参与搜索停止判断。

## 本地运行

运行环境只需 Python 和 NumPy。下列命令在本目录执行，仅运行自建环境，不连接官方模拟器。

```powershell
python -X utf8 benchmark.py --start-seed 1000 --seeds 1 --fields smooth --schedules baseline,v1,lean --split development --out results/my-local-check
python -X utf8 validate_model.py --output results/my-validation.json
```

`--out` 必须是尚不存在的目录，以免覆盖旧实验。输出中的 `total_s` 是按题目动作规则累计的虚拟时间，`runtime_s` 是本地程序计算的实际时间。

程序入口：

```python
from solver import solve_multi
trace = []
result = solve_multi(public_api, trace=trace)
```

接口只要求 `position`、`channel`、`measure(q, channel)`、`clear(q, channel)`。求解器不需要真实源位置、真实半径、源总数或环境评价结果。真正的完成判定看 `result['complete']` 和 `completion_certificate`，不能仅看某一条清除反馈。

上述整局程序与验证器可在本目录独立运行。`local_checks.py` 的393组单源开发分析额外读取相邻 `q3_improved` 中上一版的首次服务状态；单独拷贝本目录时，复现这部分分析还需保留上一版结果。现成单源记录已包含在本目录中。

## 文件与证据

| 文件 | 用途 |
|---|---|
| `solver.py` | 全局调度、沿途信息共享和完整任务入口 |
| `belief_model.py` | 正负观测、固定半径约束和完成证书 |
| `geometry.py` | v2独立几何，包含量化余量；不改原版几何 |
| `local_policy.py` | 测向/光学行动比较、短横向候选及覆盖检查 |
| `coverage_model.py`、`scan_planning.py` | 任意负测点的连续覆盖证据及自适应扫描点 |
| `recovery.py` | 局部异常或迭代上限时的有限光学后备 |
| `baseline_solver.py`、`v1_solver.py` | 初版与上一版的对照入口 |
| `benchmark.py`、`environment.py` | 同布局、同误差场的本地配对实验 |
| `validate_model.py` | 隐藏真值接口、独立计时、覆盖与几何验证 |
| `results/selection.json` | 开发选择依据和留出测试前的源码指纹 |
| `results/holdout_summary.json` | 冻结版本的完整留出集统计 |

`coverage/shared/optical/full/short/range/adaptive_short/adaptive_range` 等配置保留供模型拆分比较。`results/development`、`range-development`、`adaptive-development` 等为开发过程记录，期间其他文件可能继续编辑，部分标有 `code_hashes_unchanged=false`；正式结果采用最终冻结后的 `development-final` 和 `holdout-*`，不把探索中的单局最佳结果当作最终性能。

## 适用范围

本版仅适用于第三问的全向、固定源。第四问可能因辐射朝向返回无信号，不能直接使用这里的距离排除、正负半平面和1000米负观测覆盖证书。

局部评分中的面积样本和误差场是明确的设计/实验假设。完成性依赖题目约束与有效反馈，时间收益依赖布局及误差场；本地平均降低不等于每局都降低，也不等于官方评分保证。本目录未执行官方演练或正式测试。
