# 第四问：混合源覆盖与联合路线

先看[代码导读](代码导读.md)了解主流程和文件分工；日常运行使用[项目根目录的运行第四问](../../运行第四问.cmd)。

当前推荐方案为 **`joint_solver.py` 的 `shared` 配置**：25站双环覆盖、扫描与已知源的联合路线、扫描停点的信息复用。第四问独立启动入口已准备好，见[官方演练操作说明](第四问官方演练操作说明.md)；接入验证仅使用本地HTTP替身，尚无第四问官方成绩。第三问lean的15个冻结文件保持一致。

先读[覆盖与联合路线改进](第四问覆盖与联合路线改进.md)。位置、类型、半径、朝向的相容约束以及光学后备原理仍见[首版模型说明](第四问模型与本地验证.md)。原31站版本和历史结果保留作对照。

最新验证采用12种新布局、每布局两种固定误差场，共24组配对条件。两版均全清，新版每组更快，平均整局时间由9941.76降至6848.07秒，下降31.12%；合并单源时间由811.57降至559.03秒/源。边界组平均下降28.36%。这些是本地证据，不是官方演练成绩或任意场景保证。

随后完成[30组压力条件、14项HTTP检查和3项阈值检查](第四问压力测试与接口验证.md)，全部通过，求解模型与上述留出验证的源码保持一致。压力数据单独归档，不并入性能均值。

## 本地运行

需保留相邻的 `q3_model_v2` 文件夹，以读取冻结的几何模块和自建环境。Python 3.11或更新版本，在项目根目录执行：

```powershell
python -m pip install -r src/q4_model/requirements.txt
python -X utf8 -m unittest discover -s q4_model -p test_model.py -v
python -X utf8 -m unittest discover -s q4_model -p test_joint.py -v
python -X utf8 src/q4_model/benchmark_joint.py --out src/q4_model/results/my-joint-check --start-seed 18000 --layouts 1 --populations uniform --fields smooth --variants baseline,shared
```

输出目录必须尚不存在。以上测试不连接网络或官方模拟器。样例是复现运行方法，不是新的独立性能结论；用于调参的数据今后应归入开发资料。

## 文件

| 文件 | 用途 |
|---|---|
| [q4_official_client.py](q4_official_client.py)、[运行第四问.cmd](运行第四问.cmd) | 由使用者启动的第四问接入程序与独立完成核验 |
| [verify_release.py](verify_release.py)、[发布记录](results/client_release.json) | 15个文件的只读版本核查 |
| [stress_check.py](stress_check.py)、[综合验证记录](results/readiness_analysis.json) | 确定性压力集合、误差端点与接口验证 |
| [polar_cover.py](polar_cover.py) | 当前25站、36三角形的连续方向覆盖 |
| [joint_solver.py](joint_solver.py)、[route_planning.py](route_planning.py) | 当前联合路线与扫描停点复用 |
| [benchmark_joint.py](benchmark_joint.py)、[test_joint.py](test_joint.py) | 新版配对实验与四项针对性核查 |
| [results/joint_selection.json](results/joint_selection.json)、[新验证汇总](results/joint_holdout_analysis.json) | 当前配置选择、冻结指纹及24组结果 |
| [guarded_policy.py](guarded_policy.py) | 未入选的类型保守评分，保留作对照 |
| [directional_cover.py](directional_cover.py) | 连续三角网、负测点凸包与不存在证书 |
| [localization.py](localization.py) | 混合类型情景、矩形覆盖和主动测向评分 |
| [solver.py](solver.py) | 原31站整局闭环，保留对照 |
| [validation.py](validation.py) | 独立反馈计时、真源包含性和完整覆盖验证 |
| [benchmark.py](benchmark.py) | 合成案例、公开接口包装与配对实验 |
| [test_model.py](test_model.py) | 六项定向几何、退化审计及接口边界检查 |
| [results/selection.json](results/selection.json) | 新测试前的参数选择和源码指纹 |
| [results/analysis.json](results/analysis.json) | 配对变化、费用分解及审计规模 |
| [results/audit_revision.json](results/audit_revision.json) | 测试后补强退化线段复核，30份轨迹重查结果不变 |

求解器调用形式为 `from joint_solver import solve_multi`，然后 `solve_multi(public_api, variant='shared', trace=trace)`，公开接口形式与初版一致。自行开展官方演练时使用第四问入口，并保存官方源总数与两份客户端日志。
