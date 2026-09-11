# B题闭环原型：单源与第三问全向多源

这是一套已经运行通过的本地实验原型。适用于单个全向源定位，以及第三问的全向多源搜索清除。本轮没有运行官方模拟器，没有产生官方演练或正式成绩，第四问尚未接入。

## 文件

- `solver.py`：保守位置集合、接收与清除判据、两种选点策略、多源扫描及调度。
- `environment.py`：本地环境，模拟方位/near/no_signal、固定地点误差、移动、切频、清除和计时。
- `experiment.py`：固定随机种子的案例生成与公平比较。真值只由环境和事后评估读取，不传入策略。
- `checks.py`：规则边界、覆盖证书及特殊布局检查。
- `official_client.py`：按附件接口编写的本机HTTP适配器，待真实演练验证。
- `check_http.py`：本地替身HTTP服务集成验证，包含一次“动作已执行但响应丢失”的重试。
- `report.md`、`aggregate.json`：结果解读和汇总数据。
- `single-benchmark/`、`multi-benchmark/`：逐次动作、逐步可行域、真值、结果及汇总。
- `checks-initial-results.json`：保留初轮边界检查的问题记录；`checks-results.json`为修正测试数据后的结果。
- `figures/`：时间组成比较图与示例轨迹。

## 本地复现

解压后，在本目录打开终端。需要 Python 3.11 或较新版本。

```bash
python -m pip install -r requirements.txt
python experiment.py --kind single --seeds 40 --fields smooth,hash,extreme --out reproduced-single
python experiment.py --kind multi --seeds 20 --start-seed 1000 --fields smooth,hash,extreme --policies geometry,time --schedules immediate,deferred --out reproduced-multi
python checks.py
python check_http.py
```

同一版本的代码与 NumPy 下，虚拟任务时间及动作应可复现；现实运行时间随计算机负载变化。

## 官方演练接入

官方模拟器需要在使用者本机运行，并按附件登录竞赛账号。本次执行环境访问 `127.0.0.1:2026` 得到连接拒绝，因此未启动真实测试。

在模拟器中选择**问题3的演练测试**，等待接口就绪后，在同一台机器运行：

```bash
python official_client.py --robot-id 你的参赛队号 --policy time --schedule immediate --log practice-01.jsonl
```

下一局换一个日志名。程序使用 `/enter`、`/measure`、`/clear`、`/exit`；动作串行发送，网络重试复用相同request_id和payload。程序遵守clear不改变测向频道的约定。接收到拒绝响应时不将其中的0误记为当前虚拟时间。

`practice-01.jsonl`是程序自己的调试记录，不能替代模拟器导出的官方加密日志。此适配器通过的是本地替身服务验证，仍需要官方演练确认。不要把第三问算法直接用于含定向源的第四问。

## 限制

1. 自建源布局与误差场是测试条件，不是从官方生成器获得的分布。
2. 接收与清除使用整个保守可行域验证；选点评分采用离散场景和有限步展开，不是全局最优求解。
3. 几何圆盘以128边外接正多边形表示；没有通过内接近似错误排除真实位置。暂时不使用direction反馈中的5米排除空洞，也不使用失败光学清除缩小区域。
4. time策略对当前观测枚举−1°、0°、+1°三种误差，后续展开暂用零误差；展开三步后仍未清除时使用明示的启发式尾项。
5. 未发现频道在七个覆盖点全部得到无信号后才认定不存在；成功清除16个源时也可依据数量上限结束。局部定位设有30步上限，尚未证明任意合法场景都在此上限内终止。
6. 实验中的100%清除率只适用于列出的案例。第一、二问的几何事实、第三问覆盖证明和经验性能结果应在论文中分别表述。
